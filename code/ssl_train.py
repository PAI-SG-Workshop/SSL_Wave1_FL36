"""
ssl_train.py - SSL Wave 1a: self-supervised pre-training of ResNet-18 on FashionLetters36 (no labels).

Methods (solo-learn CIFAR recipes, MIT, used as the "official-equivalent" small-data source):
    simclr  NT-Xent t 0.2, projector 512-2048-256,                  LARS lr 0.4  wd 1e-4, symmetric aug
    byol    projector 512-4096(BN)-256 + predictor 256-4096(BN)-256, EMA tau 0.99 -> 1, LARS lr 1.0 wd 1e-5, asymmetric
    dino    head 512-2048-2048-256 (GELU) + 4096 prototypes (norm last layer), student t 0.1, teacher t 0.07,
            center momentum 0.9, EMA tau 0.9995 -> 1, last layer frozen for epoch 0, LARS lr 0.3 wd 1e-6, asymmetric
    vicreg  expander 512-2048(BN)-2048(BN)-2048, weights 25 / 25 / 1, LARS lr 0.3 wd 1e-4, rrc min 0.2, solarize 0.1
All: batch 256, LARS momentum 0.9 eta 0.02 clip_lr, no LARS / wd on 1-D params, warmup 10 epochs from 3e-5 then
cosine to 0 (per step), fp32, seed 777. Two 96 x 96 views per sample.

[Unified] differences from the recipes
  - backbone: torchvision ResNet-18 with the ImageNet stem (= the CenterNet ResNet-18; keys match net.* there)
  - input: grayscale -> 3 identical channels, x / 255, no mean / std (same as the detection pipeline)
  - views: a square region (side 128-256 px, at least 2 % bright pixels, up to 10 tries) is drawn from a 608 x 608
    image; each view is a RandomResizedCrop of that region resized to 96. No horizontal flip (letters are not
    mirror-invariant). Saturation / hue / grayscale are no-ops on grayscale data and are skipped.
  - DINO uses 2 global views (as the solo-learn CIFAR recipe), no multi-crop.
  - 1 epoch = --samples-per-epoch region samples (default 32,768 = 8 per image).

Usage (from C:\\SSL_Wave1\\code\\ssl, env ssl_env):
    python ssl_train.py --method simclr --bench 30            # throughput / memory / temperature, no files kept
    python ssl_train.py --method simclr --epochs 1 --smoke    # gate 3: one short epoch + collapse metric + kNN
    python ssl_train.py --method simclr --epochs 100          # training (resumes from runs/<method>/last.pth)
    python ssl_train.py --eval-only --init random             # kNN + linear probe of a random-init ResNet-18
    python ssl_train.py --eval-only --init C:\\SSL_Wave1\\runs\\ssl_simclr\\backbone_final.pth
The log ends with "DONE <method>" (or "GATE FAIL" / "STOP") so a queue can wait on the log line.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.environ.get("PHYSICAL_AI_ROOT", r"C:\SSL_Wave1"))
DATA = ROOT / "datasets" / "ssl"
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"

RECIPES = {
    "simclr": dict(lr=0.4, wd=1e-4, aug="symmetric"),
    "byol": dict(lr=1.0, wd=1e-5, aug="asymmetric", tau=0.99),
    "dino": dict(lr=0.3, wd=1e-6, aug="asymmetric", tau=0.9995),
    "vicreg": dict(lr=0.3, wd=1e-4, aug="vicreg"),
}
# per view: rrc min scale, brightness, contrast, jitter prob, blur prob, solarize prob   (solo-learn cifar yaml)
AUGS = {
    "symmetric": [(0.08, 0.8, 0.8, 0.8, 0.5, 0.0)] * 2,
    "asymmetric": [(0.08, 0.4, 0.4, 0.8, 0.0, 0.0), (0.08, 0.4, 0.4, 0.8, 0.0, 0.2)],
    "vicreg": [(0.2, 0.4, 0.4, 0.8, 0.0, 0.1)] * 2,
}


# ----------------------------------------------------------------------------------------------------- data
class SceneViews(Dataset):
    """Two augmented 96 x 96 views of one random region of one unlabeled FL36 train image."""

    def __init__(self, aug, samples_per_epoch, size=96):
        self.path = DATA / "fl36_train_gray.npy"
        self.n_img = int(np.load(self.path, mmap_mode="r").shape[0])
        self.aug, self.n, self.size = AUGS[aug], samples_per_epoch, size
        self.mm = None
        self.rng = None

    def __len__(self):
        return self.n

    def _region(self, img):
        for _ in range(10):
            s = int(self.rng.integers(128, 257))
            x, y = int(self.rng.integers(0, 608 - s + 1)), int(self.rng.integers(0, 608 - s + 1))
            r = img[y:y + s, x:x + s]
            if (r > 30).mean() >= 0.02:
                break
        return np.ascontiguousarray(r)

    def _rrc(self, r, min_scale):
        s = r.shape[0]
        area = s * s
        for _ in range(10):
            a = area * self.rng.uniform(min_scale, 1.0)
            ratio = math.exp(self.rng.uniform(math.log(3 / 4), math.log(4 / 3)))
            w, h = int(round(math.sqrt(a * ratio))), int(round(math.sqrt(a / ratio)))
            if 0 < w <= s and 0 < h <= s:
                x, y = int(self.rng.integers(0, s - w + 1)), int(self.rng.integers(0, s - h + 1))
                crop = r[y:y + h, x:x + w]
                break
        else:
            crop = r
        interp = cv2.INTER_AREA if crop.shape[0] > self.size else cv2.INTER_LINEAR
        return cv2.resize(crop, (self.size, self.size), interpolation=interp)

    def _view(self, r, p):
        min_scale, bri, con, p_jit, p_blur, p_sol = p
        v = self._rrc(r, min_scale).astype(np.float32)
        if self.rng.random() < p_jit:
            for op in self.rng.permutation(2):
                if op == 0:
                    v = v * self.rng.uniform(1 - bri, 1 + bri)
                else:
                    v = (v - v.mean()) * self.rng.uniform(1 - con, 1 + con) + v.mean()
                v = np.clip(v, 0, 255)
        if self.rng.random() < p_blur:
            v = cv2.GaussianBlur(v, (0, 0), self.rng.uniform(0.1, 2.0))
        if self.rng.random() < p_sol:
            v = np.where(v >= 128, 255 - v, v)
        return np.clip(v, 0, 255).astype(np.uint8)

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = np.load(self.path, mmap_mode="r")
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32) + (info.id if info else 0))
        r = self._region(self.mm[int(self.rng.integers(0, self.n_img))])
        return torch.from_numpy(np.stack([self._view(r, p) for p in self.aug]))   # [2, 96, 96] uint8


def to_input(x_uint8, device):
    """uint8 [B, H, W] -> float [B, 3, H, W] in 0..1 (no mean / std, like the detection pipeline)."""
    x = x_uint8.to(device, non_blocking=True).float().div_(255.0)
    return x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()


# ----------------------------------------------------------------------------------------------------- model
def resnet18():
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Identity()
    return m


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18()
        self.projector = nn.Sequential(nn.Linear(512, 2048), nn.ReLU(), nn.Linear(2048, 256))

    def forward(self, v1, v2, state):
        f1, f2 = self.backbone(v1), self.backbone(v2)
        z = F.normalize(torch.cat([self.projector(f1), self.projector(f2)]), dim=1)
        n = f1.shape[0]
        sim = z @ z.t() / 0.2
        sim.fill_diagonal_(float("-inf"))
        target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
        return F.cross_entropy(sim, target), f1


def mlp_bn(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, o))


class BYOL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.projector, self.predictor = resnet18(), mlp_bn(512, 4096, 256), mlp_bn(256, 4096, 256)
        self.m_backbone, self.m_projector = copy.deepcopy(self.backbone), copy.deepcopy(self.projector)
        for p in list(self.m_backbone.parameters()) + list(self.m_projector.parameters()):
            p.requires_grad = False

    def momentum_pairs(self):
        return [(self.backbone, self.m_backbone), (self.projector, self.m_projector)]

    def forward(self, v1, v2, state):
        f1, f2 = self.backbone(v1), self.backbone(v2)
        p1, p2 = self.predictor(self.projector(f1)), self.predictor(self.projector(f2))
        with torch.no_grad():
            t1, t2 = self.m_projector(self.m_backbone(v1)), self.m_projector(self.m_backbone(v2))

        def l(p, z):
            return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()

        return l(p1, t2) + l(p2, t1), f1


class DINOHead(nn.Module):
    def __init__(self, i=512, h=2048, b=256, k=4096):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, b))
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.last = nn.Parameter(torch.empty(k, b))           # weight-normalised rows, g fixed at 1
        nn.init.kaiming_uniform_(self.last, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(F.normalize(self.mlp(x), dim=-1), F.normalize(self.last, dim=1))


class DINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.head = resnet18(), DINOHead()
        self.m_backbone, self.m_head = copy.deepcopy(self.backbone), copy.deepcopy(self.head)
        for p in list(self.m_backbone.parameters()) + list(self.m_head.parameters()):
            p.requires_grad = False
        self.register_buffer("center", torch.zeros(1, 4096))

    def momentum_pairs(self):
        return [(self.backbone, self.m_backbone), (self.head, self.m_head)]

    def forward(self, v1, v2, state):
        f1, f2 = self.backbone(v1), self.backbone(v2)
        s = [self.head(f1) / 0.1, self.head(f2) / 0.1]
        with torch.no_grad():
            t_raw = torch.cat([self.m_head(self.m_backbone(v1)), self.m_head(self.m_backbone(v2))])
            t = F.softmax((t_raw - self.center) / 0.07, dim=-1).chunk(2)
            self.center.mul_(0.9).add_(t_raw.mean(dim=0, keepdim=True), alpha=0.1)
        loss = (torch.sum(-t[0] * F.log_softmax(s[1], dim=-1), dim=-1).mean()
                + torch.sum(-t[1] * F.log_softmax(s[0], dim=-1), dim=-1).mean()) / 2
        return loss, f1


class VICReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18()
        self.projector = nn.Sequential(nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                                       nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                                       nn.Linear(2048, 2048))

    def forward(self, v1, v2, state):
        f1, f2 = self.backbone(v1), self.backbone(v2)
        z1, z2 = self.projector(f1), self.projector(f2)
        inv = F.mse_loss(z1, z2)
        std = lambda z: torch.sqrt(z.var(dim=0) + 1e-4)                       # noqa: E731
        var = F.relu(1 - std(z1)).mean() + F.relu(1 - std(z2)).mean()
        n, d = z1.shape

        def cov(z):
            z = z - z.mean(dim=0)
            c = (z.t() @ z) / (n - 1)
            return (c.pow(2).sum() - c.diagonal().pow(2).sum()) / d

        return 25.0 * inv + 25.0 * var + 1.0 * (cov(z1) + cov(z2)), f1


MODELS = {"simclr": SimCLR, "byol": BYOL, "dino": DINO, "vicreg": VICReg}


# ----------------------------------------------------------------------------------------------------- optim
class LARS(torch.optim.Optimizer):
    """solo-learn LARS (MIT): eta 0.02, clip_lr, no LARS adaptation / weight decay on 1-D parameters."""

    def __init__(self, params, lr, weight_decay, momentum=0.9, eta=0.02, eps=1e-8):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta, eps=eps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if p.ndim != 1:
                    p_norm, g_norm = torch.norm(p), torch.norm(d_p)
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (g_norm + p_norm * g["weight_decay"] + g["eps"]) * g["eta"]
                        lars_lr = torch.clamp(lars_lr / g["lr"], max=1.0)
                        d_p = d_p.add(p, alpha=g["weight_decay"]) * lars_lr
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = d_p.clone()
                else:
                    st["buf"].mul_(g["momentum"]).add_(d_p)
                p.add_(st["buf"], alpha=-g["lr"])


def lr_at(step, total, warm, base, start=3e-5):
    if step < warm:
        return start + (base - start) * step / max(1, warm)
    return 0.5 * base * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


@torch.no_grad()
def ema_update(model, step, total, base_tau):
    tau = 1 - (1 - base_tau) * (math.cos(math.pi * step / total) + 1) / 2
    for online, target in model.momentum_pairs():
        for po, pt in zip(online.parameters(), target.parameters()):
            pt.mul_(tau).add_(po.detach(), alpha=1 - tau)
    return tau


# ----------------------------------------------------------------------------------------------------- eval
@torch.no_grad()
def features(backbone, x, device, bs=512):
    backbone.eval()
    out = []
    for k in range(0, len(x), bs):
        out.append(backbone(to_input(torch.from_numpy(x[k:k + bs]), device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def knn_top1(ftr, ytr, fte, yte, k=20, t=0.07, n_cls=36, device="cuda"):
    ftr, fte = F.normalize(ftr, dim=1).to(device), F.normalize(fte, dim=1).to(device)
    ytr = torch.as_tensor(ytr).to(device)
    correct = 0
    for i in range(0, len(fte), 1024):
        s, idx = (fte[i:i + 1024] @ ftr.t()).topk(k, dim=1)
        votes = torch.zeros(len(s), n_cls, device=device).scatter_add_(1, ytr[idx], (s / t).exp())
        correct += (votes.argmax(1).cpu() == torch.as_tensor(yte[i:i + 1024])).sum().item()
    return correct / len(fte)


def linear_probe(ftr, ytr, fte, yte, device, epochs=100, n_cls=36):
    from sklearn.metrics import f1_score
    torch.manual_seed(777)
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    xtr, xte = ((ftr - mu) / sd).to(device), ((fte - mu) / sd).to(device)
    ytr_t = torch.as_tensor(ytr).to(device)
    lin = nn.Linear(xtr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), device=device)
        for k in range(0, len(xtr), 1024):
            b = perm[k:k + 1024]
            loss = F.cross_entropy(lin(xtr[b]), ytr_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = lin(xte).argmax(1).cpu().numpy()
    return float((pred == yte).mean()), float(f1_score(yte, pred, average="macro"))


def evaluate(backbone, device, probe=True):
    tr, te = np.load(DATA / "crop36_train.npz"), np.load(DATA / "crop36_test.npz")
    ftr, fte = features(backbone, tr["x"], device), features(backbone, te["x"], device)
    res = {"knn_top1": knn_top1(ftr, tr["y"], fte, te["y"], device=device)}
    if probe:
        res["probe_top1"], res["probe_macro_f1"] = linear_probe(ftr, tr["y"], fte, te["y"], device)
    return res


@torch.no_grad()
def collapse_index(f):
    """std over the batch of L2-normalised features, averaged over dims, x sqrt(dim). ~0 = collapsed."""
    z = F.normalize(f.float(), dim=1)
    return float(z.std(dim=0).mean() * math.sqrt(z.shape[1]))


def gpu_status():
    try:
        q = "temperature.gpu,clocks.sm,power.draw,memory.used"
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception:
        return "n/a"


# ----------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=list(MODELS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--samples-per-epoch", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--gate-epoch", type=int, default=None, help="kNN gate epoch (default 10 %% of --epochs)")
    ap.add_argument("--gate-margin", type=float, default=0.05, help="kNN must beat random init by this")
    ap.add_argument("--knn-every", type=int, default=0, help="log Crop36 kNN every N epochs (record only)")
    ap.add_argument("--bench", type=int, default=0, help="time N steps after 5 warm-up steps, write nothing")
    ap.add_argument("--smoke", action="store_true", help="1 short epoch (20 steps) + collapse + kNN, own folder")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--init", default="random", help="eval-only: random or a backbone .pth")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: CUDA not available - refusing to train on CPU (lesson 11).")
        if not a.eval_only:
            sys.exit(2)

    if a.eval_only:
        bb = resnet18().to(device)
        if a.init != "random":
            bb.load_state_dict(torch.load(a.init, map_location=device))
        res = evaluate(bb, device)
        res["init"] = a.init
        print(json.dumps(res))
        out = RUNS / "ssl_eval.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        if a.init == "random":
            (RUNS / "ssl_random_knn.json").write_text(json.dumps(res))      # reference for the kNN gate
        return

    assert a.method, "--method is required"
    rec = RECIPES[a.method]
    tag = a.method + ("_smoke" if a.smoke else "")
    run_dir, log_path = RUNS / f"ssl_{tag}", LOGS / f"ssl_{tag}.log"
    if not a.bench:
        run_dir.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
    stop_file = LOGS / f"ssl_{a.method}.stop"

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        if not a.bench:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    spe = a.samples_per_epoch if not a.smoke else a.batch_size * 20
    ds = SceneViews(rec["aug"], spe)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, drop_last=True,
                        pin_memory=True, persistent_workers=a.workers > 0)
    steps_per_epoch = len(loader)
    total, warm = a.epochs * steps_per_epoch, a.warmup_epochs * steps_per_epoch
    if a.smoke:
        total, warm = steps_per_epoch, max(1, steps_per_epoch // 4)

    model = MODELS[a.method]().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = LARS(params, lr=rec["lr"], weight_decay=rec["wd"])
    start_epoch, step = 0, 0
    ckpt = run_dir / "last.pth"
    if ckpt.is_file() and not (a.overwrite or a.bench or a.smoke):
        s = torch.load(ckpt, map_location=device)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        start_epoch, step = s["epoch"] + 1, s["step"]
        log(f"resume from {ckpt} at epoch {start_epoch}")
    elif not a.bench:
        log(f"start {a.method}: epochs {a.epochs}, steps/epoch {steps_per_epoch}, batch {a.batch_size}, "
            f"recipe {rec}, aug {AUGS[rec['aug']]}, workers {a.workers}")

    gate_epoch = a.gate_epoch if a.gate_epoch is not None else max(1, round(0.1 * a.epochs))
    random_knn = None
    rk = RUNS / "ssl_random_knn.json"
    if rk.is_file():
        random_knn = json.loads(rk.read_text())["knn_top1"]

    # ------------------------------------------------------------------ bench
    if a.bench:
        torch.cuda.reset_peak_memory_stats()
        it, n_done, t0 = iter(loader), 0, None
        model.train()
        t_wait = 0.0
        for k in range(5 + a.bench):
            tw = time.time()
            x = next(it)
            t_wait += (time.time() - tw) if k >= 5 else 0
            if k == 5:
                torch.cuda.synchronize()
                t0 = time.time()
            v1, v2 = to_input(x[:, 0], device), to_input(x[:, 1], device)
            loss, f1 = model(v1, v2, None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if hasattr(model, "momentum_pairs"):
                ema_update(model, 1, 100, rec["tau"])
            n_done += a.batch_size if k >= 5 else 0
        torch.cuda.synchronize()
        dt = time.time() - t0
        sps = n_done / dt
        ep_min = a.samples_per_epoch / sps / 60
        print(f"BENCH {a.method}: {sps:.0f} samples/s ({2 * sps:.0f} views/s), {1000 * dt / a.bench:.0f} ms/step, "
              f"data wait {100 * t_wait / dt:.0f} %, peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
              f"loss {loss.item():.3f}, collapse {collapse_index(f1):.2f}")
        print(f"BENCH {a.method}: 1 epoch ({a.samples_per_epoch} samples) ~ {ep_min:.1f} min; "
              f"GPU temp,clock,W,MiB = {gpu_status()}")
        return

    # ------------------------------------------------------------------ train
    t_start = time.time()
    for epoch in range(start_epoch, a.epochs if not a.smoke else 1):
        model.train()
        t_ep, loss_sum, n_b, ci = time.time(), 0.0, 0, 0.0
        for x in loader:
            if stop_file.is_file() or (LOGS / "wave1a.stop").is_file():
                log(f"STOP file found - stopping at epoch {epoch + 1} step {step}; resume restarts this epoch")
                return
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warm, rec["lr"])
            v1, v2 = to_input(x[:, 0], device), to_input(x[:, 1], device)
            loss, f1 = model(v1, v2, None)
            if not torch.isfinite(loss):
                log(f"FAILED non-finite loss at epoch {epoch} step {step}")
                sys.exit(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.method == "dino" and epoch == 0:
                model.head.last.grad = None                 # freeze_last_layer: 1 epoch
            opt.step()
            tau = ema_update(model, step, total, rec["tau"]) if hasattr(model, "momentum_pairs") else None
            loss_sum += loss.item()
            n_b += 1
            step += 1
            if n_b % 32 == 1:
                ci = collapse_index(f1)
        dt = time.time() - t_ep
        msg = (f"epoch {epoch + 1}/{a.epochs} loss {loss_sum / n_b:.4f} lr {opt.param_groups[0]['lr']:.4f} "
               f"collapse {ci:.2f} {'tau %.5f ' % tau if tau else ''}time {dt / 60:.1f} min "
               f"({spe / dt:.0f} samples/s) gpu[T,MHz,W,MiB] {gpu_status()}")
        log(msg)
        if not a.smoke:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step},
                       run_dir / "last.tmp")
            os.replace(run_dir / "last.tmp", ckpt)
        if a.knn_every and (epoch + 1) % a.knn_every == 0 and (epoch + 1) != gate_epoch and not a.smoke:
            torch.save(model.backbone.state_dict(), run_dir / f"backbone_e{epoch + 1:03d}.pth")
            log(f"knn epoch {epoch + 1} top1 {evaluate(model.backbone, device, probe=False)['knn_top1']:.4f}")
        if (epoch + 1) == gate_epoch and not a.smoke:
            torch.save(model.backbone.state_dict(), run_dir / f"backbone_e{epoch + 1:03d}.pth")
            r = evaluate(model.backbone, device, probe=False)
            log(f"gate kNN top1 {r['knn_top1']:.4f}  (random init {random_knn})")
            if random_knn is not None and r["knn_top1"] < random_knn + a.gate_margin:
                log(f"GATE FAIL {a.method}: kNN {r['knn_top1']:.4f} < random {random_knn:.4f} + {a.gate_margin}")
                sys.exit(3)

    if a.smoke:
        r = evaluate(model.backbone, device, probe=False)
        log(f"SMOKE {a.method}: collapse {ci:.2f}, kNN top1 {r['knn_top1']:.4f} (random {random_knn})")
        log(f"DONE {tag}")
        return
    torch.save(model.backbone.state_dict(), run_dir / "backbone_final.pth")
    r = evaluate(model.backbone, device)
    r.update(method=a.method, epochs=a.epochs, hours=round((time.time() - t_start) / 3600, 2))
    (run_dir / "eval_final.json").write_text(json.dumps(r, indent=1))
    log(f"final kNN {r['knn_top1']:.4f} probe top1 {r['probe_top1']:.4f} macro F1 {r['probe_macro_f1']:.4f}")
    log(f"DONE {a.method}")


if __name__ == "__main__":
    main()
