"""
sup_ref.py - supervised upper reference for SSL Wave 1a: ResNet-18 trained WITH labels on Crop36 train.

Reference only (it uses the labels the SSL methods never see). Same backbone, input scaling (x / 255,
3 identical channels, no mean / std) and 96 x 96 crops as the SSL evaluation.
Recipe: 30 epochs, batch 256, Adam lr 1e-3 cosine to 0, seed 777, last epoch reported.
Augmentation on the GPU: random scale 0.8-1.2 and shift +-10 % (affine, black fill), brightness / contrast
x U(0.6, 1.4) with prob 0.8, no flip (letters are not mirror-invariant).

    python sup_ref.py            # ~1 h on the GTX 1060 under throttling
Outputs runs/ssl_supervised/backbone_final.pth (evaluated by final_eval.py like the SSL backbones) and
eval_final.json with the classifier's own Crop36 test top-1 / macro F1. Log ends with "DONE supervised".
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ssl_train as S


def augment(x):
    b = x.shape[0]
    s = torch.empty(b, device=x.device).uniform_(0.8, 1.2)
    t = torch.empty(b, 2, device=x.device).uniform_(-0.1, 0.1)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, :, 2] = 1 / s, 1 / s, t
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    jit = torch.rand(b, 1, 1, 1, device=x.device) < 0.8
    bri = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.6, 1.4)
    con = torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.6, 1.4)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    xj = ((x * bri - m * bri) * con + m * bri).clamp(0, 1)
    return torch.where(jit, xj, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 1,024 crops, batch 32, own folder")
    a = ap.parse_args()
    torch.manual_seed(777)
    device = "cuda"
    tag = "supervised_smoke" if a.smoke else "supervised"
    run_dir = S.RUNS / f"ssl_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = S.LOGS / f"ssl_{tag}.log"

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    tr, te = np.load(S.DATA / "crop36_train.npz"), np.load(S.DATA / "crop36_test.npz")
    n_tr = 1024 if a.smoke else len(tr["y"])
    xtr = torch.from_numpy(tr["x"][:n_tr]).to(device)
    ytr = torch.from_numpy(tr["y"][:n_tr]).to(device)
    model = nn.Sequential(S.resnet18(), nn.Linear(512, 36)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs, bs = (1, 32) if a.smoke else (30, 256)
    steps = epochs * (len(xtr) // bs)
    step = 0
    log(f"start supervised reference: {len(xtr)} crops, epochs {epochs}, batch {bs}, Adam 1e-3 cosine")
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=device)
        tot, n = 0.0, 0
        for k in range(0, len(xtr) - bs + 1, bs):
            for g in opt.param_groups:
                g["lr"] = 0.5 * 1e-3 * (1 + math.cos(math.pi * step / steps))
            idx = perm[k:k + bs]
            x = augment(S.to_input(xtr[idx], device))
            loss = F.cross_entropy(model(x), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot, n, step = tot + loss.item(), n + 1, step + 1
        log(f"epoch {ep + 1}/{epochs} loss {tot / n:.4f} time {(time.time() - t0) / 60:.1f} min "
            f"gpu[T,MHz,W,MiB] {S.gpu_status()}")
    model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(te["x"]), 512):
            preds.append(model(S.to_input(torch.from_numpy(te["x"][k:k + 512]), device)).argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    from sklearn.metrics import f1_score
    res = {"classifier_top1": float((pred == te["y"]).mean()),
           "classifier_macro_f1": float(f1_score(te["y"], pred, average="macro")),
           "hours": round((time.time() - t0) / 3600, 2)}
    torch.save(model[0].state_dict(), run_dir / "backbone_final.pth")
    (run_dir / "eval_final.json").write_text(json.dumps(res, indent=1))
    log(f"classifier test top1 {res['classifier_top1']:.4f} macro F1 {res['classifier_macro_f1']:.4f}")
    log(f"DONE {tag}")


if __name__ == "__main__":
    main()
