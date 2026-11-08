"""
ssl_train_ext.py - adds SimSiam and Barlow Twins to ssl_train.py (SSL Wave 1a, second queue).

ssl_train.py is used by the running queue and is NOT modified. This file imports it, registers two more
methods in its tables and calls its main(), so data, views, LARS, schedule, evaluation, checkpoints, logs
and resume are exactly the same code as for the first four methods.

    simsiam  projector 512-2048(BN)-2048(BN)-2048(BN, no affine), predictor 2048-512(BN)-2048,
             loss -cos(p, stopgrad z) symmetrised / 2, SGD lr 0.5 momentum 0.9 wd 1e-5 (all params),
             predictor at a constant lr 0.5 (solo-learn static_lr), symmetric_weak aug       (solo-learn cifar)
    barlow   projector 512-2048(BN)-2048(BN)-2048, loss on batch-standardised z: on-diagonal (1 - c)^2 +
             0.0051 x off-diagonal c^2, x 0.1, LARS lr 0.3 wd 1e-4, asymmetric aug               (solo-learn cifar)
Same as the other four: batch 256, warmup 10 epochs from 3e-5 then cosine (per step), fp32, seed 777,
no horizontal flip, scene-region views (see ssl_train.py).

    python ssl_train_ext.py --method simsiam --epochs 50 --workers 2 --gate-margin -1 --knn-every 10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import ssl_train as S

# symmetric_weak (solo-learn cifar): rrc 0.08, brightness / contrast 0.4, jitter 0.8, blur 0.5, no solarize
S.AUGS["symmetric_weak"] = [(0.08, 0.4, 0.4, 0.8, 0.5, 0.0)] * 2
S.RECIPES["simsiam"] = dict(lr=0.5, wd=1e-5, aug="symmetric_weak", optimizer="sgd, predictor static lr")
S.RECIPES["barlow"] = dict(lr=0.3, wd=1e-4, aug="asymmetric")

_PREDICTOR_IDS = set()


class SimSiam(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = S.resnet18()
        self.projector = nn.Sequential(
            nn.Linear(512, 2048, bias=False), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048, bias=False), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048, affine=False))
        self.projector[6].bias.requires_grad = False          # followed by BN (as solo-learn)
        self.predictor = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
                                       nn.Linear(512, 2048))
        _PREDICTOR_IDS.update(id(p) for p in self.predictor.parameters())

    def forward(self, v1, v2, state):
        f1, f2 = self.backbone(v1), self.backbone(v2)
        z1, z2 = self.projector(f1), self.projector(f2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        loss = (-F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
                - F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) / 2
        return loss, f1


class BarlowTwins(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = S.resnet18()
        self.projector = nn.Sequential(nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                                       nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
                                       nn.Linear(2048, 2048))

    def forward(self, v1, v2, state):
        f1, f2 = self.backbone(v1), self.backbone(v2)
        z1, z2 = self.projector(f1), self.projector(f2)
        n = z1.shape[0]

        def std(z):                                           # = BatchNorm1d(affine=False) in train mode
            return (z - z.mean(0)) / torch.sqrt(z.var(0, unbiased=False) + 1e-5)

        c = std(z1).t() @ std(z2) / n
        on = (torch.diagonal(c) - 1).pow(2).sum()
        off = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
        return 0.1 * (on + 0.0051 * off), f1


class SGDStaticPredictor(torch.optim.SGD):
    """SGD whose 'static' group keeps a constant lr even though ssl_train.py rewrites every group's lr."""

    def step(self, closure=None):
        for g in self.param_groups:
            if g.get("static_lr") is not None:
                g["lr"] = g["static_lr"]
        return super().step(closure)


_LARS = S.LARS


def make_optimizer(params, lr, weight_decay):
    params = list(params)
    if _PREDICTOR_IDS and any(id(p) in _PREDICTOR_IDS for p in params):          # SimSiam
        pred = [p for p in params if id(p) in _PREDICTOR_IDS]
        rest = [p for p in params if id(p) not in _PREDICTOR_IDS]
        return SGDStaticPredictor([{"params": rest, "static_lr": None},
                                   {"params": pred, "static_lr": lr}],
                                  lr=lr, momentum=0.9, weight_decay=weight_decay)
    return _LARS(params, lr=lr, weight_decay=weight_decay)


S.MODELS["simsiam"] = SimSiam
S.MODELS["barlow"] = BarlowTwins
S.LARS = make_optimizer          # ssl_train.main() builds its optimizer through this name

if __name__ == "__main__":
    S.main()
