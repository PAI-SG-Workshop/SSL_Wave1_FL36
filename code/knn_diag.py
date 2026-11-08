"""
knn_diag.py - is a low SSL kNN caused by BatchNorm statistics? (SSL Wave 1a diagnosis, ~1 min per backbone)

    python knn_diag.py random  C:\\SSL_Wave1\\runs\\ssl_simclr\\backbone_e005.pth

For each backbone (random = untrained ResNet-18):
  A  kNN with the stored BN statistics (what ssl_train.py reports)
  B  kNN after re-estimating the BN statistics on the Crop36 TRAIN images (labels not used,
     cumulative average over all 26,716 crops, like PreciseBN in the detection pipeline)
Prints one line per backbone and appends to C:\\SSL_Wave1\\runs\\ssl_knn_diag.jsonl.
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

import ssl_train as S


@torch.no_grad()
def recompute_bn(backbone, x, device, bs=256):
    bns = [m for m in backbone.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                       # cumulative average
    backbone.train()
    for k in range(0, len(x), bs):
        backbone(S.to_input(torch.from_numpy(x[k:k + bs]), device))
    backbone.eval()


def main():
    device = "cuda"
    tr, te = np.load(S.DATA / "crop36_train.npz"), np.load(S.DATA / "crop36_test.npz")
    for spec in sys.argv[1:]:
        bb = S.resnet18().to(device)
        if spec != "random":
            bb.load_state_dict(torch.load(spec, map_location=device))
        a = S.knn_top1(S.features(bb, tr["x"], device), tr["y"], S.features(bb, te["x"], device), te["y"],
                       device=device)
        recompute_bn(bb, tr["x"], device)
        b = S.knn_top1(S.features(bb, tr["x"], device), tr["y"], S.features(bb, te["x"], device), te["y"],
                       device=device)
        res = {"backbone": spec, "knn_stored_bn": round(a, 4), "knn_bn_recomputed": round(b, 4)}
        print(json.dumps(res), flush=True)
        with open(S.RUNS / "ssl_knn_diag.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
