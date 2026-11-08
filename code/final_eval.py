"""
final_eval.py - one evaluation protocol for every Wave 1a backbone (random, 6 SSL, supervised reference).

For each backbone found:
  knn_stored    Crop36 kNN (k 20, cosine, t 0.07) with the BN statistics stored in the backbone
  knn_bn        the same after re-estimating the BN statistics on the Crop36 TRAIN images (labels unused)
  probe_top1 / probe_f1   linear probe (standardised frozen features, Adam 1e-3, 100 epochs, seed 777) on the
                          re-estimated-BN features
The method comparison uses knn_bn and the probe (knn_stored is listed because the random init looks
strong under it; see the plan doc, section 2). Random init uses torch.manual_seed(777).

    python final_eval.py
Writes runs/wave1a_final_eval.json and runs/wave1a_final_eval.md. Log ends with "DONE final_eval".
"""
import json
import time

import numpy as np
import torch

import ssl_train as S
from knn_diag import recompute_bn

METHODS = ["simclr", "byol", "dino", "vicreg", "simsiam", "barlow", "supervised"]


def main():
    device = "cuda"
    tr, te = np.load(S.DATA / "crop36_train.npz"), np.load(S.DATA / "crop36_test.npz")
    specs = [("random", None)] + [(m, S.RUNS / f"ssl_{m}" / "backbone_final.pth") for m in METHODS]
    rows = []
    for name, path in specs:
        if path is not None and not path.is_file():
            print(f"skip {name}: {path} not found", flush=True)
            continue
        torch.manual_seed(777)
        bb = S.resnet18().to(device)
        if path is not None:
            bb.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        t0 = time.time()
        ftr, fte = S.features(bb, tr["x"], device), S.features(bb, te["x"], device)
        knn_stored = S.knn_top1(ftr, tr["y"], fte, te["y"], device=device)
        recompute_bn(bb, tr["x"], device)
        ftr, fte = S.features(bb, tr["x"], device), S.features(bb, te["x"], device)
        knn_bn = S.knn_top1(ftr, tr["y"], fte, te["y"], device=device)
        p1, pf = S.linear_probe(ftr, tr["y"], fte, te["y"], device)
        row = {"backbone": name, "knn_stored": round(knn_stored, 4), "knn_bn": round(knn_bn, 4),
               "probe_top1": round(p1, 4), "probe_f1": round(pf, 4)}
        ej = None if path is None else path.parent / "eval_final.json"
        if ej is not None and ej.is_file():
            row["train_info"] = json.loads(ej.read_text())
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "train_info"}),
              f"({time.time() - t0:.0f} s)", flush=True)
    (S.RUNS / "wave1a_final_eval.json").write_text(json.dumps(rows, indent=1))
    md = ["| backbone | kNN (stored BN) | kNN (BN re-estimated) | probe top-1 | probe macro F1 |",
          "| --- | --- | --- | --- | --- |"]
    md += [f"| {r['backbone']} | {r['knn_stored']:.4f} | {r['knn_bn']:.4f} | {r['probe_top1']:.4f} | "
           f"{r['probe_f1']:.4f} |" for r in rows]
    (S.RUNS / "wave1a_final_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print("DONE final_eval")


if __name__ == "__main__":
    main()
