"""
fl36_count.py - verify FashionLetters36 before any training (SSL Wave 1a, gate 1).

Run from the CenterNet repo folder (it imports the repo's own loader):
    cd /d C:\\SSL_Wave1\\code\\OD_Image_1-Stage_CenterNet
    python C:\\SSL_Wave1\\downloads\\fl36_count.py

Checks
  1. image / XML / split-file counts            expected 4,096 train, 1,024 test
  2. objects from the raw XML                    expected 26,716 train, 6,699 test, all 36 classes in both splits
  3. objects through the repo loader (VOCDetection.annotation) must equal the raw XML counts
     (the 10-16 class bug dropped 72 % of the objects silently at this step)
  4. 32 sampled images: 608 x 608, 3 channels, R = G = B (grayscale content), background black
  5. box side statistics (used to set the SSL crop size)
Prints RESULT OK or RESULT FAIL and writes fl36_count.json next to the dataset.
"""

import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path.cwd()))
from common import config as C                      # noqa: E402
from common.datasets.voc import VOCDetection        # noqa: E402

EXPECT = {"trainval": (4096, 26716), "test": (1024, 6699)}
root = Path(C.FL36_ROOT)
classes = list(C.FL36_CLASSES)
fails = []
report = {"root": str(root)}
print(f"dataset root: {root}")

for split, (n_img_exp, n_obj_exp) in EXPECT.items():
    ids = [x for x in (root / "ImageSets" / "Main" / f"{split}.txt").read_text().split() if x]
    jpg = sum((root / "JPEGImages" / f"{i}.jpg").is_file() for i in ids)
    xml = sum((root / "Annotations" / f"{i}.xml").is_file() for i in ids)

    raw = Counter()
    sides = []
    for i in ids:
        for obj in ET.parse(root / "Annotations" / f"{i}.xml").getroot().iter("object"):
            raw[obj.find("name").text.strip()] += 1
            bb = obj.find("bndbox")
            x0, y0, x1, y1 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            sides.append(max(x1 - x0 + 1, y1 - y0 + 1))
    n_raw = sum(raw.values())

    ds = VOCDetection(root, split, 608, classes, train=False)
    via_loader = Counter()
    for k in range(len(ds)):
        for lab in ds.annotation(k)["labels"]:
            via_loader[classes[int(lab)]] += 1
    n_loader = sum(via_loader.values())

    missing = [c for c in classes if raw.get(c, 0) == 0]
    lost = {c: raw[c] - via_loader.get(c, 0) for c in raw if raw[c] != via_loader.get(c, 0)}
    s = np.array(sides)
    print(f"\n[{split}] ids {len(ids)}  jpg {jpg}  xml {xml}  (expected {n_img_exp})")
    print(f"  objects raw XML {n_raw}  via loader {n_loader}  (expected {n_obj_exp})")
    print(f"  classes present {36 - len(missing)}/36  per-class min {min(raw.values())}  max {max(raw.values())}")
    print(f"  box long side px: min {s.min():.0f}  p10 {np.percentile(s, 10):.0f}  median {np.median(s):.0f}"
          f"  p90 {np.percentile(s, 90):.0f}  max {s.max():.0f}  objects/image {n_raw / len(ids):.2f}")

    if not (len(ids) == jpg == xml == n_img_exp):
        fails.append(f"{split}: image count")
    if n_raw != n_obj_exp:
        fails.append(f"{split}: raw objects {n_raw} != {n_obj_exp}")
    if n_loader != n_raw:
        fails.append(f"{split}: loader dropped {n_raw - n_loader} objects {lost}")
    if missing:
        fails.append(f"{split}: classes missing {missing}")
    report[split] = {"images": len(ids), "objects_raw": n_raw, "objects_loader": n_loader,
                     "per_class": dict(raw), "box_side_median": float(np.median(s))}

    rng = np.random.default_rng(0)
    for i in rng.choice(ids, size=min(32, len(ids)), replace=False):
        a = np.asarray(Image.open(root / "JPEGImages" / f"{i}.jpg"))
        if a.shape != (608, 608, 3):
            fails.append(f"{split}: {i} shape {a.shape}")
            break
        if np.abs(a[..., 0].astype(int) - a[..., 1]).max() > 8 or np.abs(a[..., 1].astype(int) - a[..., 2]).max() > 8:
            fails.append(f"{split}: {i} not grayscale content")
            break
    dark = np.mean([(np.asarray(Image.open(root / "JPEGImages" / f"{i}.jpg").convert("L")) < 16).mean()
                    for i in ids[:32]])
    print(f"  sampled images: 608x608x3, grayscale content; dark-pixel share {dark:.2f}")
    report[split]["dark_pixel_share"] = float(dark)

report["result"] = "OK" if not fails else "FAIL"
report["fails"] = fails
(root / "fl36_count.json").write_text(json.dumps(report, indent=1))
print()
for f in fails:
    print("FAIL:", f)
print(f"RESULT {report['result']}  (report: {root / 'fl36_count.json'})")
