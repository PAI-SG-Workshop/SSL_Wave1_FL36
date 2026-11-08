"""
prep_ssl_data.py - build the inputs of SSL Wave 1a from FashionLetters36 (CPU only).

    python prep_ssl_data.py                       # uses $PHYSICAL_AI_ROOT (default C:\\SSL_Wave1)

Outputs under $PHYSICAL_AI_ROOT/datasets/ssl/
  fl36_train_gray.npy   uint8 [4096, 608, 608]  the 4,096 FL36 train images, grayscale, NO labels or boxes.
                        Read with np.load(mmap_mode="r"): data-loader workers share it through the OS cache.
  crop36_train.npz      x uint8 [26716, 96, 96], y int64   object crops of FL36 train  (evaluation only)
  crop36_test.npz       x uint8 [6699, 96, 96],  y int64   object crops of FL36 test   (evaluation only)
  ssl_data.json         counts, md5 of the arrays, crop rule

Crop36 rule: square around the box centre, side = 1.15 x max(box w, box h) (+ black padding at the image edge),
resized to 96 x 96 (INTER_AREA when shrinking, INTER_LINEAR when enlarging). Classes = common.config FL36 order.
Checks (RESULT OK / FAIL): 4,096 train images; 26,716 / 6,699 crops; all 36 classes in both splits.
"""

import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot",
           *[chr(ord("A") + i) for i in range(26)])
EXPECT = {"trainval": (4096, 26716), "test": (1024, 6699)}
CROP, MARGIN = 96, 1.15

root = Path(os.environ.get("PHYSICAL_AI_ROOT", r"C:\SSL_Wave1"))
fl36 = root / "datasets" / "FashionLetters36"
out = root / "datasets" / "ssl"
out.mkdir(parents=True, exist_ok=True)
cls_id = {c.lower(): i for i, c in enumerate(CLASSES)}
fails, info = [], {"crop_rule": f"square side {MARGIN} x max(w,h), black pad, resize {CROP}", "classes": CLASSES}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def square_crop(img, x0, y0, x1, y1):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(round(MARGIN * max(x1 - x0, y1 - y0)))
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = side
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    patch = padded[b + pad:b + pad + side, a + pad:a + pad + side]
    interp = cv2.INTER_AREA if side > CROP else cv2.INTER_LINEAR
    return cv2.resize(patch, (CROP, CROP), interpolation=interp)


for split, (n_img_exp, n_obj_exp) in EXPECT.items():
    ids = [x for x in (fl36 / "ImageSets" / "Main" / f"{split}.txt").read_text().split() if x]
    if len(ids) != n_img_exp:
        fails.append(f"{split}: {len(ids)} images != {n_img_exp}")
    if split == "trainval":
        mm_path = out / "fl36_train_gray.npy"
        mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.uint8, shape=(len(ids), 608, 608))
    xs, ys = [], []
    for k, i in enumerate(ids):
        img = cv2.imread(str(fl36 / "JPEGImages" / f"{i}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (608, 608):
            fails.append(f"{split}: bad image {i}")
            break
        if split == "trainval":
            mm[k] = img
        for obj in ET.parse(fl36 / "Annotations" / f"{i}.xml").getroot().iter("object"):
            name = obj.find("name").text.strip().lower()
            bb = obj.find("bndbox")
            x0, y0, x1, y1 = (float(bb.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
            xs.append(square_crop(img, x0 - 1, y0 - 1, x1, y1))
            ys.append(cls_id[name])
        if (k + 1) % 512 == 0:
            print(f"\r{split}: {k + 1}/{len(ids)}", end="", flush=True)
    print()
    if split == "trainval":
        mm.flush()
        del mm
    x, y = np.stack(xs), np.array(ys, np.int64)
    name = "crop36_train.npz" if split == "trainval" else "crop36_test.npz"
    np.savez(out / name, x=x, y=y)
    per_class = np.bincount(y, minlength=len(CLASSES))
    print(f"{name}: {len(y)} crops (expected {n_obj_exp}), classes {int((per_class > 0).sum())}/36, "
          f"per-class min {per_class.min()} max {per_class.max()}")
    if len(y) != n_obj_exp:
        fails.append(f"{split}: {len(y)} crops != {n_obj_exp}")
    if (per_class == 0).any():
        fails.append(f"{split}: classes missing")
    info[name] = {"n": int(len(y)), "per_class": per_class.tolist(), "md5": md5(out / name)}

info["fl36_train_gray.npy"] = {"shape": [4096, 608, 608], "md5": md5(out / "fl36_train_gray.npy")}
info["result"] = "OK" if not fails else "FAIL"
info["fails"] = fails
(out / "ssl_data.json").write_text(json.dumps(info, indent=1))
for f in fails:
    print("FAIL:", f)
print(f"RESULT {info['result']}  ({out})")
sys.exit(0 if not fails else 1)
