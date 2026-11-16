# SSL_Wave1_FL36

**Self-supervised pre-training of ResNet-18 on FashionLetters36 - PAI-SG SSL Wave 1**

Part of the *Physical AI* reference series of the **Statera Working Group**. The detection series trained every
detector from scratch; this repository asks whether self-supervised pre-training on the *same unlabeled images*
helps detection, and how much of the help is real once labels and training time vary.
Semi-supervised learning on the same data: [SemiSL_Wave2_FL36](https://github.com/Statera-Workshop/SemiSL_Wave2_FL36).

## Setup

SSL pre-training of a ResNet-18 (the CenterNet backbone; `torchvision.models.resnet18` keys = `CenterNet.net.*`)
on the 4,096 FashionLetters36 train images, labels and boxes never used. Evaluation:

- **Crop36** representation: kNN (k 20, cosine) and linear probe on 96 x 96 object crops (26,716 / 6,699),
  after re-estimating BN statistics on the train crops (labels unused). Under the stored statistics a *random*
  ResNet-18 looks strong (kNN 0.785) because its default BN passes the raw pixel scale through; with re-estimated
  BN it drops to 0.642.
- **CenterNet downstream** (FL36, 36 classes): the common detection protocol of the series
  ([OD_Image_1-Stage_CenterNet](https://github.com/Statera-Workshop/OD_Image_1-Stage_CenterNet) v1.1: 30 epochs,
  Adam 1e-4, effective batch 8, seed 777, PreciseBN, last epoch, test mAP50); only the backbone initialisation differs
  (`code/train_image_centernet_ssl.py`, `code/train_image_centernet_lab.py`).

Methods (solo-learn CIFAR recipes; see `code/ssl_train.py`, `code/ssl_train_ext.py`): SimCLR, BYOL, DINO, VICReg,
SimSiam, Barlow Twins; batch 256, 96 x 96 views of 128-256 px scene regions, no horizontal flip, fp32, seed 777.
A supervised reference (ResNet-18 trained *with* labels on Crop36) is included.

## Results

Two machines: **GTX 1060 6 GB** (Windows 10, Wave 1a: 4 methods x 50 epochs, `results/gtx1060/`) and
**RTX 4070 Laptop 8 GB** (Ubuntu, everything else, `results/rtx4070/`). The same code and seed reproduce across the
two GPUs within 0.016 kNN (stored BN). All CenterNet runs (30 and 90 epochs): `results/rtx4070/centernet_summary.md`.

### 1. Representation and detection (RTX 4070, CenterNet 30 epochs, 100 % labels)

| Backbone | kNN (BN re-est.) 50 / 200 ep | Probe top-1 50 / 200 ep | CenterNet mAP50 50 / 200 ep |
| --- | --- | --- | --- |
| random / from scratch | 0.642 | 0.605 | 0.611 (rerun 0.609; public baseline 0.644) |
| supervised on Crop36 (labels) | 0.935 | 0.927 | 0.663 |
| Barlow Twins | 0.867 / 0.886 | 0.896 / 0.907 | 0.744 / **0.822** |
| VICReg | 0.871 / 0.893 | 0.905 / 0.911 | 0.757 / 0.770 |
| SimCLR | 0.844 / 0.871 | 0.890 / 0.902 | 0.701 / 0.701 |
| SimSiam | 0.706 / 0.784 | 0.844 / 0.880 | 0.583 / 0.610 |
| BYOL | 0.561 / 0.702 | 0.769 / 0.835 | 0.566 / 0.590 |
| DINO | 0.708 / - | 0.754 / - | 0.590 / - |

SSL seeds (CenterNet mAP50, seeds 777 / 778 / 779): 50 epochs - VICReg 0.757 / 0.743 / 0.751, Barlow 0.744 / 0.755 /
0.765, SimCLR 0.701 / 0.649 / 0.695; 200 epochs - **Barlow 0.822 / 0.803 / 0.814 (mean 0.813)**, VICReg 0.770 / 0.810 /
0.737 (mean 0.772). Re-running the same CenterNet job varies by ~0.005; changing the SSL seed by up to 0.07.

### 2. Label efficiency and training budget (CenterNet mAP50)

Labeled subsets: nested, seed 777, repeated to ~4,100 samples per epoch so every ratio takes the same number of
optimizer steps (`code/make_label_splits.py`). Two budgets: the series' 30 epochs and 3x (90 epochs).

| Init | Budget | 1 % | 5 % | 10 % | 25 % | 100 % |
| --- | --- | --- | --- | --- | --- | --- |
| from scratch | 30 ep | 0.139 | 0.463 | 0.564 | 0.583 | 0.611 |
| SimCLR 200 | 30 ep | 0.241 | 0.561 | 0.638 | 0.679 | 0.701 |
| VICReg 200 | 30 ep | 0.214 | 0.611 | 0.715 | 0.785 | 0.770 |
| Barlow 200 | 30 ep | 0.319 | 0.676 | 0.771 | 0.806 | 0.813 (3 SSL seeds) |
| from scratch | 90 ep | 0.131 | 0.605 | 0.725 / 0.720 (2 fine-tune seeds) | 0.814 | 0.862 / 0.859 |
| Barlow 200 | 90 ep | 0.330 | 0.711 | 0.770 / 0.774 / 0.775 (3 SSL seeds) | 0.831 | 0.867 / 0.872 |
| **gap at 90 ep** | | **+0.20** | **+0.11** | **+0.05** | **+0.02** | **+0.01** |

Also at 90 epochs: VICReg 200 - 10 % 0.741, 100 % 0.881; Barlow 800 - 10 % 0.775, 100 % 0.884 (one seed each).

### 3. Longer SSL (800 epochs)

| Backbone | kNN (BN re-est.) 200 / 800 ep | CenterNet 100 %, 30 ep, 200 / 800 ep |
| --- | --- | --- |
| Barlow Twins | 0.886 / 0.893 | 0.813 (3-seed mean) / 0.814 |
| VICReg | 0.893 / 0.899 | 0.772 (3-seed mean) / 0.771 |

### Conclusions

1. **SSL pre-training mainly speeds up convergence.** At the series' 30-epoch budget it adds +0.20 mAP50.
2. **After sufficient training, the benefit falls with the number of labels**: +0.20 (1 %), +0.11 (5 %), +0.05
   (10 %, outside the seed spread of 0.006), +0.02 (25 %), +0.01 (100 %, both seeds in favour of SSL).
3. **SSL saturates at ~200 epochs** on these 4,096 images: 800 epochs raise kNN slightly but not detection.
4. **Redundancy-reduction methods win.** At 200 epochs Barlow Twins is higher and steadier than VICReg at
   30 fine-tuning epochs (seed means 0.813 vs 0.772); the asymmetric methods (BYOL, SimSiam) lag, and DINO did not
   train on this data (loss fixed near ln 4096 = 8.32 from epoch 3 - uniform collapse of the 2-view recipe).
5. **kNN / probe separate strong from weak methods but not the top methods**: Barlow and VICReg 200 differ by
   < 0.01 in kNN / probe and by 0.04 in detection. Supervised Crop36 features are the best representation and a
   weak detection backbone.
6. **Method lesson: conclusions drawn at one short budget flipped** (here and for semi-supervised detection in
   SemiSL_Wave2_FL36). Compare methods at two budgets or more and with two seeds or more.

## Weights

Releases: `gtx1060-wave1a` (4 backbones, Windows run), `rtx4070-wave1` (16 backbones: 50 epochs x 6 methods +
supervised, 200 epochs x 5, 200-epoch seeds 778 / 779 for Barlow and VICReg) and `rtx4070-e800` (Barlow Twins and
VICReg, 800 epochs); SHA-256 in each `SHA256SUMS.txt`.

```python
import torch, torchvision
m = torchvision.models.resnet18(weights=None); m.fc = torch.nn.Identity()
m.load_state_dict(torch.load("ssl_barlow_e200_backbone.pth", weights_only=True))     # strict
# input: grayscale x / 255 repeated to 3 channels, no mean / std
```

## Reproduce

Windows (GTX 1060): `results/gtx1060/` and `code/run_wave1a.bat`. Ubuntu (RTX 4070): one-GPU-job queue
`code/queue_runner.sh` with the job list `code/ubuntu_jobs.txt` (every command that produced `results/rtx4070/`).
Environment: Python 3.10 / 3.11, PyTorch 2.5.1 (cu118 on Windows, cu124 on Ubuntu), torchvision 0.20.1.

Third-party code and data: [THIRD_PARTY.md](THIRD_PARTY.md).
