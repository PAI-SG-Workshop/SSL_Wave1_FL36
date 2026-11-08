# SSL_Wave1_FL36

**Self-supervised pre-training of ResNet-18 on FashionLetters36 - PAI-SG SSL Wave 1**

Part of the *Physical AI* reference series of the **Statera Working Group**. The detection series trained every
detector from scratch; this repository asks whether self-supervised pre-training on the *same unlabeled images*
fixes the weak spots of that protocol.

> **Status: work in progress.** This commit holds the **Wave 1a run on a GTX 1060 6 GB laptop (Windows 10)**:
> 4 methods x 50 epochs and their representation evaluation. The RTX 4070 re-run of all 6 methods, the supervised
> reference, the CenterNet downstream and the 200-epoch runs (Wave 1b) will be added under `results/rtx4070/`.

## Question

Pre-train a ResNet-18 with SSL on the 4,096 FashionLetters36 train images (labels and boxes never used), then
(a) measure the representation on object crops and (b) plug the backbone into CenterNet (FL36, 36 classes,
30 epochs, the common detection protocol; public from-scratch baseline mAP50 0.6437).
The backbone is the CenterNet ResNet-18: `torchvision.models.resnet18` keys = `CenterNet.net.*` keys.

## Methods (solo-learn CIFAR recipes)

| Method | Family | Loss / heads | Optimizer |
| --- | --- | --- | --- |
| SimCLR | contrastive | NT-Xent t 0.2, projector 512-2048-256 | LARS lr 0.4, wd 1e-4 |
| BYOL | non-contrastive, EMA teacher | predictor, EMA tau 0.99 -> 1 | LARS lr 1.0, wd 1e-5 |
| DINO | self-distillation | 4,096 prototypes, student t 0.1, teacher t 0.07, centering 0.9, EMA 0.9995 -> 1 | LARS lr 0.3, wd 1e-6 |
| VICReg | redundancy reduction | expander 2048-2048-2048, weights 25 / 25 / 1 | LARS lr 0.3, wd 1e-4 |
| SimSiam (*) | non-contrastive, stop-grad | predictor at constant lr | SGD lr 0.5, wd 1e-5 |
| Barlow Twins (*) | redundancy reduction | lambda 0.0051, scale 0.1 | LARS lr 0.3, wd 1e-4 |

(*) registered in `code/ssl_train_ext.py`; trained on the RTX 4070 only (results to be added).
All: batch 256, LARS momentum 0.9 / eta 0.02 / clip_lr, no LARS or weight decay on 1-D parameters, warmup
10 epochs from 3e-5 then cosine (per step), fp32, seed 777, 50 epochs of 32,768 samples (8 per image).

**Differences from the recipes ([Unified]):** ImageNet-stem ResNet-18 (to match CenterNet); grayscale input as
3 identical channels, x / 255, no mean / std (as the detection pipeline); two 96 x 96 views per sample, each a
RandomResizedCrop of one 128-256 px region (at least 2 % bright pixels) of a 608 x 608 image; **no horizontal flip**
(letters are not mirror-invariant); saturation / hue / grayscale skipped (no-ops on grayscale data);
DINO with 2 global views (as the solo-learn CIFAR recipe), no multi-crop.

## Evaluation

- **Crop36** (`code/prep_ssl_data.py`): every FL36 object cut as a square (1.15 x box), resized to 96 x 96;
  26,716 train / 6,699 test crops, 36 classes. Evaluation only.
- **kNN** k 20, cosine, t 0.07; **linear probe** on standardised frozen features (Adam 1e-3, 100 epochs).
- **BN re-estimated** (`code/final_eval.py`): kNN / probe after re-estimating the BN statistics on the Crop36
  train images (labels unused). Methods are compared on this column: under the stored statistics a *random*
  ResNet-18 looks strong (kNN 0.785), because its default BN (mean 0, var 1) passes the raw pixel scale through;
  with re-estimated BN the random backbone drops to 0.642 while SimCLR (5 epochs) stays at 0.733.

## Results - GTX 1060 6 GB laptop, Windows 10 (Wave 1a)

From the training logs (stored BN). The BN-re-estimated table of the same backbones is
`results/gtx1060/eval/wave1a_final_eval.md`.

| Backbone | kNN top-1 | Linear probe top-1 | Probe macro F1 |
| --- | --- | --- | --- |
| random init | 0.7852 | 0.8137 | 0.8122 |
| SimCLR | 0.8303 | 0.8924 | 0.8910 |
| BYOL | 0.6111 | 0.7976 | 0.7955 |
| DINO | 0.7577 | 0.8037 | 0.8025 |
| **VICReg** | **0.8731** | **0.9043** | **0.9029** |

Findings:

1. **VICReg > SimCLR >> BYOL** after 50 epochs. kNN curves (epochs 10 / 20 / 30 / 40 / 50): SimCLR
   0.756 / 0.817 / 0.815 / 0.832 / 0.830, VICReg 0.836 / 0.852 / 0.862 / 0.871 / 0.873, BYOL 0.575 / 0.593 /
   0.608 / 0.613 / 0.611.
2. **DINO did not train (uniform collapse).** The loss stays at 8.28-8.29 from epoch 3 on, next to
   ln 4096 = 8.32 (teacher output spread evenly over the prototypes), and kNN is flat at 0.75-0.76 from epoch 10.
   The backbone-level collapse index used during training (0.49-0.50) does not detect this head-level collapse.
   The 2-view solo-learn CIFAR recipe does not transfer to this data as is.
3. **Thermal throttling.** Under load the GPU reaches 89-92 C within 30 s and reports SW Thermal Slowdown
   (0x20); the SM clock drops to 139-450 MHz. Two data-loader workers and a 70 % CPU cap (shared heat sink)
   brought one epoch from 6.5 to ~4.5 min; hot periods still reach 10-21 min. CenterNet (30 epochs) would take
   ~13 h on this machine and was moved to the RTX 4070.
4. **Cross-hardware reproducibility.** The RTX 4070 re-run (same code and seed) gives stored-BN kNN SimCLR
   0.836, BYOL 0.627, VICReg 0.875, DINO 0.761 - within 0.016 of the GTX 1060 values.

Weights: Release `gtx1060-wave1a` - `ssl_<method>_backbone_final.pth` (torchvision ResNet-18 keys, `fc` removed,
SHA-256 in `SHA256SUMS.txt`).

```python
import torch, torchvision
m = torchvision.models.resnet18(weights=None); m.fc = torch.nn.Identity()
m.load_state_dict(torch.load("ssl_vicreg_backbone_final.pth", weights_only=True))   # strict
# input: grayscale x / 255 repeated to 3 channels, no mean / std
```

For CenterNet: copy `code/train_image_centernet_ssl.py` into
[OD_Image_1-Stage_CenterNet](https://github.com/Statera-Workshop/OD_Image_1-Stage_CenterNet) (v1.1) and run
`python train_image_centernet_ssl.py --pretrained <backbone.pth>` (same protocol; only the backbone initialisation
differs; the script refuses a checkpoint whose backbone keys do not all load).

## Reproduce (Windows, GTX 1060 run)

```bat
rem 1. FL36: generate with the CenterNet repository (EMNIST: see its README if NIST refuses the download)
set PHYSICAL_AI_ROOT=C:\SSL_Wave1
python data_processing_fl36.py
python code\fl36_count.py                    (from the CenterNet folder: RESULT OK = 26,716 / 6,699 objects)
rem 2. SSL inputs and the random-init reference
python code\prep_ssl_data.py                 (RESULT OK; md5 of the arrays in ssl_data.json)
python code\ssl_train.py --eval-only --init random
rem 3. Queue (4 methods x 50 epochs) and the final table
code\run_wave1a.bat
python code\final_eval.py
```

Environment: Python 3.11, PyTorch 2.5.1 + cu118, torchvision 0.20.1, NVIDIA driver 456.71 (CUDA 11.1).
Logs, GPU monitor, data checks and the environment report are in `results/gtx1060/`.

Third-party code and data: see [THIRD_PARTY.md](THIRD_PARTY.md).
