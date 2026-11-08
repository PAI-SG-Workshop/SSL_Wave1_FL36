# Third-party code and data

| Component | Use here | License |
| --- | --- | --- |
| [solo-learn](https://github.com/vturrisi/solo-learn) | LARS optimizer (`code/ssl_train.py`, ported), CIFAR recipe values (learning rates, weight decay, projector sizes, augmentation probabilities, EMA schedules, DINO / Barlow / VICReg / SimSiam settings) | MIT |
| [SimCLR](https://github.com/google-research/simclr), [BYOL](https://github.com/deepmind/deepmind-research/tree/master/byol), [DINO](https://github.com/facebookresearch/dino), [VICReg](https://github.com/facebookresearch/vicreg), [SimSiam](https://github.com/facebookresearch/simsiam), [Barlow Twins](https://github.com/facebookresearch/barlowtwins) | loss definitions re-implemented from the papers / official code | see each repository |
| [torchvision](https://github.com/pytorch/vision) | ResNet-18 definition | BSD-3-Clause |
| [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | object glyphs of FashionLetters36 | MIT |
| [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) (NIST) | letter glyphs of FashionLetters36 | NIST terms of use |
| [OD_Image_1-Stage_CenterNet](https://github.com/Statera-Workshop/OD_Image_1-Stage_CenterNet) | FL36 generator and loader, CenterNet downstream (`code/train_image_centernet_ssl.py` subclasses its model) | see that repository |

The pre-trained weights in the Releases were trained from scratch by the Working Group on FashionLetters36 only.
