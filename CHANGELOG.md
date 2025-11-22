<!-- insertion marker -->
<a name="0.1.0"></a>

## [0.1.0](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/compare/98ada3df53b728277995029fefb9bff6fde8c8a3...0.1.0) (2025-11-22)

### Features

- add uv.lock file ([4e92467](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/4e924671a5b8e8e3232d19d914b149e48be1dc1d))
- add data handling functions for CSV, JSON, and directory paths in stats.py ([a750471](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/a75047168555ac6a97512f5d430d2dc2fbe325ef))
- implement Standard5LayersUNet with double convolution, downsampling, and upsampling blocks ([341894a](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/341894a76a44b3b7884054892e0fd211570ed99a))
- rename and refactor Standard5LayersUNet to Standard4LayersUNet with improved architecture and added bilinear upsampling option ([7904c58](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/7904c581b74ce02e3522134ca889e19aead3bae0))
- refactor dataset class for improved image and mask handling in seg_ALB.py ([33e8488](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/33e8488dbf90a040d55e5aa7b2ba1e83cf69bb64))
- update pyproject.toml with new dependencies and git-changelog configuration ([10e1202](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/10e120293a3767b56c48c4fcd9dd685eb3ff10de))
- add TorchRandomSeed, device checking, and custom DataLoader for enhanced PyTorch integration ([546fbdd](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/546fbdded0755ea5ff306a667223254e6739dca7))
- implement data preparation and prediction functionality in predictor.py ([0f26907](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/0f26907646883891d6d6f4a869bc120c8d2e537f))
- implement dataset preparation and model setup for UNet segmentation ([ff0a60a](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/ff0a60aeda533d5155750b39802d3985a4c64c5b))
- add text highlighting functions and formatting utilities ([6166050](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/61660506c1203a925001f0d140fcccc70c8ba38b))
- add Timer, Beautifier, and RandomSeed classes for enhanced code block management ([e1ece4c](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/e1ece4cab9439768c1662f9a7b7c93985c20480c))
- implement ForegroundFocalLoss class for enhanced binary segmentation loss ([a71dd04](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/a71dd04fe03bf191f1821c3106cab2493522d338))
- add EdgeAwareLoss class for enhanced loss calculation in segmentation tasks ([39265bc](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/39265bcd69dc2835427701e9cef1904da7e549b7))
- add ComprehensiveSegLoss class for advanced loss calculation in segmentation tasks ([0b3b0a3](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/0b3b0a3308bc4bebea935b5ad7636d80643b6c93))
- implement DiceBCELoss class for improved loss calculation ([2d3d6c9](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/2d3d6c931bc9fe1ec75415692b91c903aa901341))
- add timer and beautifier decorators for function performance and output enhancement ([32aebfe](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/32aebfe8b93f81537836adba0732ec477bde94f3))

### Bug Fixes

- correct file name in header comment and clean up unused code in seg_PTV.py ([64aff0e](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/64aff0ec7ed9c1c172317b6dd53f37b901476648))
- update file header information in iou.py ([4334474](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/43344741a78af98539927bf3b13ddee2350f5958))

### Docs

- add Chinese translation of the README with project overview, privacy, and environment setup instructions ([73627de](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/73627de8f9a7db945cf6fc03affce135c31d3706))
- add instructions for generating requirements.txt from pyproject.toml ([057f158](https://github.com/DaoChaShao/py-ml-pt-unet-binary-classification/commit/057f158f1220660b2fc0cfe63c5c636eed3a7625))

