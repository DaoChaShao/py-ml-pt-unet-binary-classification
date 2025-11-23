#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from albumentations.core.composition import Compose
from albumentations import Resize, Normalize, ToTensorV2
from numpy import ndarray, array, uint8
from PIL import Image
from pathlib import Path
from random import randint
from torch import load, device, no_grad, sigmoid

from src.nets.standard4 import Standard4LayersUNet
from src.utils.config import CONFIG
from src.utils.decorator import timer
from src.utils.helper import Timer
from src.utils.stats import load_paths


@timer
def display_mask(mask: Image.Image | ndarray) -> None:
    # Convert mask to numpy array
    arr = array(mask)
    # Enhance visibility by scaling pixel values
    enhanced_arr = (arr * 255).astype(uint8)
    enhanced_mask = Image.fromarray(enhanced_arr)
    enhanced_mask.show()


def preprocess_data():
    """ Prepare Data Function """
    base: Path = Path(CONFIG.FILEPATH.DATASET_TEST)
    # print(base)
    paths_images, paths_masks = load_paths(base)

    return paths_images, paths_masks


def prepare_data():
    """ Prepare Data Function """
    paths_images, paths_masks = preprocess_data()


def main() -> None:
    """ Main Function """
    with Timer("UNet Predicting"):
        # prepare_data()
        params: Path = Path(CONFIG.FILEPATH.MODEL)
        """
        ****************************************************************
        Segmentation Evaluation Metrics
        ----------------------------------------------------------------
        TP: 6829070, FP:  268228
        FN: 382294, TN: 35774168
        
        Precision: 0.9622
        Recall:    0.9470
        F1-Score:  0.9545
        ****************************************************************
        
        ****************************************************************
        Binary Segmentation Metrics
        ----------------------------------------------------------------
        - Background IoU:               98.2141%
        - Foreground IoU:               91.3027%
        - mean Intersection over Union: 94.7584%
        ****************************************************************
        
        Epoch [33/200]:
        - Train Loss:                    0.0204
        - Valid Loss:                    0.1939
        - Pixel Accuracy:                98.4960%
        """

        if params.exists():
            print(f"Model {params.name} already EXISTS!")

            test_transformer = Compose([
                Resize(height=CONFIG.PREPROCESSOR.IMAGE_HEIGHT, width=CONFIG.PREPROCESSOR.IMAGE_WIDTH),
                Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2()
            ])

            paths_images, paths_masks = preprocess_data()
            index: int = randint(0, len(paths_images) - 1)
            # print(f"Random index selected: {index}")
            # print(f"Randomly selected image path: {paths_images[index]}")
            # print(f"Corresponding mask path: {paths_masks[index]}")
            # print(type(paths_images[index]), type(paths_masks[index]))
            img_image: Image.Image = Image.open(paths_images[index]).convert("RGB")
            img_mask: Image.Image = Image.open(paths_masks[index])
            # img_image.show()
            display_mask(img_mask)

            arr_image: ndarray = array(img_image)
            transformed = test_transformer(image=arr_image)
            # Add batch dimension
            image_tensor = transformed["image"].unsqueeze(0).to(device(CONFIG.HYPERPARAMETERS.ACCELERATOR))
            # print(f"Image tensor shape: {image_tensor.shape}")

            channel, height, width = image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]
            model = Standard4LayersUNet(
                in_channels=channel,
                n_classes=CONFIG.UNET_PARAMS.SEG_CLASSES,
                height=height,
                width=width,
                channels=128
            )
            # Load model weights from GPU to CPU or specified device using "map_location"
            state_dict = load(params, map_location=device(CONFIG.HYPERPARAMETERS.ACCELERATOR))
            model.load_state_dict(state_dict)
            model.to(device(CONFIG.HYPERPARAMETERS.ACCELERATOR))
            model.eval()
            print(f"Model {params.name} loaded!")

            with no_grad():
                output = model(image_tensor)
                prediction = (sigmoid(output) > 0.5).cpu().numpy()

            # Drop batch dimension and convert to Image
            pred_mask = prediction.squeeze()
            pred_mask_arr = (pred_mask * 255).astype(uint8)
            pred_mask_img = Image.fromarray(pred_mask_arr)
            # print(f"Prediction shape: {pred_mask.shape}")
            # print(f"Prediction unique values: {set(pred_mask_arr.flatten())}")

            pred_mask_img.show()

        else:
            print(f"Model {params.name} does NOT exist!")


if __name__ == "__main__":
    main()
