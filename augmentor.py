#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/22 17:48
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   augmentor.py
# @Desc     :   

from PIL import Image
from albumentations.core.composition import Compose
from albumentations import (HorizontalFlip, VerticalFlip,
                            RandomBrightnessContrast, HueSaturationValue, RGBShift, RandomGamma,
                            GaussNoise, Blur, MotionBlur, MedianBlur,
                            Affine, OpticalDistortion, GridDistortion, ElasticTransform)
from numpy import ndarray, array
from pathlib import Path
from random import randint

from src.utils.config import CONFIG
from src.utils.decorator import timer
from src.utils.stats import load_paths


@timer
def augment_data(base_directory: Path, augmented_ratio: int):
    """ Generate Augmented Data Function
    - Delete the new files
        - rm data/train/images/*_aug*.png
        - rm data/train/masks/*_aug*.png
    :param base_directory: Path - Base directory containing images and masks folders
    :param augmented_ratio: int - Number of augmentations to generate per original image
    """
    # Set up augmentation pipeline
    transformer = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.3),
        RandomBrightnessContrast(p=0.5),
        HueSaturationValue(p=0.5),
        RGBShift(p=0.5),
        RandomGamma(p=0.5),
        GaussNoise(p=0.5),
        Blur(blur_limit=3, p=0.3),
        MotionBlur(blur_limit=3, p=0.3),
        MedianBlur(blur_limit=3, p=0.3),
        Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
        OpticalDistortion(p=0.3),
        GridDistortion(p=0.3),
        ElasticTransform(p=0.3),
    ])

    images_dir: Path = base_directory / "images"
    masks_dir: Path = base_directory / "masks"

    if images_dir.exists() and masks_dir.exists():
        print(f"Images Folder and Masks Folder EXISTS!")
    else:
        print(f"Images Folder and Masks Folder NOT EXISTS!")

    image_paths, mask_paths = load_paths(base_directory)
    assert len(image_paths) == len(mask_paths), "Number of images and masks must be equal."
    # index: int = randint(0, len(image_paths) - 1)
    # print(f"Randomly Selected Index: {index}")
    # print(f"Image Path: {image_paths[index]}")
    # print(f"Mask Path: {mask_paths[index]}")
    amount: int = len(image_paths)

    # Calculate how many augmentations per original image are needed
    # print(f"Original dataset has {amount} samples")
    # print(f"Each image will be augmented {augmented_ratio} times")
    # print(f"Total new samples to generate: {amount * augmented_ratio}")

    counter: int = 0
    for i, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        # Transform original image and mask into arrays
        image_arr: ndarray = array(Image.open(image_path).convert("RGB"))
        mask_arr: ndarray = array(Image.open(mask_path))
        # print(image_arr.shape, mask_arr.shape)
        assert image_arr.shape[:2] == mask_arr.shape[:2], "Original image and mask dimensions is NOT match."

        # Get file names without extensions
        image_name: str = Path(image_path).stem
        mask_name: str = Path(mask_path).stem
        mask_name: str = mask_name.split("_")[0]
        # print(image_name)
        # print(mask_name)

        for j in range(augmented_ratio):
            # Apply augmentation
            augmented = transformer(image=image_arr, mask=mask_arr)
            aug_image: ndarray = augmented["image"]
            aug_mask: ndarray = augmented["mask"]

            # Generate new file names
            new_image_name: str = f"{image_name}_aug{j}.png"
            new_mask_name: str = f"{mask_name}_aug{j}_mask.png"
            # print(new_image_name)
            # print(new_mask_name)

            # Save augmented images and masks
            aug_image_pil: Image.Image = Image.fromarray(aug_image)
            aug_mask_pil: Image.Image = Image.fromarray(aug_mask)
            aug_image_pil.save(images_dir / new_image_name)
            aug_mask_pil.save(masks_dir / new_mask_name)

            counter += 1

            print(f"Augmentation completed! [{counter}/{amount * augmented_ratio}] new samples generated.")


def main() -> None:
    """ Main Function """
    base: Path = Path(CONFIG.FILEPATH.DATASET_TRAIN)
    # print(base)

    augmented_ratio: int = 10
    augment_data(base, augmented_ratio)


if __name__ == "__main__":
    main()
