#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :

from albumentations.core.composition import Compose
from albumentations import (Resize, HorizontalFlip, VerticalFlip, OneOf,
                            RandomBrightnessContrast, HueSaturationValue, RGBShift, RandomGamma,
                            GaussNoise, Blur, MotionBlur, MedianBlur,
                            Affine, OpticalDistortion, GridDistortion, ElasticTransform,
                            Normalize, ToTensorV2)
from numpy import array, ndarray, unique, uint8, set_printoptions
from PIL import Image
from pathlib import Path
from random import randint
from sys import maxsize
from torch import optim, tensor, nn
from torchvision import transforms

from src.criteria.dice import DiceBCELoss
from src.criteria.focal import ForegroundFocalLoss
from src.criteria.DNF import ComprehensiveSegLoss
from src.criteria.edge import EdgeAwareLoss
from src.datasets.seg_PTV import mask_map_class_id, PTVDataset
from src.datasets.seg_ALB import ALBDataset
from src.nets.standard4 import Standard4LayersUNet
from src.nets.standard5 import Standard5LayersUNet
from src.trainers.iou import UNetSegmentationTrainer
from src.utils.config import CONFIG
from src.utils.decorator import timer
from src.utils.PT import TorchDataLoader, TorchRandomSeed
from src.utils.stats import load_paths, split_paths

set_printoptions(threshold=maxsize)


@timer
def get_classes(mask: Image.Image) -> tuple:
    arr = array(mask)

    unique_classes: ndarray[tuple] = unique(arr)

    seg_arr: ndarray = (arr > 0).astype(uint8)
    seg_classes: ndarray[tuple] = unique(seg_arr)

    return unique_classes, seg_classes


@timer
def display_mask(mask: Image.Image | ndarray) -> None:
    # Convert mask to numpy array
    arr = array(mask)
    # Enhance visibility by scaling pixel values
    enhanced_arr = (arr * 255).astype(uint8)
    enhanced_mask = Image.fromarray(enhanced_arr)
    enhanced_mask.show()


@timer
def check_dataset_status(image_paths) -> tuple[float, float]:
    heights: list[int] = []
    widths: list[int] = []

    for path in image_paths:
        with Image.open(path) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)

    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)

    print(f"The average image size: {avg_height:.1f} x {avg_width:.1f}")
    print(f"The range of image: {min(heights)}x{min(widths)} - {max(heights)}x{max(widths)}")

    return avg_height, avg_width


@timer
def check_mask_distribution(mask_paths, sample_size: int = 10):
    """ Check the distribution of foreground and background pixels in the masks """
    foreground_pixels: int = 0
    total_pixels: int = 0

    for i in range(min(sample_size, len(mask_paths))):
        mask = Image.open(mask_paths[i])
        arr = array(mask)
        foreground_pixels += (arr > 0).sum()
        total_pixels += arr.size

    foreground_ratio = foreground_pixels / total_pixels

    print(f"The foreground pixels occurred {foreground_ratio * 100:.1f} % in the sampled masks.")

    return foreground_ratio


def preprocess_data():
    base: Path = Path(CONFIG.FILEPATH.DATASET_TRAIN)
    # print(f"The path of training dataset: {base}")

    paths_images, paths_masks = load_paths(base)
    # index: int = randint(0, len(paths_images) - 1)
    # print(f"Random index selected: {index}")
    # print(f"Randomly selected image path: {paths_images[index]}")
    # print(f"Corresponding mask path: {paths_masks[index]}")
    # print(type(paths_images[index]), type(paths_masks[index]))

    train_image_paths, train_mask_paths, valid_image_paths, valid_mask_paths = split_paths(paths_images, paths_masks)
    # index_train: int = randint(0, len(train_image_paths) - 1)
    # print(f"Random index for training set: {index_train}")
    # print(f"Training image path: {train_image_paths[index_train]}")
    # print(f"Training mask path: {train_mask_paths[index_train]}")
    # train_image: Image.Image = Image.open(train_image_paths[index_train])
    # train_image.show()
    # train_mask: Image.Image = Image.open(train_mask_paths[index_train])
    # display_mask(train_mask)
    # ins_classes, seg_classes = get_classes(train_mask)
    # print(ins_classes)
    # print(seg_classes)
    # mask_arr, mask_seg = mask_map_class_id(train_mask)
    # display_mask(mask_arr)
    # print(mask_seg)

    # print(f"Random index for validation set: {index_valid}")
    # print(f"Validation image path: {valid_image_paths[index_valid]}")
    # print(f"Validation mask path: {valid_mask_paths[index_valid]}")

    return train_image_paths, train_mask_paths, valid_image_paths, valid_mask_paths


def prepare_dataset():
    train_image_paths, train_mask_paths, valid_image_paths, valid_mask_paths = preprocess_data()

    # Check mask distribution
    check_mask_distribution(train_mask_paths, sample_size=20)

    # Get the average image size in the dataset
    avg_height, avg_width = check_dataset_status(train_image_paths)

    # Setup image enhancements
    # img_transformer = transforms.Compose([
    #     transforms.Resize((CONFIG.PREPROCESSOR.IMAGE_HEIGHT, CONFIG.PREPROCESSOR.IMAGE_WIDTH)),
    #     transforms.RandomHorizontalFlip(0.5),
    #     transforms.RandomVerticalFlip(0.3),
    #     transforms.RandomRotation(15),
    #     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    #     transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])
    # mask_transformer = transforms.Compose([
    #     transforms.Resize((CONFIG.PREPROCESSOR.IMAGE_HEIGHT, CONFIG.PREPROCESSOR.IMAGE_WIDTH)),
    #     transforms.ToTensor(),
    # ])
    train_transformer = Compose([
        Resize(height=CONFIG.PREPROCESSOR.IMAGE_HEIGHT, width=CONFIG.PREPROCESSOR.IMAGE_WIDTH),
        # HorizontalFlip(p=0.5),
        # VerticalFlip(p=0.3),
        # RandomBrightnessContrast(p=0.5),
        # HueSaturationValue(p=0.5),
        # RGBShift(p=0.5),
        # RandomGamma(p=0.5),
        # GaussNoise(p=0.5),
        # Blur(blur_limit=3, p=0.3),
        # MotionBlur(blur_limit=3, p=0.3),
        # MedianBlur(blur_limit=3, p=0.3),
        # Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
        # OpticalDistortion(p=0.3),
        # GridDistortion(p=0.3),
        # ElasticTransform(p=0.3),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ], is_check_shapes=True)
    valid_transformer = Compose([
        Resize(height=CONFIG.PREPROCESSOR.IMAGE_HEIGHT, width=CONFIG.PREPROCESSOR.IMAGE_WIDTH),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ], is_check_shapes=True)

    # Setup datasets
    # dataset_train = PTVDataset(
    #     image_paths=train_image_paths,
    #     mask_paths=train_mask_paths,
    #     img_transformer=img_transformer,
    #     mask_transformer=mask_transformer
    # )
    # dataset_valid = PTVDataset(
    #     image_paths=valid_image_paths,
    #     mask_paths=valid_mask_paths,
    #     img_transformer=img_transformer,
    #     mask_transformer=mask_transformer
    # )
    dataset_train = ALBDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        transformer=train_transformer
    )
    dataset_valid = ALBDataset(
        image_paths=valid_image_paths,
        mask_paths=valid_mask_paths,
        transformer=valid_transformer
    )
    # index_train_dataset: int = randint(0, len(dataset_train) - 1)
    # index_valid_dataset: int = randint(0, len(dataset_valid) - 1)
    # img_train, mask_train = dataset_train[index_train_dataset]
    # img_valid, mask_valid = dataset_valid[index_valid_dataset]
    # print(f"Transformed train image shape: {img_train.shape}, mask shape: {mask_train.shape}")
    # print(f"Transformed test image shape: {img_valid.shape}, mask shape: {mask_valid.shape}")
    # print(mask_train.numpy(), mask_valid.numpy(), sep="\n")

    # Set up dataloader
    dataloader_train = TorchDataLoader(
        dataset=dataset_train,
        batch_size=CONFIG.PREPROCESSOR.BATCHES,
        shuffle_status=CONFIG.PREPROCESSOR.SHUFFLE_STATUS,
    )
    dataloader_valid = TorchDataLoader(
        dataset=dataset_valid,
        batch_size=CONFIG.PREPROCESSOR.BATCHES,
        shuffle_status=CONFIG.PREPROCESSOR.SHUFFLE_STATUS,
    )

    # print(f"Number of training batches: {len(dataloader_train)}")
    # print(f"Number of validation batches: {len(dataloader_valid)}")

    return dataloader_train, dataloader_valid


def main() -> None:
    """ Main Function """
    with TorchRandomSeed("Training UNet Segmentation Model"):
        # prepare_dataset()
        print("1.Preparing dataset...")
        train, valid = prepare_dataset()

        # train_index: int = randint(0, len(train) - 1)
        # valid_index: int = randint(0, len(valid) - 1)
        # print(f"Random training batch index: {train_index}")
        # print(f"Random validation batch index: {valid_index}")
        # print(f"Training batch image shape: {train[train_index][0].shape}, mask shape: {train[train_index][1].shape}")
        # print(f"Valid batch image shape: {valid[valid_index][0].shape}, mask shape: {valid[valid_index][1].shape}")
        print("2.Dataset preparation completed.")

        print("3.Setting up model...")
        # Set up a model
        channels, height, width = train[0][0].shape
        # print(channels, height, width)
        model = Standard4LayersUNet(channels, CONFIG.UNET_PARAMS.SEG_CLASSES, height, width, channels=128)
        print("4.Model setup completed.")

        print("5.Setting up optimiser...")
        # Set up an optimiser and loss function
        optimiser = optim.AdamW(model.parameters(), lr=CONFIG.HYPERPARAMETERS.ALPHA, weight_decay=1e-4)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=CONFIG.HYPERPARAMETERS.EPOCHS)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=5)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=5)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=10, T_mult=2, eta_min=1e-6)
        print("6.Optimiser setup completed.")

        print("7.Setting up loss function...")
        # Compute class balance weights
        # - foreground pixel occupied 18.6%, which is 0.186
        # - background pixel occupied 81.4%, which is 1 - 0.186
        # - pos_weight = foreground / background = 81.4% / 18.6% ≈ 4.38
        pos_weight = tensor([(1 - 0.186) / 0.186]).to(CONFIG.HYPERPARAMETERS.ACCELERATOR)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Background IoU: 87.2620% | Foreground IoU: 59.2802%
        # criterion = DiceBCELoss(pos_weight=pos_weight)  # - Background IoU: 88.4831% | Foreground IoU: 61.2048%
        # criterion = ForegroundFocalLoss(alpha=0.9, gamma=1.2)  # Background IoU: 89.2834% | Foreground IoU: 60.5484%
        criterion = ComprehensiveSegLoss(
            pos_weight=pos_weight,
            alpha=0.8,
            gamma=2.0,
            weights_ratio=[0.4, 0.3, 0.3]
        )  # - Background IoU: 89.5123% | Foreground IoU: 62.1154%
        criterion = EdgeAwareLoss(pos_weight=pos_weight, edge_weight=3.0)
        print("8.Loss function setup completed.")

        print("9.Setting up trainer and start to train...\n")
        # Initialise a trainer
        trainer = UNetSegmentationTrainer(model, optimiser, criterion, scheduler, CONFIG.HYPERPARAMETERS.ACCELERATOR)
        trainer.fit(train, valid, CONFIG.HYPERPARAMETERS.EPOCHS, CONFIG.FILEPATH.MODEL)
        print("10.Training completed.")


if __name__ == "__main__":
    main()
