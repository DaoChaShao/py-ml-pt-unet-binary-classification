#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:12
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :

from PIL import Image
from albumentations.core.composition import Compose
from albumentations import Resize, Normalize, ToTensorV2
from numpy import array, ndarray, unique, uint8, set_printoptions
from pathlib import Path
from pprint import pprint
from random import randint
from sys import maxsize
from torch import optim, tensor, nn

from src.criteria.dice import DiceBCELoss
from src.criteria.focal import FocalLoss
from src.criteria.DNF import ComprehensiveSegLoss
from src.criteria.edge import EdgeAwareLoss
from src.datasets.seg_PTV import mask_map_class_id, PTVDataset
from src.datasets.seg_ALB import ALBDataset
from src.nets.standard4 import Standard4LayersUNet
from src.nets.standard5 import Standard5LayersUNet
from src.trainers.binary_seg import UNetSegmentationTrainer
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
def check_dataset_status(image_paths) -> None:
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

    foreground_ratio: float = foreground_pixels / total_pixels

    print(f"The foreground pixels occurred {foreground_ratio * 100:.1f} %({foreground_ratio}) in the sampled masks.")

    return foreground_ratio


def preprocess_data():
    base: Path = Path(CONFIG.FILEPATH.DATASET_TRAIN)
    # print(f"The path of training dataset: {base}")

    paths_images, paths_masks = load_paths(base)
    # pprint(paths_images[:11])
    # pprint(paths_masks[:11])
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
    weight: float = check_mask_distribution(train_mask_paths, sample_size=len(train_mask_paths))

    # Get the average image size in the dataset
    check_dataset_status(train_image_paths)

    # Setup image enhancements
    train_transformer = Compose([
        Resize(height=CONFIG.PREPROCESSOR.IMAGE_HEIGHT, width=CONFIG.PREPROCESSOR.IMAGE_WIDTH),
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

    return dataloader_train, dataloader_valid, weight


def main() -> None:
    """ Main Function """
    with TorchRandomSeed("Training UNet Segmentation Model"):
        # prepare_dataset()
        print("1.Preparing dataset...")
        train, valid, weight = prepare_dataset()

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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=5)
        print("6.Optimiser setup completed.")

        print("7.Setting up loss function...")
        # Compute class balance weights
        pos_weight = tensor([(1 - weight) / weight]).to(CONFIG.HYPERPARAMETERS.ACCELERATOR)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = DiceBCELoss(pos_weight=pos_weight)
        # criterion = FocalLoss(alpha=0.8, gamma=2.0)
        # criterion = ComprehensiveSegLoss(
        #     pos_weight=pos_weight,
        #     alpha=0.8,
        #     gamma=2.0,
        #     weights_ratio=[0.4, 0.3, 0.3]
        # )
        criterion = EdgeAwareLoss(pos_weight=pos_weight, edge_weight=3.0)
        print("8.Loss function setup completed.")

        print("9.Setting up trainer and start to train...\n")
        # Initialise a trainer
        trainer = UNetSegmentationTrainer(model, optimiser, criterion, scheduler, CONFIG.HYPERPARAMETERS.ACCELERATOR)
        trainer.fit(train, valid, CONFIG.HYPERPARAMETERS.EPOCHS, CONFIG.FILEPATH.MODEL)
        print("10.Training completed.")


if __name__ == "__main__":
    main()
