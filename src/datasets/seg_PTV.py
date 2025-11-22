#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 02:40
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seg_PTV.py
# @Desc     :   

from numpy import array, ndarray, float32, unique
from PIL import Image
from torch import Tensor, from_numpy
from torch.utils.data import Dataset


def mask_map_class_id(mask: Image.Image) -> tuple[ndarray, ndarray]:
    """ Convert mask pixel values to class IDs """
    # Transfer PIL Image to numpy array - shape: (H, W, C) or (H, W)
    arr = array(mask)

    # Calculate the frequency of each class in the mask
    # print(f"Mask Distribution:")
    # classes_before, freq_before = unique(arr, return_counts=True)
    # for cls, freq in zip(classes_before, freq_before):
    #     percentage = freq / arr.size * 100
    #     print(f"- class {cls}: {freq} pixels ({percentage:.2f}%)")

    # Convert to binary segmentation: foreground (1) and background (0)
    seg_arr: ndarray = (arr > 0).astype(float32)
    classes = unique(seg_arr)

    # Calculate the frequency of each class after conversion
    # print(f"Converted Mask Distribution:")
    # classes_after, freq_after = unique(seg_arr, return_counts=True)
    # for cls, freq in zip(classes_after, freq_after):
    #     percentage = freq / seg_arr.size * 100
    #     print(f"- class {cls}: {freq} pixels ({percentage:.2f}%)")

    return seg_arr, classes


# class UNetDataset(Dataset):
#     """ A custom PyTorch Dataset class for UNet model """
#
#     def __init__(self, image_paths: list[str], mask_paths: list[str], img_transformer=None, mask_transformer=None):
#         """ Initialise the UNetDataset class """
#         assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
#
#         self._image_paths = image_paths
#         self._mask_paths = mask_paths
#         self._img_transformer = img_transformer
#         self._mask_transformer = mask_transformer
#
#     def __len__(self) -> int:
#         """ Return the total number of samples in the dataset """
#         return len(self._image_paths)
#
#     def __getitem__(self, index: int) -> tuple:
#         """ Return a single (feature, label) pair or a batch via slice """
#         assert index < len(self._image_paths), f"Index {index} out of range."
#
#         image: Image.Image = Image.open(self._image_paths[index]).convert("RGB")
#         mask: Image.Image = Image.open(self._mask_paths[index])
#
#         if self._img_transformer:
#             image: Tensor = self._img_transformer(image)
#         else:
#             image: Tensor = from_numpy(array(image)).permute(2, 0, 1).float() / 255.0
#
#         converted_mask, _ = mask_map_class_id(mask)
#         converted_mask: Image.Image = Image.fromarray(converted_mask)
#
#         if self._mask_transformer:
#             mask: Tensor = self._mask_transformer(converted_mask)
#         else:
#             mask: Tensor = from_numpy(array(converted_mask)).float()
#
#         return image, mask

class PTVDataset(Dataset):
    """ A custom PyTorch Dataset class for UNet model """

    def __init__(self, image_paths: list[str], mask_paths: list[str], img_transformer=None, mask_transformer=None):
        assert len(image_paths) == len(mask_paths), "Number of images and masks must match"
        self._image_paths = image_paths
        self._mask_paths = mask_paths
        self._img_transformer = img_transformer
        self._mask_transformer = mask_transformer

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> tuple:
        assert index < len(self._image_paths), f"Index {index} out of range."

        image: Image.Image = Image.open(self._image_paths[index]).convert("RGB")
        mask: Image.Image = Image.open(self._mask_paths[index])

        if self._img_transformer:
            image: Tensor = self._img_transformer(image)
        else:
            image: Tensor = from_numpy(array(image)).permute(2, 0, 1).float() / 255.0

        converted_mask, _ = mask_map_class_id(mask)
        converted_mask: Image.Image = Image.fromarray(converted_mask)

        if self._mask_transformer:
            mask: Tensor = self._mask_transformer(converted_mask)
        else:
            mask: Tensor = from_numpy(array(converted_mask)).float()

        return image, mask


if __name__ == "__main__":
    pass
