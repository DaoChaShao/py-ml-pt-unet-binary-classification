#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/21 16:14
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   seg_A.py
# @Desc     :   

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

        # mask_map_class_id 是你原来自定义函数
        converted_mask, _ = mask_map_class_id(mask)
        converted_mask: Image.Image = Image.fromarray(converted_mask)

        if self._mask_transformer:
            mask: Tensor = self._mask_transformer(converted_mask)
        else:
            mask: Tensor = from_numpy(array(converted_mask)).float()

        # 强化小目标权重：可选，对 Pedestrian 数据集小目标增加权重
        # mask = mask * 2.0  # 训练时可尝试

        return image, mask


if __name__ == "__main__":
    pass
