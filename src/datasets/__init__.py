#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Datasets Module
----------------------------------------------------------------
Medical image segmentation datasets and utilities.
- ALBDataset: Albumentations-based segmentation dataset
- PTVDataset: Generic UNet segmentation dataset
- mask_map_class_id: Mask to class ID mapping utility
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .seg_ALB import ALBDataset
from .seg_map import mask_map_class_id
from .seg_PTV import PTVDataset

__all__ = [
    "ALBDataset",
    "mask_map_class_id",
    "PTVDataset",
]
