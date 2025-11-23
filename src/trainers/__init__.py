#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:19
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Neural Nets Trainer Module
----------------------------------------------------------------
Training utilities for neural network models.
- UNetSegmentationTrainer: Trainer for UNet-based image segmentation tasks
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .binary_seg import UNetSegmentationTrainer

__all__ = [
    "UNetSegmentationTrainer",
]
