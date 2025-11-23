#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:22
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py.py
# @Desc     :   

"""
****************************************************************
Criterion Module
----------------------------------------------------------------
Loss functions for neural network training.
- DiceBCELoss: Combined Dice and Binary Cross-Entropy loss
- ComprehensiveSegLoss: Comprehensive segmentation loss
- EdgeAwareLoss: Edge-aware segmentation loss
- FocalLoss: Focal loss for class imbalance
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .dice import DiceBCELoss
from .DNF import ComprehensiveSegLoss
from .edge import EdgeAwareLoss
from .focal import FocalLoss

__all__ = [
    "DiceBCELoss",
    "ComprehensiveSegLoss",
    "EdgeAwareLoss",
    "FocalLoss",
]
