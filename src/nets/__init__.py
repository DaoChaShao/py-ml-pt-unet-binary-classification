#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:17
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Neural Nets Module
----------------------------------------------------------------
UNet-based neural network architectures.
- Standard4LayersUNet: 4-layer UNet variant
- Standard5LayersUNet: 5-layer UNet variant
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .standard4 import Standard4LayersUNet
from .standard5 import Standard5LayersUNet

__all__ = [
    "Standard4LayersUNet",
    "Standard5LayersUNet",
]
