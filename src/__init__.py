#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:13
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :   

"""
****************************************************************
Machine Learning Package
----------------------------------------------------------------
A comprehensive ML package containing:
- criteria: Loss functions and evaluation metrics
- datasets: Data loading and processing
- nets: Neural network models
- trainers: Training loops and strategies
- utils: Utility functions and tools
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from . import criteria
from . import datasets
from . import nets
from . import trainers
from . import utils

__all__ = [
    "criteria",
    "datasets",
    "nets",
    "trainers",
    "utils",
]
