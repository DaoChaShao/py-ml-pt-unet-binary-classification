#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 00:13
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   __init__.py
# @Desc     :

"""
****************************************************************
Utility Module
----------------------------------------------------------------
This module provides various utility functions and classes, including:
- Configuration management
- Performance measurement
- Text highlighting
- PyTorch tensor operations
- Statistical data processing
****************************************************************
"""

__author__ = "Shawn Yu"
__version__ = "0.1.0"

from .config import CONFIG
from .decorator import timer, beautifier
from .helper import Timer, Beautifier, RandomSeed, read_file
from .highlighter import (black, red, green, yellow, blue, purple, cyan, white,
                          bold, underline, invert, strikethrough,
                          starts, lines, sharps)
from .PT import (TorchRandomSeed, TorchDataLoader, GrayTensorReshaper,
                 check_device, get_device, arr2tensor, df2tensor)
from .stats import (NumpyRandomSeed,
                    load_csv, load_text, summary_dataframe,
                    load_paths, split_paths, save_json, load_json,
                    standardise_data,
                    split_array,
                    select_pca_importance)

__all__ = [
    "CONFIG",

    "timer", "beautifier",

    "Timer", "Beautifier", "RandomSeed", "read_file",

    "black", "red", "green", "yellow", "blue", "purple", "cyan", "white",
    "bold", "underline", "invert", "strikethrough",
    "starts", "lines", "sharps",

    "TorchRandomSeed", "TorchDataLoader", "GrayTensorReshaper",
    "check_device", "get_device", "arr2tensor", "df2tensor",

    "NumpyRandomSeed",
    "load_csv", "load_text", "summary_dataframe",
    "load_paths", "split_paths", "save_json", "load_json",
    "standardise_data",
    "split_array",
    "select_pca_importance",
]
