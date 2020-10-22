"""Defines colors to be assigned to classes for visualization in segmentation
tasks.
"""
import numpy as np


class SegmentationColors:
    colors = {
        0: np.array([255, 0, 0]),
        1: np.array([0, 255, 0]),
        2: np.array([0, 0, 255]),
        3: np.array([128, 128, 0]),
        4: np.array([0, 255, 255]),
        5: np.array([255, 0, 255]),
        6: np.array([255, 255, 0]),
        7: np.array([0, 255, 128])
    }
