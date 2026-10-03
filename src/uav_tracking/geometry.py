"""Pairwise box overlap used by causal recovery."""
from __future__ import annotations
import numpy as np


def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(-1, 4)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 4)
    if not len(left) or not len(right):
        return np.empty((len(left), len(right)), dtype=np.float64)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    size = np.maximum(0.0, bottom_right - top_left)
    intersection = size[..., 0] * size[..., 1]
    left_area = np.maximum(0.0, left[:, 2] - left[:, 0]) * np.maximum(
        0.0, left[:, 3] - left[:, 1]
    )
    right_area = np.maximum(0.0, right[:, 2] - right[:, 0]) * np.maximum(
        0.0, right[:, 3] - right[:, 1]
    )
    union = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(
        intersection, union,
        out=np.zeros_like(intersection), where=union > 0,
    )
