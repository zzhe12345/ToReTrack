"""Persistent cross-view identity bookkeeping."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[b] = a


def track_token(key):
    key = tuple(key)
    return str(key[0]), int(key[1]), int(key[3])


def persistent_cross_view_matches(records, threshold):
    """Collapse repeated window evidence before linking persistent tracks.

    A short-window false positive must not irreversibly union two complete tracks.
    The MDMT decision is therefore made once per persistent view-1/view-2 track
    pair from the median of all its window scores, followed by a scene-wise
    one-to-one Hungarian assignment.
    """
    evidence = defaultdict(list)
    left_by_scene = defaultdict(set)
    right_by_scene = defaultdict(set)
    for record in records:
        for row, col, score in record["pairs"]:
            left = track_token(record["left_keys"][row])
            right = track_token(record["right_keys"][col])
            if left[0] != right[0]:
                continue
            evidence[(left, right)].append(float(score))
            left_by_scene[left[0]].add(left)
            right_by_scene[right[0]].add(right)
    selected = []
    for scene in sorted(set(left_by_scene) | set(right_by_scene), key=int):
        left = sorted(left_by_scene[scene])
        right = sorted(right_by_scene[scene])
        if not left or not right:
            continue
        scores = np.full((len(left), len(right)), -1.0, dtype=np.float32)
        for i, lkey in enumerate(left):
            for j, rkey in enumerate(right):
                values = evidence.get((lkey, rkey))
                if values:
                    scores[i, j] = float(np.median(values))
        rows, cols = linear_sum_assignment(-scores)
        for row, col in zip(rows, cols):
            if scores[row, col] >= threshold:
                selected.append((left[row], right[col], float(scores[row, col])))
    return selected
