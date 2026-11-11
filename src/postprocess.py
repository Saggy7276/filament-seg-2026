"""Post-processing: turn raw Mask R-CNN detections into non-overlapping instances.

A detection is stored compactly as {score, box=(x0,y0,x1,y1), prob=uint8 crop of the
mask probability (0..255) inside the box}. Working on crops keeps memory low at 2048².
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

IMG_SIZE = 2048


def postprocess(dets: list[dict], score_thr: float, mask_thr: float, min_area: int,
                overlap_keep: float, size: int = IMG_SIZE) -> list[tuple[int, int, np.ndarray]]:
    """Greedy, score-ordered painting: each pixel belongs to at most one instance.

    An instance is dropped when its score < score_thr, its binarised area < min_area,
    or less than `overlap_keep` of it survives after removing pixels already claimed
    by higher-scoring instances (acts as mask-level NMS; reduces FPs and duplicates).
    Returns a list of (x0, y0, bool crop mask), in descending score order.
    """
    return [(x0, y0, m) for x0, y0, m, _ in postprocess_scored(dets, score_thr, mask_thr, min_area, overlap_keep, size)]


def postprocess_scored(dets, score_thr, mask_thr, min_area, overlap_keep, size: int = IMG_SIZE):
    """Same as `postprocess` but returns (x0, y0, mask, score).

    Because instances are processed in descending score order, the result for a
    higher score_thr is exactly the prefix with score >= that threshold.
    """
    occ = np.zeros((size, size), bool)
    out = []
    t = int(round(mask_thr * 255))
    for d in sorted(dets, key=lambda d: -d["score"]):
        if d["score"] < score_thr:
            break
        x0, y0, x1, y1 = d["box"]
        m = d["prob"] > t
        a = int(m.sum())
        if a < max(min_area, 1):
            continue
        region = occ[y0:y1, x0:x1]
        new = m & ~region
        na = int(new.sum())
        if na < overlap_keep * a or na < max(min_area, 1):
            continue
        region |= new  # view into occ
        out.append((x0, y0, new, d["score"]))
    return out


def instances_to_rles(insts, size: int = IMG_SIZE) -> list[dict]:
    rles = []
    for x0, y0, m, *_ in insts:
        full = np.zeros((size, size), np.uint8, order="F")
        full[y0 : y0 + m.shape[0], x0 : x0 + m.shape[1]] = m
        rles.append(mask_utils.encode(full))
    return rles
