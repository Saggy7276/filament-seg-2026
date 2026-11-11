"""Sanity tests: metrics, postprocess, RLE round-trip, and the GT-vs-GT upper bound.

  python -m pytest tests -q      (or: python tests/test_pipeline.py)
"""
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import aggregate, image_metrics  # noqa: E402
from src.postprocess import instances_to_rles, postprocess  # noqa: E402
from src.predict import rle_counts  # noqa: E402

S = 2048


def rect(x0, y0, x1, y1):
    m = np.zeros((S, S), np.uint8, order="F")
    m[y0:y1, x0:x1] = 1
    return mask_utils.encode(m)


def test_perfect_match():
    gt = [rect(0, 0, 10, 10), rect(100, 100, 150, 120)]
    m = aggregate([image_metrics(gt, gt)])
    assert m["tp"] == 2 and m["fp"] == m["fn"] == 0 and abs(m["pq"] - 1) < 1e-9


def test_fragmentation_penalised():
    gt = [rect(0, 0, 100, 10)]
    pred = [rect(0, 0, 50, 10), rect(50, 0, 100, 10)]  # one GT split in two halves (IoU 0.5 each, not > 0.5)
    m = aggregate([image_metrics(pred, gt)])
    assert m["tp"] == 0 and m["fp"] == 2 and m["fn"] == 1 and m["pq"] == 0
    assert m["one_to_many"] == 1


def test_merge_counted():
    gt = [rect(0, 0, 40, 10), rect(60, 0, 100, 10)]
    pred = [rect(0, 0, 100, 10)]
    m = aggregate([image_metrics(pred, gt)])
    assert m["many_to_one"] == 1 and m["tp"] == 0


def test_postprocess_removes_duplicates_and_roundtrip():
    prob = np.full((20, 20), 255, np.uint8)
    dets = [{"score": 0.9, "box": (10, 10, 30, 30), "prob": prob},
            {"score": 0.8, "box": (12, 12, 32, 32), "prob": prob}]  # mostly overlapping duplicate
    insts = postprocess(dets, score_thr=0.5, mask_thr=0.5, min_area=0, overlap_keep=0.5)
    assert len(insts) == 1
    rle = instances_to_rles(insts)[0]
    dec = mask_utils.decode({"size": [S, S], "counts": rle_counts(rle).encode()})
    assert dec.shape == (S, S) and dec.sum() == 400


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
