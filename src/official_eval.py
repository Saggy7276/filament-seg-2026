"""Organisers' scoring code (from the competition's self-evaluation notebook), used as-is to
cross-check src/metrics.py. Plots are replaced by functions returning the same statistics.

Inputs are two DataFrames with columns `filament_id` and `segmentation_rle`:
  gt   : filament_id = "<annotator>-<image_id>_<k>"
  pred : filament_id = "<image_id>_<k>"
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util


def get_overlap_matrices(gt_layers: torch.Tensor, pred_layers: torch.Tensor):
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]
    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)
    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection
    iou_matrix = torch.where(union == 0, torch.tensor(0., device=gt_layers.device), intersection / union)
    dice_matrix = torch.where(union == 0, torch.tensor(0., device=gt_layers.device),
                              2 * intersection / (gt_areas + pred_areas))
    return iou_matrix, dice_matrix


def fp_count_hit(hit_matrix: torch.Tensor) -> int:
    return (hit_matrix.sum(dim=0) == 0).sum().item()


def fn_count_hit(hit_matrix: torch.Tensor) -> int:
    return (hit_matrix.sum(dim=1) == 0).sum().item()


def rles_to_layers(rles: list[str], height: int = 2048, width: int = 2048) -> np.ndarray:
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)


def process_entry(annotator_image, gt_annotator_image_ids, pred_image_ids, gt_df, pred_df):
    image_id = annotator_image.split("-", maxsplit=1)[1]
    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
    iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)
    return iou_matrix, dice_matrix, len(gt_rles), len(pred_rles)


def get_overlap_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]
    annotator_images = gt_annotator_image_ids.unique()
    overlap_df = pd.DataFrame({"annotator_image": annotator_images})
    results = overlap_df["annotator_image"].apply(
        process_entry, gt_annotator_image_ids=gt_annotator_image_ids, pred_image_ids=pred_image_ids,
        gt_df=gt_df, pred_df=pred_df)
    overlap_df["iou_matrix"], overlap_df["dice_matrix"], overlap_df["n_gt"], overlap_df["n_pred"] = zip(*results)
    return overlap_df


def get_pq_score(overlap_df: pd.DataFrame) -> float:
    iou_threshold = 0.5
    tp_iou_scores: list[float] = []
    fp_count = fn_count = 0
    for row in overlap_df.itertuples(index=False):
        if row.n_gt == 0:
            fp_count += row.n_pred
            continue
        if row.n_pred == 0:
            fn_count += row.n_gt
            continue
        hit_matrix = row.iou_matrix > iou_threshold
        tp_iou_scores.extend(row.iou_matrix[hit_matrix].tolist())
        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)
    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
    return sum(tp_iou_scores) / denominator if denominator > 0 else 0.0


# ---- the statistics behind the organisers' plots -------------------------------------------
def pairwise_scores(matrices) -> np.ndarray:
    """All pairwise (GT, prediction) scores > 0, as plotted by `plot_distribution`."""
    scores = np.concatenate([np.asarray(m, dtype=float).ravel() for m in matrices] or [np.zeros(0)])
    return scores[scores > 0]


def m2n_counts(overlap_df: pd.DataFrame) -> dict:
    """Degree counts behind `plot_m2n_counts` (any overlap, IoU > 0)."""
    gt_deg: list[int] = []
    pred_deg: list[int] = []
    for row in overlap_df.itertuples(index=False):
        if row.n_gt == 0:
            pred_deg.extend([0] * row.n_pred)
            continue
        if row.n_pred == 0:
            gt_deg.extend([0] * row.n_gt)
            continue
        hit = row.iou_matrix > 0
        gt_deg.extend(hit.sum(dim=1).tolist())
        pred_deg.extend(hit.sum(dim=0).tolist())
    g, p = Counter(gt_deg), Counter(pred_deg)
    return {
        "gt_degree": dict(sorted(g.items())), "pred_degree": dict(sorted(p.items())),
        "gt_misses_1:0": g.get(0, 0), "gt_hits_1:n": sum(c for d, c in g.items() if d > 0),
        "gt_one_to_many_1:n>=2": sum(c for d, c in g.items() if d >= 2),
        "pred_misses_0:1": p.get(0, 0), "pred_hits_n:1": sum(c for d, c in p.items() if d > 0),
        "pred_many_to_one_n>=2:1": sum(c for d, c in p.items() if d >= 2),
    }


def official_report(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """PQ plus the distribution statistics from the organisers' notebook."""
    ov = get_overlap_df(gt_df, pred_df)
    iou, dice = pairwise_scores(ov["iou_matrix"]), pairwise_scores(ov["dice_matrix"])
    return {
        "pq": get_pq_score(ov),
        "pairwise_iou_mean": float(iou.mean()) if iou.size else 0.0,
        "pairwise_dice_mean": float(dice.mean()) if dice.size else 0.0,
        "n_overlapping_pairs": int(iou.size),
        **m2n_counts(ov),
    }


def rles_df(by_sample: dict[str, list[dict]]) -> pd.DataFrame:
    """{sample_or_image_id: [pycocotools RLE dicts]} -> DataFrame in submission format."""
    rows = []
    for sid, rles in by_sample.items():
        for k, r in enumerate(rles, start=1):
            c = r["counts"]
            rows.append({"filament_id": f"{sid}_{k}", "segmentation_rle": c.decode("ascii") if isinstance(c, bytes) else c})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
