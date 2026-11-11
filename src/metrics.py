"""Evaluation metrics: Panoptic Quality, Dice, IoU distributions, split/merge counts.

All masks are COCO RLEs (pycocotools), which keeps IoU computation fast at 2048x2048.

PQ = sum(IoU of TP) / (|TP| + 0.5|FP| + 0.5|FN|), a pair is a TP when IoU > 0.5.
Dice is the standard binary Dice 2|A∩B|/(|A|+|B|) (identical to
torchmetrics.segmentation.DiceScore for one binary class).

Split/merge definitions (our approximation of the rubric's "one-to-many" and
"many-to-one" counts):
  one-to-many : a GT that has >= 2 predictions each lying > 50% inside it
  many-to-one : a prediction that has >= 2 GTs each lying > 50% inside it
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

IOU_THR = 0.5


def _area(rles):
    return np.asarray(mask_utils.area(rles), np.float64) if rles else np.zeros(0)


def image_metrics(pred: list[dict], gt: list[dict], thr: float = IOU_THR) -> dict:
    P, G = len(pred), len(gt)
    pa, ga = _area(pred), _area(gt)
    if P and G:
        iou = np.asarray(mask_utils.iou(pred, gt, [0] * G), np.float64).reshape(P, G)
    else:
        iou = np.zeros((P, G))
    inter = iou * (pa[:, None] + ga[None, :]) / (1.0 + iou)

    # one-to-one matching, greedy by IoU (unique anyway when IoU > 0.5 and masks don't overlap)
    matched_iou = []
    used_p, used_g = set(), set()
    cand = np.argwhere(iou > thr)
    for p, g in sorted(cand.tolist(), key=lambda pg: -iou[pg[0], pg[1]]):
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
            matched_iou.append(iou[p, g])
    tp = len(matched_iou)

    one_to_many = int(((inter / np.maximum(pa[:, None], 1)) > 0.5).sum(0).__ge__(2).sum()) if P and G else 0
    many_to_one = int(((inter / np.maximum(ga[None, :], 1)) > 0.5).sum(1).__ge__(2).sum()) if P and G else 0

    # semantic (union) Dice for the image
    if P and G:
        pu, gu = mask_utils.merge(pred), mask_utils.merge(gt)
        i = float(mask_utils.area(mask_utils.merge([pu, gu], intersect=True)))
        dice_sem = 2 * i / (float(mask_utils.area(pu)) + float(mask_utils.area(gu)))
    else:
        dice_sem = 1.0 if P == G == 0 else 0.0

    # best-IoU per GT (distribution, includes unmatched / poorly matched GTs)
    best_iou_per_gt = iou.max(0).tolist() if P else [0.0] * G

    return {
        "tp": tp,
        "fp": P - tp,
        "fn": G - tp,
        "sum_iou": float(sum(matched_iou)),
        "matched_iou": [float(x) for x in matched_iou],
        "best_iou_per_gt": best_iou_per_gt,
        "dice_sem": dice_sem,
        "one_to_many": one_to_many,
        "many_to_one": many_to_one,
        "n_pred": P,
        "n_gt": G,
    }


def torchmetrics_dice(pred: list[dict], gt: list[dict]) -> float:
    """Semantic (foreground) Dice via torchmetrics.segmentation.DiceScore, as named in the
    competition rubric. Numerically equal to `dice_sem` from image_metrics."""
    import torch
    from torchmetrics.segmentation import DiceScore

    def union(rles):
        if not rles:
            return np.zeros(gt[0]["size"] if gt else pred[0]["size"], np.uint8)
        return mask_utils.decode(mask_utils.merge(rles))

    if not pred and not gt:
        return 1.0
    p = torch.from_numpy(union(pred).astype(np.int64))[None]
    t = torch.from_numpy(union(gt).astype(np.int64))[None]
    d = DiceScore(num_classes=2, include_background=False, input_format="index")
    d.update(p, t)
    return float(d.compute())


def pq_of(m: dict) -> float:
    den = m["tp"] + 0.5 * m["fp"] + 0.5 * m["fn"]
    return m["sum_iou"] / den if den > 0 else 1.0


def aggregate(per_image: list[dict]) -> dict:
    """Dataset-level metrics. `pq` pools TP/FP/FN over all images; `pq_img` averages per image."""
    tot = {k: sum(m[k] for m in per_image) for k in ("tp", "fp", "fn", "sum_iou", "one_to_many", "many_to_one", "n_pred", "n_gt")}
    tp, fp, fn = tot["tp"], tot["fp"], tot["fn"]
    matched = [x for m in per_image for x in m["matched_iou"]]
    best = np.array([x for m in per_image for x in m["best_iou_per_gt"]])
    dice_best = 2 * best / (1 + best) if len(best) else best
    return {
        "pq": pq_of(tot),
        "pq_img": float(np.mean([pq_of(m) for m in per_image])) if per_image else 0.0,
        "sq": tot["sum_iou"] / tp if tp else 0.0,
        "rq": tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice_sem": float(np.mean([m["dice_sem"] for m in per_image])) if per_image else 0.0,
        "dice_per_gt": float(dice_best.mean()) if len(best) else 0.0,
        "iou_matched_mean": float(np.mean(matched)) if matched else 0.0,
        "one_to_many": tot["one_to_many"],
        "many_to_one": tot["many_to_one"],
        "n_pred": tot["n_pred"],
        "n_gt": tot["n_gt"],
    }


def fmt(m: dict) -> str:
    return (f"PQ {m['pq']:.4f} (img {m['pq_img']:.4f}) SQ {m['sq']:.3f} RQ {m['rq']:.3f} | "
            f"TP {m['tp']} FP {m['fp']} FN {m['fn']} | Dice sem {m['dice_sem']:.3f} perGT {m['dice_per_gt']:.3f} | "
            f"1:N {m['one_to_many']} N:1 {m['many_to_one']}")
