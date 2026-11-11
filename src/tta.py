"""Test-time augmentation (flips / 90-degree rotations) with instance-level fusion.

A view is a horizontal flip (optional) followed by k counter-clockwise 90-degree rotations
(np.rot90). The 8 views form the dihedral group of the square image. Each view's detections
are mapped back to original coordinates, then matched across views by mask IoU. A fused
instance has:
  * mask probability = mean over the views that detected it (on the union box),
  * score            = sum of its scores / number of views (a view that missed it
                       counts as 0), which down-weights detections seen in only one view.
"""
from __future__ import annotations

import numpy as np

# name -> (hflip first, number of CCW 90-degree rotations)
VIEW_DEFS = {
    "none": (False, 0), "hflip": (True, 0), "vflip": (True, 2), "rot180": (False, 2),
    "rot90": (False, 1), "rot270": (False, 3), "hflip_rot90": (True, 1), "hflip_rot270": (True, 3),
}
VIEWS = tuple(VIEW_DEFS)


def flip_image(img: np.ndarray, view: str) -> np.ndarray:
    hf, k = VIEW_DEFS[view]
    if hf:
        img = img[:, ::-1]
    return np.ascontiguousarray(np.rot90(img, k))


def _unrot1(box, prob, h_rot: int, w_rot: int):
    """Undo one CCW np.rot90 for a detection in a (h_rot, w_rot) rotated image.

    np.rot90(A)[i, j] = A[j, n-1-i] with A of shape (m, n) = (w_rot, h_rot).
    """
    x0, y0, x1, y1 = box
    n = h_rot
    return (n - y1, x0, n - y0, x1), np.rot90(prob, -1)


def unflip_dets(dets: list[dict], view: str, h: int, w: int) -> list[dict]:
    """Map detections from view coordinates back to the original (h, w) image."""
    hf, k = VIEW_DEFS[view]
    out = []
    for d in dets:
        box, p = d["box"], d["prob"]
        # rotated image shape after k CCW rotations of (h, w)
        hh, ww = (h, w) if k % 2 == 0 else (w, h)
        for _ in range(k):
            box, p = _unrot1(box, p, hh, ww)
            hh, ww = ww, hh
        if hf:
            x0, y0, x1, y1 = box
            box, p = (w - x1, y0, w - x0, y1), p[:, ::-1]
        out.append({"score": d["score"], "box": tuple(int(v) for v in box), "prob": np.ascontiguousarray(p)})
    return out


def _mask_iou(a: dict, b: dict, t: int = 127) -> float:
    ax0, ay0, ax1, ay1 = a["box"]
    bx0, by0, bx1, by1 = b["box"]
    ix0, iy0, ix1, iy1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    ma, mb = a["prob"] > t, b["prob"] > t
    sa = ma[iy0 - ay0:iy1 - ay0, ix0 - ax0:ix1 - ax0]
    sb = mb[iy0 - by0:iy1 - by0, ix0 - bx0:ix1 - bx0]
    inter = int((sa & sb).sum())
    union = int(ma.sum()) + int(mb.sum()) - inter
    return inter / union if union else 0.0


def fuse(views: list[list[dict]], iou_thr: float = 0.5) -> list[dict]:
    """Fuse per-view detections (already in original coordinates)."""
    n = len(views)
    clusters: list[list[dict]] = []  # each cluster: list of member dets (<= 1 per view)
    reps: list[dict] = []            # representative det per cluster (first / highest-score member)
    for vi, dets in enumerate(views):
        dets = sorted(dets, key=lambda d: -d["score"])
        if vi == 0:
            clusters = [[d] for d in dets]
            reps = list(dets)
            continue
        taken = set()
        new_clusters = []
        for d in dets:
            best, best_iou = -1, iou_thr
            for ci, r in enumerate(reps):
                if ci in taken:
                    continue
                iou = _mask_iou(d, r)
                if iou > best_iou:
                    best, best_iou = ci, iou
            if best >= 0:
                clusters[best].append(d)
                taken.add(best)
            else:
                new_clusters.append([d])
        for c in new_clusters:
            clusters.append(c)
            reps.append(c[0])

    fused = []
    for c in clusters:
        x0 = min(d["box"][0] for d in c); y0 = min(d["box"][1] for d in c)
        x1 = max(d["box"][2] for d in c); y1 = max(d["box"][3] for d in c)
        acc = np.zeros((y1 - y0, x1 - x0), np.float32)
        for d in c:
            bx0, by0, bx1, by1 = d["box"]
            acc[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] += d["prob"]
        prob = np.clip(np.round(acc / len(c)), 0, 255).astype(np.uint8)
        votes = [d["score"] for d in c]
        fused.append({"score": sum(votes) / n, "box": (x0, y0, x1, y1), "prob": prob,
                      "votes": votes, "n_views": n})
    return fused
