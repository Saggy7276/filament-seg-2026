"""Qualitative visualisation: instance masks overlaid on H-alpha images."""
from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils


def _colors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hsv = np.stack([rng.permutation(np.linspace(0, 179, max(n, 1), endpoint=False)),
                    np.full(max(n, 1), 220), np.full(max(n, 1), 255)], 1).astype(np.uint8)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB)[0]


def overlay(img: np.ndarray, rles: list[dict], alpha: float = 0.45, outline: bool = True) -> np.ndarray:
    """Grayscale uint8 image + instance RLEs -> RGB uint8 with one colour per instance."""
    rgb = np.repeat(img[..., None], 3, 2).astype(np.float32)
    cols = _colors(len(rles))
    for c, r in zip(cols, rles):
        m = mask_utils.decode(r).astype(bool)
        rgb[m] = (1 - alpha) * rgb[m] + alpha * c
        if outline:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            tmp = rgb.astype(np.uint8).copy()
            cv2.drawContours(tmp, cnts, -1, tuple(int(x) for x in c), 2)
            rgb = tmp.astype(np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def show_pair(img, gt_rles, pred_rles, title: str = "", crop=None, figsize=(14, 7)):
    """Side-by-side ground truth vs prediction. `crop` = (x0, y0, size) to zoom in."""
    a, b = overlay(img, gt_rles), overlay(img, pred_rles)
    if crop is not None:
        x0, y0, s = crop
        a, b = a[y0:y0 + s, x0:x0 + s], b[y0:y0 + s, x0:x0 + s]
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(a); ax[0].set_title(f"Ground truth ({len(gt_rles)})")
    ax[1].imshow(b); ax[1].set_title(f"Prediction ({len(pred_rles)})")
    for x in ax:
        x.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
