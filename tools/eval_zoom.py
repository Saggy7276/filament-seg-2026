"""Validation study of a 2x "zoom" view for small filaments.

The detector misses most small filaments (recall 28% at 200-500 px). Here the image is cut
into overlapping tiles; the model's input transform upsamples each tile to cfg infer_size
(1024 px tile -> 2048 px = 2x zoom). Detections cut by an interior tile border are dropped
(the neighbouring, overlapping tile sees them whole), duplicates across tiles are removed
by mask NMS. The zoom detections are then combined with the cached full-image flip views.

  python tools/eval_zoom.py --weights weights/maskrcnn_r50_filament.pt
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import read_gray  # noqa: E402
from src.metrics import fmt  # noqa: E402
from src.models import build_model, load_weights, set_mode  # noqa: E402
from src.predict import ValSet, infer_image, tune_postprocess  # noqa: E402
from src.tta import _mask_iou, flip_image, fuse, unflip_dets  # noqa: E402
from src.utils import append_result, get_device, load_config, resolve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/baseline_tta.yaml")
ap.add_argument("--weights", required=True)
ap.add_argument("--base-cache", default="runs/tta_val_cache.pkl")
ap.add_argument("--cache", default="runs/tta_val_cache_zoom.pkl")
ap.add_argument("--tile", type=int, default=1024)
ap.add_argument("--stride", type=int, default=512)
ap.add_argument("--views", default="none", help="flip views applied to each tile")
ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
args = ap.parse_args()

cfg = load_config(resolve(args.config))
vs = ValSet(cfg)
paths = vs.paths[:args.limit] if args.limit else vs.paths
tviews = args.views.split(",")
key = f"zoom_t{args.tile}_s{args.stride}_{'+'.join(tviews)}"


def nms(dets: list[dict], thr: float = 0.5) -> list[dict]:
    keep = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        if all(_mask_iou(d, k) <= thr for k in keep):
            keep.append(d)
    return keep


def infer_tiles(model, img: np.ndarray, dev) -> list[dict]:
    h, w = img.shape
    t, out = args.tile, []
    for y0 in range(0, h - t + 1, args.stride):
        for x0 in range(0, w - t + 1, args.stride):
            tile = np.ascontiguousarray(img[y0:y0 + t, x0:x0 + t])
            per_view = [unflip_dets(infer_image(model, flip_image(tile, v), dev, False), v, t, t) for v in tviews]
            dets = fuse(per_view) if len(per_view) > 1 else per_view[0]
            for d in dets:
                bx0, by0, bx1, by1 = d["box"]
                m = d["prob"] > 127
                if not m.any():
                    continue
                ys, xs = np.nonzero(m)
                mx0, my0, mx1, my1 = bx0 + xs.min(), by0 + ys.min(), bx0 + xs.max() + 1, by0 + ys.max() + 1
                cut = ((mx0 <= 1 and x0 > 0) or (my0 <= 1 and y0 > 0) or
                       (mx1 >= t - 1 and x0 + t < w) or (my1 >= t - 1 and y0 + t < h))
                if cut:
                    continue
                out.append({"score": d["score"], "box": (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0), "prob": d["prob"]})
    return nms(out)


cache = resolve(args.cache)
store = pickle.loads(cache.read_bytes()) if cache.exists() else {}
if key not in store or any(p.stem not in store[key] for p in paths):
    dev = get_device()
    model = build_model(cfg["model"], pretrained=False)
    load_weights(model, resolve(args.weights))
    model.to(dev)
    set_mode(model, cfg["model"], train=False)
    model.roi_heads.score_thresh = 0.3
    store.setdefault(key, {})
    t0 = time.time()
    for i, p in enumerate(paths):
        if p.stem not in store[key]:
            store[key][p.stem] = infer_tiles(model, read_gray(p), dev)
        if i % 10 == 0:
            print(f"zoom infer {i}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    if not args.limit:
        cache.write_bytes(pickle.dumps(store))
    del model

zoom = store[key]
base_views = pickle.loads(resolve(args.base_cache).read_bytes())
bv = ["none", "hflip", "vflip"]
ids = [p.stem for p in paths]
if args.limit:
    keep = set(ids)
    vs.samples = [s for s in vs.samples if s["timestamp"] in keep]
base = {ts: fuse([base_views[v][ts] for v in bv]) for ts in ids}


def union(b: list[dict], z: list[dict], alpha: float, boost: bool, max_side: int) -> list[dict]:
    """Base detections + zoom detections that match no base detection (score * alpha).
    boost: a base detection confirmed by a zoom detection takes max(score_base, score_zoom)."""
    out = [dict(d) for d in b]
    for d in z:
        x0, y0, x1, y1 = d["box"]
        j = next((k for k, o in enumerate(b) if _mask_iou(d, o) > 0.5), -1)
        if j >= 0:
            if boost:
                out[j]["score"] = max(out[j]["score"], d["score"])
        elif max(x1 - x0, y1 - y0) <= max_side:
            out.append({"score": d["score"] * alpha, "box": d["box"], "prob": d["prob"]})
    return out


def report(name: str, dets: dict):
    t0 = time.time()
    pp, _ = tune_postprocess(vs, dets, cfg["tune"], log=lambda *a: None)
    m = vs.score(dets, pp)
    print(f"RESULT {name:34s} pp={pp} | {fmt(m)} ({time.time() - t0:.0f}s)", flush=True)
    if not args.limit:
        append_result({"run": f"zoom_eval:{key}:{name}", "weights": args.weights,
                       **{f"val_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, "pp": pp})


print(f"zoom dets/img {np.mean([len(zoom[t]) for t in ids]):.1f} | base dets/img {np.mean([len(base[t]) for t in ids]):.1f}")
report("base 3-view", base)
report("zoom alone", {ts: zoom[ts] for ts in ids})
report("fuse 4 views", {ts: fuse([base_views[v][ts] for v in bv] + [zoom[ts]]) for ts in ids})
for alpha, boost, max_side in [(1.0, False, 9999), (0.9, False, 9999), (1.0, True, 9999), (0.9, True, 9999),
                               (1.0, False, 128), (1.0, True, 128), (1.0, True, 64)]:
    report(f"union a{alpha} boost{int(boost)} side{max_side}",
           {ts: union(base[ts], zoom[ts], alpha, boost, max_side) for ts in ids})
