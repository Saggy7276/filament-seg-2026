"""Validation study of the stage-2 crop refiner on cached detector outputs (tools/eval_tta.py cache).

Variants
  paint : detector instances after score-ordered painting -> refine each -> resolve overlaps
          again by painting (pixels go to the higher-scoring instance).
  merge : same, but refined masks that touch are merged into one instance.
  group : detector masks that overlap are merged first, each group is refined once,
          refined masks that touch are merged (the order used by the public notebook).

  python tools/eval_refiner.py --refiner runs/<refiner run>/best.pt
"""
import argparse
import itertools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import read_gray  # noqa: E402
from src.metrics import aggregate, fmt, image_metrics  # noqa: E402
from src.postprocess import instances_to_rles, merge_touching, postprocess_scored  # noqa: E402
from src.predict import ValSet  # noqa: E402
from src.refiner import load_refiner, refine  # noqa: E402
from src.tta import fuse  # noqa: E402
from src.utils import append_result, get_device, load_config, resolve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/baseline_tta.yaml")
ap.add_argument("--cache", nargs="+", default=["runs/tta_val_cache.pkl"], help="one cache per detector (several = ensemble)")
ap.add_argument("--views", default="none,hflip,vflip")
ap.add_argument("--refiner", required=True)
ap.add_argument("--refiner-tta", default="none,hflip,vflip,hvflip")
ap.add_argument("--score-thr", type=float, nargs="*", default=[0.6, 0.7, 0.75, 0.8, 0.85, 0.9])
ap.add_argument("--refine-thr", type=float, nargs="*", default=[0.3, 0.4, 0.5, 0.6, 0.7])
ap.add_argument("--min-area", type=int, nargs="*", default=[0, 50])
ap.add_argument("--variants", nargs="*", default=["paint", "merge", "group"])
ap.add_argument("--out", default="results/refiner_eval.csv")
args = ap.parse_args()

cfg = load_config(resolve(args.config))
vs = ValSet(cfg)
caches = [pickle.loads(resolve(c).read_bytes()) for c in args.cache]
cv = args.views.split(",")
dets = {ts: (fuse([c[v][ts] for c in caches for v in cv]) if len(cv) * len(caches) > 1 else caches[0][cv[0]][ts])
        for ts in caches[0]["none"]}
device = get_device()
model, rcfg = load_refiner(args.refiner, device)
rtta = args.refiner_tta.split(",")
imgs = {p.stem: p for p in vs.paths}
S_MIN = min(args.score_thr)


def score(preds: dict) -> dict:
    return aggregate([image_metrics(preds[s["timestamp"]], vs.gt[s["sample_id"]]) for s in vs.samples])


def paint(insts: list, min_area: int, keep: float = 0.5) -> list:
    """insts: (x0, y0, bool mask), already in descending score order."""
    occ = np.zeros((2048, 2048), bool)
    out = []
    for x0, y0, m in insts:
        a = int(m.sum())
        reg = occ[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        new = m & ~reg
        na = int(new.sum())
        if a == 0 or na < keep * a or na < max(min_area, 1):
            continue
        reg |= new
        out.append((x0, y0, new))
    return out


def binarise(inst, prob, thr: float):
    """Refined mask at `thr`; falls back to the detector mask when the refiner returns nothing."""
    m = prob[2] > int(round(thr * 255))
    return (prob[0], prob[1], m) if m.any() else (inst[0], inst[1], inst[2])


rows = []


def record(variant, st, rt, ma, preds):
    m = score(preds)
    rows.append({"variant": variant, "score_thr": st, "refine_thr": rt, "min_area": ma, **m})


# ---- baseline (current submission post-processing)
base = {ts: instances_to_rles(postprocess_scored(d, 0.8, 0.5, 0, 0.5)) for ts, d in dets.items()}
print("BASELINE (no refiner, score 0.8):", fmt(score(base)), flush=True)

# ---- paint / merge: refine every painted detector instance once
t0 = time.time()
if {"paint", "merge"} & set(args.variants):
    ref = {}
    for ts, d in dets.items():
        insts = postprocess_scored(d, S_MIN, 0.5, 0, 0.5)
        probs = refine(model, rcfg, read_gray(imgs[ts]), insts, device, rtta)
        ref[ts] = (insts, [(p[0], p[1], np.round(p[2] * 255).astype(np.uint8)) for p in probs])
    print(f"refined painted instances in {time.time() - t0:.0f}s", flush=True)
    for st, rt in itertools.product(args.score_thr, args.refine_thr):
        sel = {ts: [binarise(i, p, rt) for i, p in zip(*ref[ts]) if i[3] >= st] for ts in ref}
        for ma in args.min_area:
            if "paint" in args.variants:
                record("paint", st, rt, ma, {ts: instances_to_rles(paint(v, ma)) for ts, v in sel.items()})
            if "merge" in args.variants:
                record("merge", st, rt, ma, {ts: merge_touching(instances_to_rles([i for i in v if i[2].sum() >= max(ma, 1)]))
                                             for ts, v in sel.items()})
        print(f"score_thr {st} refine_thr {rt} done ({time.time() - t0:.0f}s)", flush=True)

# ---- group: merge overlapping detector masks first, refine each group
if "group" in args.variants:
    for st in args.score_thr:
        ref = {}
        for ts, d in dets.items():
            raw = [(x["box"][0], x["box"][1], x["prob"] > 127) for x in d if x["score"] >= st]
            raw = [r for r in raw if r[2].any()]
            groups = merge_touching(instances_to_rles(raw))
            insts = []
            for g in groups:
                x, y, w, h = (int(v) for v in mask_utils.toBbox(g))
                insts.append((x, y, mask_utils.decode(g)[y:y + h, x:x + w].astype(bool)))
            probs = refine(model, rcfg, read_gray(imgs[ts]), insts, device, rtta)
            ref[ts] = (insts, [(p[0], p[1], np.round(p[2] * 255).astype(np.uint8)) for p in probs])
        for rt, ma in itertools.product(args.refine_thr, args.min_area):
            record("group", st, rt, ma, {
                ts: merge_touching(instances_to_rles([b for b in (binarise(i, p, rt) for i, p in zip(*ref[ts]))
                                                      if b[2].sum() >= max(ma, 1)]))
                for ts in ref})
        print(f"group score_thr {st} done ({time.time() - t0:.0f}s)", flush=True)

df = pd.DataFrame(rows).sort_values("pq", ascending=False)
out = resolve(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
for variant in args.variants:
    b = df[df.variant == variant].iloc[0]
    print(f"RESULT {variant}: score_thr {b.score_thr} refine_thr {b.refine_thr} min_area {int(b.min_area)} | "
          f"{fmt(b.to_dict())}", flush=True)
    append_result({"run": f"refiner_eval:{variant}:{'+'.join(Path(c).stem for c in args.cache)}", "weights": args.refiner,
                   **{f"val_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in b.to_dict().items()
                      if k not in ("variant",)}})
print(df.head(15).to_string())
