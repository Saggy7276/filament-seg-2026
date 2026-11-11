"""Evaluate model ensembles on validation from per-model TTA caches (tools/eval_tta.py).

  python tools/eval_tta.py --weights weights/maskrcnn_r50_filament.pt --cache runs/tta_val_cache.pkl --combos none,hflip,vflip
  python tools/eval_tta.py --weights runs/<seed7>/best.pt --cache runs/tta_val_cache_seed7.pkl --combos none,hflip,vflip
  python tools/eval_ensemble.py --caches runs/tta_val_cache.pkl runs/tta_val_cache_seed7.pkl
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import fmt  # noqa: E402
from src.predict import ValSet, tune_postprocess  # noqa: E402
from src.tta import fuse  # noqa: E402
from src.utils import append_result, load_config, resolve  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/baseline.yaml")
ap.add_argument("--caches", nargs="+", required=True)
ap.add_argument("--views", default="none,hflip,vflip")
ap.add_argument("--fuse-iou", type=float, nargs="*", default=[0.5])
args = ap.parse_args()

cfg = load_config(resolve(args.config))
vs = ValSet(cfg)
caches = [pickle.loads(resolve(c).read_bytes()) for c in args.caches]
views = args.views.split(",")
names = [Path(c).stem for c in args.caches]

for fi in args.fuse_iou:
    for subset in ([0], list(range(len(caches)))) if len(caches) > 1 else ([0],):
        t0 = time.time()
        dets = {ts: fuse([caches[i][v][ts] for i in subset for v in views], fi) for ts in caches[0]["none"]}
        pp, _ = tune_postprocess(vs, dets, cfg["tune"], log=lambda *a: None)
        m = vs.score(dets, pp)
        tag = "+".join(names[i] for i in subset)
        print(f"RESULT {tag} views={args.views} fuse_iou={fi} pp={pp} | {fmt(m)} ({time.time() - t0:.0f}s)", flush=True)
        append_result({"run": f"ensemble:{tag}:{args.views}:fuse{fi}",
                       **{f"val_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, "pp": pp})
