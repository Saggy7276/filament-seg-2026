"""Compare TTA view combinations on the validation split (tuning post-processing for each).

Runs inference once per view (cached in runs/tta_val_cache.pkl), then fuses/tunes each combination.
  python tools/eval_tta.py --weights weights/maskrcnn_r50_filament.pt
  python tools/eval_tta.py --weights ... --combos "none,hflip,vflip" "none,hflip,vflip,rot180"
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import read_gray  # noqa: E402
from src.metrics import fmt  # noqa: E402
from src.models import build_model, load_weights, set_mode  # noqa: E402
from src.predict import ValSet, infer_image, tune_postprocess  # noqa: E402
from src.tta import VIEWS, flip_image, fuse, unflip_dets  # noqa: E402
from src.utils import append_result, get_device, load_config, resolve  # noqa: E402

DEFAULT_COMBOS = ["none", "none,hflip,vflip", "none,hflip,vflip,rot180",
                  "none,rot90,rot180,rot270", ",".join(VIEWS)]

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/baseline.yaml")
ap.add_argument("--weights", required=True)
ap.add_argument("--cache", default="runs/tta_val_cache.pkl")
ap.add_argument("--combos", nargs="*", default=DEFAULT_COMBOS)
ap.add_argument("--fuse-iou", type=float, default=0.5)
args = ap.parse_args()

cfg = load_config(resolve(args.config))
vs = ValSet(cfg)
cache = resolve(args.cache)
views = pickle.loads(cache.read_bytes()) if cache.exists() else {}
needed = sorted({v for c in args.combos for v in c.split(",")} - set(views), key=VIEWS.index)
if needed:
    dev = get_device()
    model = build_model(cfg["model"], pretrained=False)
    load_weights(model, resolve(args.weights))
    model.to(dev)
    set_mode(model, cfg["model"], train=False)
    for v in needed:
        views[v] = {}
    t0 = time.time()
    for i, p in enumerate(vs.paths):
        img = read_gray(p)
        for v in needed:
            views[v][p.stem] = unflip_dets(infer_image(model, flip_image(img, v), dev, False), v, *img.shape)
        if i % 20 == 0:
            print(f"infer {needed}: {i}/{len(vs.paths)} ({time.time() - t0:.0f}s)", flush=True)
    cache.write_bytes(pickle.dumps(views))

for combo in args.combos:
    cv = combo.split(",")
    t0 = time.time()
    dets = {ts: (fuse([views[v][ts] for v in cv], args.fuse_iou) if len(cv) > 1 else views[cv[0]][ts])
            for ts in views["none"]}
    pp, _ = tune_postprocess(vs, dets, cfg["tune"], log=lambda *a: None)
    m = vs.score(dets, pp)
    print(f"RESULT {combo:60s} fuse_iou={args.fuse_iou} pp={pp} | {fmt(m)} ({time.time() - t0:.0f}s)", flush=True)
    append_result({"run": f"tta_eval:{combo}:fuse{args.fuse_iou}", "weights": args.weights,
                   **{f"val_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, "pp": pp})
