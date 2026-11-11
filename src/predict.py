"""Inference, validation scoring, post-processing tuning and submission CSV writing.

CLI examples
  # tune post-processing on the validation split, then write the test submission
  python -m src.predict --config configs/baseline.yaml --weights weights/best.pt --tune --submit
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

from .data import get_split, gt_rles, load_samples, read_gray, to_model_input
from .metrics import aggregate, fmt, image_metrics
from .models import build_model, load_weights, set_mode
from .postprocess import IMG_SIZE, instances_to_rles, postprocess, postprocess_scored
from .tta import flip_image, fuse, unflip_dets
from .utils import Logger, get_device, load_config, resolve, seed_everything

PP_KEYS = ("score_thr", "mask_thr", "min_area", "overlap_keep")


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def infer_image(model, img: np.ndarray, device, amp: bool = True) -> list[dict]:
    """Run the model on one full-resolution image; return compact detections."""
    x = to_model_input(img).to(device)
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        out = model([x])[0]
    boxes = out["boxes"].float().cpu().numpy()
    scores = out["scores"].float().cpu().numpy()
    masks = out["masks"][:, 0]
    h, w = img.shape
    dets = []
    for i in range(len(scores)):
        bx0, by0, bx1, by1 = boxes[i]
        pw, ph = 0.06 * (bx1 - bx0) + 2, 0.06 * (by1 - by0) + 2  # pasted masks may exceed box slightly
        x0, y0 = max(int(bx0 - pw), 0), max(int(by0 - ph), 0)
        x1, y1 = min(int(np.ceil(bx1 + pw)), w), min(int(np.ceil(by1 + ph)), h)
        if x1 <= x0 or y1 <= y0:
            continue
        prob = (masks[i, y0:y1, x0:x1].float() * 255).round().to(torch.uint8).cpu().numpy()
        dets.append({"score": float(scores[i]), "box": (x0, y0, x1, y1), "prob": prob})
    del out, masks
    return dets


def infer_image_tta(models, img: np.ndarray, device, amp: bool, views) -> list[dict]:
    """TTA and/or model ensemble: every (model, view) pair is one vote, fused in src/tta.py.

    `models` may be a single model or a list (ensemble); `views` a list from src.tta.VIEWS.
    """
    models = models if isinstance(models, (list, tuple)) else [models]
    h, w = img.shape
    per_view = [unflip_dets(infer_image(m, flip_image(img, v), device, amp), v, h, w)
                for m in models for v in views]
    return fuse(per_view)


def run_inference(models, image_paths: list[Path], mcfg: dict, device, amp: bool = True,
                  desc: str = "infer", tta=None) -> dict[str, list[dict]]:
    """models: one model or a list (ensemble). tta: None/[] for single view, or a list of views
    from src.tta.VIEWS (default: mcfg['tta'])."""
    models = models if isinstance(models, (list, tuple)) else [models]
    for m in models:
        set_mode(m, mcfg, train=False)
    tta = mcfg.get("tta") if tta is None else tta
    views = list(tta) if tta else ["none"]
    res = {}
    for p in tqdm(image_paths, desc=desc, leave=False):
        img = read_gray(p)
        res[Path(p).stem] = (infer_image_tta(models, img, device, amp, views) if len(models) * len(views) > 1
                             else infer_image(models[0], img, device, amp))
    return res


# --------------------------------------------------------------------------- validation
class ValSet:
    """Validation samples grouped by timestamp, with cached GT RLEs."""

    def __init__(self, cfg):
        root = resolve(cfg["data"]["root"])
        samples = load_samples(root / cfg["data"]["train_json"])
        split = get_split(samples, resolve(cfg["data"]["split_file"]), cfg["data"]["val_frac"], cfg["seed"])
        val_ts = set(split["val"])
        self.samples = [s for s in samples if s["timestamp"] in val_ts]
        self.gt = {s["sample_id"]: gt_rles(s) for s in self.samples}
        img_dir = root / cfg["data"]["train_images"]
        self.paths = sorted({img_dir / s["file_name"] for s in self.samples})

    def predictions(self, dets_by_ts: dict, pp: dict) -> dict[str, list[dict]]:
        """Post-processed prediction RLEs per timestamp."""
        return {ts: instances_to_rles(postprocess(d, **{k: pp[k] for k in PP_KEYS}))
                for ts, d in dets_by_ts.items()}

    def per_image(self, dets_by_ts: dict, pp: dict) -> list[dict]:
        """Metrics per validation sample (one per annotator of each image)."""
        preds = self.predictions(dets_by_ts, pp)
        return [{"sample_id": s["sample_id"], **image_metrics(preds[s["timestamp"]], self.gt[s["sample_id"]])}
                for s in self.samples]

    def score(self, dets_by_ts: dict, pp: dict) -> dict:
        return aggregate(self.per_image(dets_by_ts, pp))


def tune_postprocess(valset: ValSet, dets_by_ts: dict, grid: dict, log=print) -> tuple[dict, pd.DataFrame]:
    keys = list(PP_KEYS)
    rows = []
    s_thrs = sorted(grid["score_thr"])
    combos = list(itertools.product(grid["mask_thr"], grid["min_area"], grid["overlap_keep"]))
    for mask_thr, min_area, ok in tqdm(combos, desc="tune", leave=False):
        # postprocess + RLE-encode once at the lowest score threshold, then filter by score
        enc = {}
        for ts, d in dets_by_ts.items():
            insts = postprocess_scored(d, s_thrs[0], mask_thr, min_area, ok)
            enc[ts] = (np.array([i[3] for i in insts]), instances_to_rles(insts))
        for st in s_thrs:
            per = []
            for s in valset.samples:
                sc, rles = enc[s["timestamp"]]
                per.append(image_metrics([r for r, keep in zip(rles, sc >= st) if keep], valset.gt[s["sample_id"]]))
            rows.append({"score_thr": st, "mask_thr": mask_thr, "min_area": min_area, "overlap_keep": ok, **aggregate(per)})
    df = pd.DataFrame(rows).sort_values("pq", ascending=False)
    best = {k: (int(v) if k == "min_area" else float(v)) for k, v in df.iloc[0][keys].items()}
    log("tuned post-process:", best, "| PQ", round(df.iloc[0]["pq"], 4))
    return best, df


# --------------------------------------------------------------------------- submission
def rle_counts(rle: dict) -> str:
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def write_submission(dets_by_ts: dict, pp: dict, out_csv: Path) -> pd.DataFrame:
    rows = []
    for ts in sorted(dets_by_ts):
        insts = postprocess(dets_by_ts[ts], **{k: pp[k] for k in PP_KEYS})
        for k, rle in enumerate(instances_to_rles(insts), start=1):
            rows.append({"filament_id": f"{ts}_{k}", "segmentation_rle": rle_counts(rle)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def validate_submission(csv_path: Path, test_dir: Path, log=print) -> bool:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    test_ids = {p.stem for p in Path(test_dir).glob("*.jp*g")}
    ok = True

    def bad(msg):
        nonlocal ok
        ok = False
        log("CSV CHECK FAILED:", msg)

    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        bad(f"columns {list(df.columns)}")
    if df["filament_id"].duplicated().any():
        bad("duplicate filament_id")
    pat = re.compile(r"^(\d{14}[A-Za-z]{2})_(\d+)$")
    img_ids = set()
    for fid in df["filament_id"]:
        mm = pat.match(fid)
        if not mm:
            bad(f"bad filament_id {fid}")
            break
        img_ids.add(mm.group(1))
    if img_ids - test_ids:
        bad(f"ids not in test set: {sorted(img_ids - test_ids)[:5]}")
    empty = 0
    for s in df["segmentation_rle"]:
        if s.startswith(("'", '"', "{", "b'")):
            bad("RLE must be bare counts string")
            break
        m = mask_utils.decode({"size": [IMG_SIZE, IMG_SIZE], "counts": s.encode("ascii")})
        if m.shape != (IMG_SIZE, IMG_SIZE):
            bad(f"mask shape {m.shape}")
            break
        empty += int(m.sum() == 0)
    if empty:
        bad(f"{empty} empty masks")
    log(f"CSV: {len(df)} rows, {len(img_ids)}/{len(test_ids)} test images with predictions, "
        f"{len(df) / max(len(img_ids), 1):.1f} filaments/img -> {'OK' if ok else 'FAILED'}")
    return ok


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--weights", required=True, nargs="+", help="one or more .pt files (ensemble)")
    ap.add_argument("--pp", default=None, help="json file with tuned post-process params")
    ap.add_argument("--tune", action="store_true", help="tune post-processing on validation")
    ap.add_argument("--submit", action="store_true", help="predict test set and write CSV")
    ap.add_argument("--out", default="submissions/submission.csv")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    seed_everything(cfg["seed"])
    log = Logger()
    device = get_device()
    model = []
    for wpath in args.weights:
        m = build_model(cfg["model"], pretrained=False)
        load_weights(m, resolve(wpath))
        model.append(m.to(device))
    log(f"loaded {len(model)} model(s): {args.weights}; tta views: {cfg['model'].get('tta') or ['none']}")
    amp = cfg["train"].get("amp", True)

    pp = dict(cfg["postprocess"])
    if args.pp:
        pp.update(json.loads(Path(resolve(args.pp)).read_text()))

    if args.tune:
        vs = ValSet(cfg)
        dets = run_inference(model, vs.paths, cfg["model"], device, amp, desc="val")
        log("val (default pp):", fmt(vs.score(dets, pp)))
        pp, df = tune_postprocess(vs, dets, cfg["tune"], log)
        log("val (tuned pp):  ", fmt(vs.score(dets, pp)))
        out = resolve(args.weights[0]).with_suffix(".ens.pp.json" if len(args.weights) > 1 else ".pp.json")
        out.write_text(json.dumps(pp, indent=1))
        log("saved", out)

    if args.submit:
        test_dir = resolve(cfg["data"]["root"]) / cfg["data"]["test_images"]
        paths = sorted(test_dir.glob("*.jp*g"))
        t0 = time.time()
        dets = run_inference(model, paths, cfg["model"], device, amp, desc="test")
        log(f"test inference: {len(paths)} images in {time.time() - t0:.0f}s")
        out_csv = resolve(args.out)
        write_submission(dets, pp, out_csv)
        log("wrote", out_csv, "with pp", pp)
        validate_submission(out_csv, test_dir, log)


if __name__ == "__main__":
    main()
