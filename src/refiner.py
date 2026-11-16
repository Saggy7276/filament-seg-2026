"""Stage 2: crop refiner. A small U-Net re-segments every detected filament on a
native-resolution square crop centred on the detection.

Idea adapted from the public "Mask R-CNN + U-Net Refiner" Kaggle notebooks for this
competition (Nomannic19; Fususu's LB 0.70 handoff). The code and the weights here are
our own: the refiner is trained from scratch (ImageNet encoder) on our training split,
using segmentation masks only.

Why it helps: Mask R-CNN predicts each mask on a 28x28 grid inside its box, which is
coarse for thin filaments. The refiner sees the same object at up to native resolution
(a 96 px crop is upsampled to 256 px) with surrounding context.

  python -m src.refiner --config configs/refiner.yaml          # train
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .data import get_split, load_samples, read_gray
from .utils import Logger, append_result, get_device, load_config, resolve, seed_everything


# --------------------------------------------------------------------------- model
def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class CropUNet(nn.Module):
    """ResNet18 encoder + U-Net decoder; input uint8-scaled grey crop [B,1,S,S] in [0,1], output logits."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # 1/2, 64
        self.pool = enc.maxpool
        self.l1, self.l2, self.l3, self.l4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.d4 = _block(512 + 256, 256)
        self.d3 = _block(256 + 128, 128)
        self.d2 = _block(128 + 64, 64)
        self.d1 = _block(64 + 64, 48)
        self.d0 = _block(48, 24)
        self.head = nn.Conv2d(24, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _up(x, ref=None):
        size = ref.shape[-2:] if ref is not None else [2 * s for s in x.shape[-2:]]
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        x0 = self.stem(x)
        x1 = self.l1(self.pool(x0))
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        y = self.d4(torch.cat([self._up(x4, x3), x3], 1))
        y = self.d3(torch.cat([self._up(y, x2), x2], 1))
        y = self.d2(torch.cat([self._up(y, x1), x1], 1))
        y = self.d1(torch.cat([self._up(y, x0), x0], 1))
        return self.head(self.d0(self._up(y)))


def load_refiner(path: str | Path, device) -> tuple[CropUNet, dict]:
    ck = torch.load(resolve(path), map_location="cpu", weights_only=True)
    model = CropUNet(pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["cfg"]


# --------------------------------------------------------------------------- crop geometry
def square_bounds(x0: float, y0: float, x1: float, y1: float, context: float, min_side: int,
                  size: int = 2048, shift=(0.0, 0.0)) -> tuple[int, int, int, int]:
    """Square crop around a box: side = max(w, h) * context (>= min_side), kept inside the image."""
    side = int(np.ceil(max(x1 - x0, y1 - y0) * context))
    side = min(max(side, min_side), size)
    cx, cy = (x0 + x1) / 2 + shift[0], (y0 + y1) / 2 + shift[1]
    bx = int(np.clip(round(cx - side / 2), 0, size - side))
    by = int(np.clip(round(cy - side / 2), 0, size - side))
    return bx, by, bx + side, by + side


def mask_box(x0: int, y0: int, m: np.ndarray) -> tuple[int, int, int, int]:
    """Tight image-coordinate box of a bool crop mask placed at (x0, y0)."""
    ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return x0 + xs[0], y0 + ys[0], x0 + xs[-1] + 1, y0 + ys[-1] + 1


def _resize(a: np.ndarray, s: int) -> np.ndarray:
    return cv2.resize(a, (s, s), interpolation=cv2.INTER_AREA if a.shape[0] > s else cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- inference
FLIPS = {"none": (), "hflip": (-1,), "vflip": (-2,), "hvflip": (-2, -1)}


@torch.no_grad()
def refine(model: CropUNet, rcfg: dict, img: np.ndarray, insts: list, device, tta=None,
           batch_size: int = 32) -> list[tuple[int, int, np.ndarray]]:
    """Re-segment instances. insts: (x0, y0, bool crop mask, ...) tuples.

    Returns one (x0, y0, float32 probability crop) per instance, at native resolution
    over the square refiner window.
    """
    if not insts:
        return []
    s = int(rcfg["size"])
    bounds = [square_bounds(*mask_box(i[0], i[1], i[2]), rcfg["context"], rcfg["min_side"], img.shape[0])
              for i in insts]
    crops = np.stack([_resize(img[b[1]:b[3], b[0]:b[2]], s) for b in bounds])
    views = list(tta) if tta else ["none"]
    probs = []
    for k in range(0, len(crops), batch_size):
        x = torch.from_numpy(crops[k:k + batch_size]).to(device).float().div_(255).unsqueeze(1)
        acc = 0
        for v in views:
            d = FLIPS[v]
            p = model(torch.flip(x, d) if d else x).sigmoid()
            acc = acc + (torch.flip(p, d) if d else p)
        probs.append((acc / len(views))[:, 0].cpu().numpy())
    probs = np.concatenate(probs)
    out = []
    for b, p in zip(bounds, probs):
        side = b[2] - b[0]
        out.append((b[0], b[1], cv2.resize(p, (side, side), interpolation=cv2.INTER_LINEAR)))
    return out


def refine_instances(model: CropUNet, rcfg: dict, img: np.ndarray, insts: list, device,
                     thr: float = 0.4, tta=("none", "hflip", "vflip", "hvflip"), min_area: int = 0) -> list[dict]:
    """Detector instances -> final RLEs: refine each, binarise at `thr`, merge refined masks that touch.

    An instance keeps its detector mask if the refiner returns an empty mask.
    """
    from .postprocess import instances_to_rles, merge_touching

    t = int(round(thr * 255))  # same 8-bit quantisation as the validation study (tools/eval_refiner.py)
    out = []
    for inst, (x0, y0, p) in zip(insts, refine(model, rcfg, img, insts, device, tta)):
        m = np.round(p * 255).astype(np.uint8) > t
        m3 = (x0, y0, m) if m.any() else (inst[0], inst[1], inst[2])
        if m3[2].sum() >= max(min_area, 1):
            out.append(m3)
    return merge_touching(instances_to_rles(out, img.shape[0]))


# --------------------------------------------------------------------------- training data
def build_cache(samples: list[dict], image_dir: Path, rcfg: dict, path: Path, log=print) -> dict:
    """One wide crop per GT instance (image + that instance's mask), resized to cache_size.

    The cache window is wider than any training window, so context/shift jitter is done
    in memory without re-reading the 2048^2 JPEGs.
    """
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    cs, t0 = int(rcfg["cache_size"]), time.time()
    imgs, masks, meta = [], [], []
    by_file: dict[str, list] = {}
    for smp in samples:
        by_file.setdefault(smp["file_name"], []).extend(smp["polygons"])
    for n, (fn, insts) in enumerate(sorted(by_file.items())):
        img = read_gray(image_dir / fn)
        for inst in insts:
            pts = [np.asarray(p, np.float32).reshape(-1, 2) for p in inst]
            allp = np.concatenate(pts)
            x0, y0 = allp.min(0)
            x1, y1 = allp.max(0)
            ext = max(x1 - x0, y1 - y0)
            if ext < rcfg["min_extent"]:
                continue
            b = square_bounds(x0, y0, x1, y1, rcfg["cache_context"],
                              int(rcfg["min_side"] * rcfg["cache_context"] / rcfg["context"]), img.shape[0])
            side = b[2] - b[0]
            m = np.zeros((side, side), np.uint8)
            cv2.fillPoly(m, [np.round(p - np.array([b[0], b[1]], np.float32)).astype(np.int32) for p in pts], 255)
            imgs.append(_resize(img[b[1]:b[3], b[0]:b[2]], cs))
            masks.append(_resize(m, cs))
            # object centre (fraction of cache window), object extent and window side in native px
            meta.append([((x0 + x1) / 2 - b[0]) / side, ((y0 + y1) / 2 - b[1]) / side, ext, side])
        if n % 100 == 0:
            log(f"cache {path.name}: {n}/{len(by_file)} images ({time.time() - t0:.0f}s)")
    cache = {"img": np.stack(imgs), "mask": np.stack(masks), "meta": np.asarray(meta, np.float32)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **cache)
    return cache


def sample_window(cache: dict, i: int, rcfg: dict, train: bool) -> tuple[np.ndarray, np.ndarray]:
    """Cut the refiner window for instance i out of its cached wide crop (jittered when training)."""
    s, cs = int(rcfg["size"]), cache["img"].shape[1]
    cx, cy, ext, side = cache["meta"][i]
    ctx = random.uniform(*rcfg["context_jitter"]) if train else rcfg["context"]
    f = min(max(ext * ctx, rcfg["min_side"]) / side, 1.0)  # window side as a fraction of the cache window
    if train:
        j = rcfg["shift_jitter"] * ext / side
        cx, cy = cx + random.uniform(-j, j), cy + random.uniform(-j, j)
    w = max(int(round(f * cs)), 8)
    ax = int(np.clip(round(cx * cs - w / 2), 0, cs - w))
    ay = int(np.clip(round(cy * cs - w / 2), 0, cs - w))
    img = _resize(cache["img"][i, ay:ay + w, ax:ax + w], s)
    msk = _resize(cache["mask"][i, ay:ay + w, ax:ax + w], s)
    if train:
        if random.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if random.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        k = random.randint(0, 3)
        img, msk = np.rot90(img, k), np.rot90(msk, k)
        x = img.astype(np.float32) / 255
        x = np.power(x, np.exp(random.uniform(-0.2, 0.2)))
        mu = x.mean()
        x = (x - mu) * (1 + random.uniform(-0.15, 0.15)) + mu + random.uniform(-0.075, 0.075)
        img = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(img), np.ascontiguousarray(msk)


def make_batch(cache: dict, idx, rcfg: dict, train: bool, device):
    pairs = [sample_window(cache, int(i), rcfg, train) for i in idx]
    x = torch.from_numpy(np.stack([p[0] for p in pairs])).to(device).float().div_(255).unsqueeze(1)
    y = (torch.from_numpy(np.stack([p[1] for p in pairs])).to(device) > 127).float().unsqueeze(1)
    return x, y


def seg_loss(logits, y):
    """BCE + soft Dice."""
    p = logits.sigmoid()
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, y) + dice.mean()


@torch.no_grad()
def evaluate(model, cache, rcfg, device, bs: int = 64) -> dict:
    """Mean IoU / Dice of the refiner on GT-centred validation windows (threshold 0.5)."""
    model.eval()
    ious, dices = [], []
    for k in range(0, len(cache["img"]), bs):
        x, y = make_batch(cache, range(k, min(k + bs, len(cache["img"]))), rcfg, False, device)
        p = model(x).sigmoid() > 0.5
        inter = (p & (y > 0.5)).sum((1, 2, 3)).float()
        a, b = p.sum((1, 2, 3)).float(), y.sum((1, 2, 3))
        ious += (inter / (a + b - inter).clamp(min=1)).tolist()
        dices += (2 * inter / (a + b).clamp(min=1)).tolist()
    return {"iou": float(np.mean(ious)), "dice": float(np.mean(dices)), "iou>0.5": float(np.mean(np.array(ious) > 0.5))}


# --------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/refiner.yaml")
    args = ap.parse_args()
    cfg = load_config(resolve(args.config))
    rcfg, tcfg = dict(cfg["refiner"]), cfg["train"]
    seed_everything(cfg["seed"])
    run = resolve("runs") / f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True)
    log = Logger(run / "train.log")
    log("config", json.dumps(dict(cfg)), "| seed", cfg["seed"])

    root = resolve(cfg["data"]["root"])
    samples = load_samples(root / cfg["data"]["train_json"])
    split = get_split(samples, resolve(cfg["data"]["split_file"]), cfg["data"]["val_frac"], cfg["seed"])
    img_dir = root / cfg["data"]["train_images"]
    tag = f"c{rcfg['cache_size']}_x{rcfg['cache_context']}"
    tr = build_cache([s for s in samples if s["timestamp"] in set(split["train"])], img_dir, rcfg,
                     resolve(f"runs/refiner_cache_train_{tag}.npz"), log)
    va = build_cache([s for s in samples if s["timestamp"] in set(split["val"])], img_dir, rcfg,
                     resolve(f"runs/refiner_cache_val_{tag}.npz"), log)
    all_data = bool(cfg["data"].get("train_on_all"))
    if all_data:
        # final-model mode: train on train + val instances. The "val" numbers below are then
        # measured on training data and are NOT a valid estimate; last.pt is the model to use.
        tr = {k: np.concatenate([tr[k], va[k]]) for k in tr}
        log("train_on_all: validation instances added to training; val metrics are not valid, use last.pt")
    log(f"instances: train {len(tr['img'])}, val {len(va['img'])}")

    device = get_device()
    model = CropUNet(pretrained=True).to(device)
    bs, epochs = int(tcfg["batch_size"]), int(tcfg["epochs"])
    steps = epochs * (len(tr["img"]) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=steps, pct_start=0.1)
    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(tr["img"]))
        t0, tot = time.time(), 0.0
        nb = len(perm) // bs
        for b in range(nb):
            x, y = make_batch(tr, perm[b * bs:(b + 1) * bs], rcfg, True, device)
            loss = seg_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        m = evaluate(model, va, rcfg, device)
        log(f"ep {ep}/{epochs} loss {tot / nb:.4f} | val IoU {m['iou']:.4f} Dice {m['dice']:.4f} "
            f"IoU>0.5 {m['iou>0.5']:.3f} | {time.time() - t0:.0f}s")
        ck = {"model": model.state_dict(), "cfg": rcfg, "epoch": ep, "val": m}
        torch.save(ck, run / "last.pt")
        if m["iou"] > best:
            best = m["iou"]
            torch.save(ck, run / "best.pt")
    log("done; best val IoU", round(best, 4))
    append_result({"run": run.name, "val_refiner_iou": round(best, 4), "config": dict(cfg)})


if __name__ == "__main__":
    main()
