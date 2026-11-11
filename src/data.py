"""Data: COCO parsing, timestamp-grouped train/val split, and PyTorch datasets.

Only the `segmentation` polygons are used as training targets (competition rule:
no spines, bboxes or class labels). All filaments are treated as one class.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

IMG_SIZE = 2048


# --------------------------------------------------------------------------- parsing
def load_samples(json_path: str | Path) -> list[dict]:
    """One sample per annotated image id (annotator prefix + timestamp).

    Returns dicts: sample_id, file_name, timestamp, polygons (list of instances,
    each a list of flat [x0, y0, x1, y1, ...] polygons).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    samples = {}
    for im in d["images"]:
        samples[im["id"]] = {
            "sample_id": im["id"],
            "file_name": im["file_name"],
            "timestamp": Path(im["file_name"]).stem,
            "height": im["height"],
            "width": im["width"],
            "polygons": [],
        }
    for a in d["annotations"]:
        samples[a["image_id"]]["polygons"].append(a["segmentation"])
    return list(samples.values())


def observatory(timestamp: str) -> str:
    return timestamp[-2:]


def gt_rles(sample: dict) -> list[dict]:
    """Full-resolution COCO RLE per GT instance (same rasterisation as annToMask)."""
    h, w = sample["height"], sample["width"]
    return [mask_utils.merge(mask_utils.frPyObjects(polys, h, w)) for polys in sample["polygons"]]


# --------------------------------------------------------------------------- split
def make_split(samples: list[dict], val_frac: float, seed: int) -> dict:
    """Group by timestamp (one physical image), stratified by observatory."""
    by_obs = defaultdict(set)
    for s in samples:
        by_obs[observatory(s["timestamp"])].add(s["timestamp"])
    rng = random.Random(seed)
    val = []
    for obs in sorted(by_obs):
        ts = sorted(by_obs[obs])
        rng.shuffle(ts)
        val += ts[: max(1, round(len(ts) * val_frac))]
    all_ts = sorted({s["timestamp"] for s in samples})
    val = sorted(val)
    train = sorted(set(all_ts) - set(val))
    return {"seed": seed, "val_frac": val_frac, "train": train, "val": val}


def get_split(samples: list[dict], split_file: Path, val_frac: float, seed: int) -> dict:
    """Load the saved split, or create and save it on first use."""
    if split_file.exists():
        with open(split_file, "r", encoding="utf-8") as f:
            split = json.load(f)
    else:
        split = make_split(samples, val_frac, seed)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=1)
    assert not set(split["train"]) & set(split["val"]), "train/val leak"
    return split


# --------------------------------------------------------------------------- io
def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_model_input(img: np.ndarray) -> torch.Tensor:
    """uint8 HxW grayscale -> float [3,H,W] in [0,1].

    The single channel is repeated only because the pretrained backbone expects
    3 input channels; no colour information is invented.
    """
    t = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0)
    return t.unsqueeze(0).expand(3, -1, -1).contiguous()


# --------------------------------------------------------------------------- datasets
class TrainDataset(Dataset):
    """Random full-resolution crops with instance masks for Mask R-CNN."""

    def __init__(self, samples: list[dict], image_dir: Path, tcfg: dict, acfg: dict):
        self.samples = samples
        self.image_dir = Path(image_dir)
        self.crop = int(tcfg["crop_size"])
        self.p_focus = float(tcfg["p_focus_crop"])
        self.min_area = int(tcfg["min_instance_area"])
        self.aug = acfg
        # polygon arrays (float32, Nx2) per instance, used for rasterisation & focus crops
        self.poly_arrays = [
            [[np.asarray(p, np.float32).reshape(-1, 2) for p in inst] for inst in s["polygons"]]
            for s in samples
        ]

    def __len__(self):
        return len(self.samples)

    def _crop_origin(self, idx: int, h: int, w: int) -> tuple[int, int]:
        c = self.crop
        insts = self.poly_arrays[idx]
        if insts and random.random() < self.p_focus:
            pts = random.choice(insts)[0]
            cx, cy = pts.mean(0)
            cx += random.uniform(-0.35, 0.35) * c
            cy += random.uniform(-0.35, 0.35) * c
            x0, y0 = int(cx - c / 2), int(cy - c / 2)
        else:
            x0, y0 = random.randint(0, w - c), random.randint(0, h - c)
        return int(np.clip(x0, 0, w - c)), int(np.clip(y0, 0, h - c))

    def _photometric(self, img: np.ndarray) -> np.ndarray:
        a = self.aug
        x = img.astype(np.float32) / 255.0
        if a.get("gamma", 0):
            x = np.power(x, np.exp(random.uniform(-a["gamma"], a["gamma"])))
        if a.get("contrast", 0):
            m = x.mean()
            x = (x - m) * (1 + random.uniform(-a["contrast"], a["contrast"])) + m
        if a.get("brightness", 0):
            x = x + random.uniform(-a["brightness"], a["brightness"]) * 0.5
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = read_gray(self.image_dir / s["file_name"])
        h, w = img.shape
        x0, y0 = self._crop_origin(idx, h, w)
        c = self.crop
        img = img[y0 : y0 + c, x0 : x0 + c]

        masks = []
        off = np.array([x0, y0], np.float32)
        for inst in self.poly_arrays[idx]:
            m = np.zeros((c, c), np.uint8)
            cv2.fillPoly(m, [np.round(p - off).astype(np.int32) for p in inst], 1)
            if m.sum() >= self.min_area:
                masks.append(m)
        masks = np.stack(masks) if masks else np.zeros((0, c, c), np.uint8)

        # geometric augmentation (identical for image and masks)
        a = self.aug
        if random.random() < a.get("hflip", 0):
            img, masks = img[:, ::-1], masks[:, :, ::-1]
        if random.random() < a.get("vflip", 0):
            img, masks = img[::-1], masks[:, ::-1]
        if a.get("rot90", False):
            k = random.randint(0, 3)
            img, masks = np.rot90(img, k), np.rot90(masks, k, axes=(1, 2))
        img = self._photometric(np.ascontiguousarray(img))
        masks = np.ascontiguousarray(masks)

        boxes = []
        for m in masks:
            ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
            boxes.append([xs[0], ys[0], xs[-1] + 1, ys[-1] + 1])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": torch.from_numpy(masks),
        }
        return to_model_input(img), target


def collate(batch):
    return tuple(zip(*batch))
