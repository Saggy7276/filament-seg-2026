"""Small shared helpers: config loading, seeding, logging, results file."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v


def load_config(path: str | Path) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        return Cfg(yaml.safe_load(f))


def resolve(p: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger:
    def __init__(self, path: Path | None = None):
        self.path = path

    def __call__(self, *msg):
        line = time.strftime("%H:%M:%S ") + " ".join(str(m) for m in msg)
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def append_result(row: dict, path: str | Path = "results/results.csv") -> None:
    """Append one experiment summary row (metrics + config) to the shared results file."""
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
    exists = path.exists()
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        fields = header + [k for k in row if k not in header]
    else:
        fields = list(row)
    if exists and fields != header:  # new columns: rewrite with the extended header
        with open(path, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(old)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)
