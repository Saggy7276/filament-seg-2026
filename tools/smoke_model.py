"""Quick GPU check of a model config: a few training iterations (memory, speed) and one
full-resolution inference.

  python tools/smoke_model.py --config configs/v2_anchors.yaml [--iters 20]
"""
import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from src.data import TrainDataset, load_samples, read_gray  # noqa: E402
from src.models import build_model, set_mode  # noqa: E402
from src.predict import infer_image  # noqa: E402
from src.utils import load_config, resolve, seed_everything  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--iters", type=int, default=20)
args = ap.parse_args()

cfg = load_config(resolve(args.config))
seed_everything(0)
root = resolve(cfg["data"]["root"])
samples = load_samples(root / cfg["data"]["train_json"])
ds = TrainDataset(samples, root / cfg["data"]["train_images"], cfg["train"], cfg["aug"])
model = build_model(cfg["model"]).cuda()
print("params (M):", round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
set_mode(model, cfg["model"], train=True)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=cfg["train"]["lr"], momentum=0.9)
batches = [ds[i] for i in random.sample(range(len(ds)), args.iters)]
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
t = time.time()
nan = 0
for img, tg in batches:
    tg = {k: v.cuda() for k, v in tg.items()}
    losses = model([img.cuda()], [tg])
    loss = sum(losses.values())
    nan += int(not torch.isfinite(loss))
    opt.zero_grad()
    loss.backward()
    opt.step()
torch.cuda.synchronize()
print(f"train: {(time.time() - t) / args.iters:.2f} s/it, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
      f"non-finite {nan}/{args.iters}, last losses " + " ".join(f"{k[5:]} {float(v):.3f}" for k, v in losses.items()))
set_mode(model, cfg["model"], train=False)
torch.cuda.reset_peak_memory_stats()
p = sorted((root / cfg["data"]["test_images"]).glob("*.jp*g"))[0]
torch.cuda.synchronize()
t = time.time()
dets = infer_image(model, read_gray(p), torch.device("cuda"), False)
torch.cuda.synchronize()
print(f"infer 2048: {time.time() - t:.2f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, {len(dets)} dets")
