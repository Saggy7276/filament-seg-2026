"""Train Mask R-CNN on random full-resolution crops; validate with PQ on full images.

  python -m src.train --config configs/baseline.yaml

Each run writes runs/<name>_<time>/ (config, log, last.pt, best.pt, best.pp.json)
and appends a summary row to results/results.csv.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time

import torch
from torch.utils.data import DataLoader

from .data import TrainDataset, collate, get_split, load_samples
from .metrics import fmt
from .models import build_model, load_weights, save_weights, set_mode
from .predict import ValSet, run_inference, tune_postprocess
from .utils import Logger, append_result, get_device, load_config, resolve, seed_everything


def resume_state(run_dir, model, opt, sched, scaler, iters_per_epoch, tc, log):
    """Restore training state. Full state from ckpt.pt if present; otherwise only the
    weights in last.pt (optimizer momentum is lost) with the epoch/LR schedule rebuilt
    from train.log."""
    import re
    ckpt = run_dir / "ckpt.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        log(f"resumed from ckpt.pt after epoch {ck['epoch']} (best PQ {ck['best_pq']:.4f})")
        return ck["epoch"] + 1, ck["step"], ck["best_pq"]
    text = (run_dir / "train.log").read_text(encoding="utf-8")
    done = max(int(e) for e in re.findall(r"epoch (\d+) done", text))
    pqs = [float(x) for x in re.findall(r"VAL ep \d+: PQ ([0-9.]+)", text)]
    load_weights(model, run_dir / "last.pt")
    step = done * iters_per_epoch // tc["accum_steps"]
    for _ in range(step):
        sched.step()
    best = max(pqs) if pqs else -1.0
    log(f"resumed weights-only from last.pt after epoch {done} (step {step}, best PQ {best:.4f}); momentum reset")
    return done + 1, step, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override (e.g. smoke test)")
    ap.add_argument("--max-iters", type=int, default=None, help="iters per epoch (smoke test)")
    ap.add_argument("--resume", default=None, help="run dir to continue (uses ckpt.pt, else last.pt)")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seed = cfg["seed"]
    seed_everything(seed)
    tc, mc = cfg["train"], cfg["model"]

    if args.resume:
        run_dir = resolve(args.resume)
    else:
        run_dir = resolve("runs") / f"{cfg['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True)
        shutil.copy(resolve(args.config), run_dir / "config.yaml")
    log = Logger(run_dir / "train.log")
    log("resume" if args.resume else "run", run_dir.name, "seed", seed)

    root = resolve(cfg["data"]["root"])
    samples = load_samples(root / cfg["data"]["train_json"])
    split = get_split(samples, resolve(cfg["data"]["split_file"]), cfg["data"]["val_frac"], seed)
    train_s = [s for s in samples if s["timestamp"] in set(split["train"])]
    valset = ValSet(cfg)
    log(f"train samples {len(train_s)} ({len(split['train'])} imgs) | val samples {len(valset.samples)} ({len(split['val'])} imgs)")

    ds = TrainDataset(train_s, root / cfg["data"]["train_images"], tc, cfg["aug"])
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, num_workers=tc["num_workers"],
                    collate_fn=collate, generator=g, persistent_workers=tc["num_workers"] > 0,
                    pin_memory=True, drop_last=True)

    device = get_device()
    model = build_model(mc).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=tc["lr"], momentum=tc["momentum"], weight_decay=tc["weight_decay"])
    iters_per_epoch = min(len(dl), args.max_iters or len(dl))
    total_steps = max(1, tc["epochs"] * iters_per_epoch // tc["accum_steps"])
    warm = tc["warmup_steps"]

    def lr_lambda(step):
        if step < warm:
            return 0.001 + (1 - 0.001) * step / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total_steps - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_pq, best_metrics, step, start_epoch = -1.0, None, 0, 1
    if args.resume:
        start_epoch, step, best_pq = resume_state(run_dir, model, opt, sched, scaler, iters_per_epoch, tc, log)
    t_start = time.time()
    for epoch in range(start_epoch, tc["epochs"] + 1):
        seed_everything(seed + epoch)  # per-epoch seeding makes resumed runs reproducible
        set_mode(model, mc, train=True)
        run_loss, n, t0 = {}, 0, time.time()
        opt.zero_grad(set_to_none=True)
        for i, (imgs, tgts) in enumerate(dl):
            if i >= iters_per_epoch:
                break
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]
            with torch.autocast(device_type="cuda", enabled=use_amp):
                losses = model(imgs, tgts)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                log("non-finite loss, skipping batch", {k: float(v) for k, v in losses.items()})
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / tc["accum_steps"]).backward()
            if (i + 1) % tc["accum_steps"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
            for k, v in losses.items():
                run_loss[k] = run_loss.get(k, 0.0) + float(v)
            n += 1
            if (i + 1) % tc["log_every"] == 0:
                log(f"ep {epoch} it {i + 1}/{iters_per_epoch} loss {sum(run_loss.values()) / n:.4f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} mem {torch.cuda.max_memory_allocated() / 2**30:.2f}G")
        log(f"epoch {epoch} done in {time.time() - t0:.0f}s | " +
            " ".join(f"{k[5:]} {v / max(n, 1):.4f}" for k, v in run_loss.items()))
        save_weights(model, run_dir / "last.pt")
        torch.save({"epoch": epoch, "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best_pq": max(best_pq, -1.0)}, run_dir / "ckpt.pt")

        if epoch % tc["eval_every"] == 0 or epoch == tc["epochs"]:
            torch.cuda.empty_cache()
            dets = run_inference(model, valset.paths, mc, device, use_amp, desc="val")
            m = valset.score(dets, cfg["postprocess"])
            log(f"VAL ep {epoch}: {fmt(m)}")
            if m["pq"] > best_pq:
                best_pq, best_metrics = m["pq"], m
                save_weights(model, run_dir / "best.pt")
                log("  new best -> best.pt")
                ck = torch.load(run_dir / "ckpt.pt", weights_only=False)
                ck["best_pq"] = best_pq
                torch.save(ck, run_dir / "ckpt.pt")
                del ck
            del dets
            gc.collect()
            torch.cuda.empty_cache()

    # tune post-processing on validation with the best checkpoint
    load_weights(model, run_dir / "best.pt")
    model.to(device)
    dets = run_inference(model, valset.paths, mc, device, use_amp, desc="val")
    pp, df = tune_postprocess(valset, dets, cfg["tune"], log)
    df.to_csv(run_dir / "tune.csv", index=False)
    (run_dir / "best.pp.json").write_text(json.dumps(pp, indent=1))
    m = valset.score(dets, pp)
    log(f"FINAL VAL (tuned pp): {fmt(m)}")

    append_result({
        "run": run_dir.name, "seed": seed, "epochs": tc["epochs"],
        "train_min": round((time.time() - t_start) / 60, 1),
        **{f"val_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "val_pq_default_pp": round(best_pq, 4), "pp": pp, "config": dict(cfg),
    })
    log("done")


if __name__ == "__main__":
    main()
