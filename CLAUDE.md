# CLAUDE.md: Solar Filament Segmentation Challenge 2026

## Project summary
Kaggle competition (NSF NSO sponsor): instance segmentation of solar filaments in GONG H-alpha images.
- Competition page: https://www.kaggle.com/competitions/filament-segmentation-2026/
- Self-evaluation notebook (reference for scoring): https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
- Goal: one polygon/mask per filament, scored mainly by Panoptic Quality (PQ).

## Deadlines
- Final submission deadline: Nov 15, 2026, 11:59 pm CDT (about 10:30 am IST on Nov 16).
- Winners announced: Nov 30, 2026.
- Optional: IEEE BigData presentation, Dec 14-17, 2026.

## Data
Root folder: MAGFiLO_1.0_Kaggle_2026/
- train/train_images/: training JPEGs
- train/MAGFiLO_1.0_Annotations_kaggle2026_train.json: COCO-style annotations
- test/test_images/: test JPEGs (no labels)

Image facts:
- 2048 x 2048, 8-bit grayscale JPEG (converted from GONG FITS). Load as single channel. Do not treat as RGB.
- Filename format: YYYYMMDDHHMMSSII. The last two characters are the observatory code (e.g., Bh = Big Bear, Lh = Learmonth, Mh appears in examples).

Annotation facts:
- Format: COCO. Each annotation has `segmentation` (one polygon, flat [x0,y0,x1,y1,...], first point repeated at end), `bbox` [x,y,w,h], `area`, `spine`, `iscrowd` = 0, `category_id`.
- Categories: 1 Left, 2 Right, 3 Unidentifiable, 4 Ambiguous. Left/Right are likely filament chirality (verify in the MAGFiLO paper, DOI 10.1038/s41597-024-03876-y).
- Several filaments per image.
- The same image can appear multiple times with different annotator prefixes (e.g., 010101-20260901165702Bh and 010102-20260901165702Bh). These are separate samples, but they share one image.

## Rules that affect implementation
- Train only on segmentation masks. The rules say models may not use other ground-truth metadata (spines, bboxes, class labels) for training. Confirm with organizers if unsure.
- For ML inference, use only the test images in test/test_images/.
- External data is allowed if publicly available and free for all participants. Pretrained weights are fine.
- Use the data for non-commercial and research purposes only.
- No private code or data sharing outside the team. Team size maximum is 5.
- Submission limit: 5 per day. Pick 2 final submissions. Do not tune against the public leaderboard, and never use hand labeling or human prediction on test data.

## Evaluation
Primary: Panoptic Quality (PQ).
- A predicted mask matches a ground-truth mask if IoU > 0.5.
- PQ = (sum of IoU over TP) / (|TP| + 0.5|FP| + 0.5|FN|).
- Penalties apply to fragmentation (one GT split into several predictions) and over-merging (several GT filaments merged into one prediction).
- Implement PQ ourselves. pycocotools COCOeval computes AP, not PQ.

Secondary (rubric, 70% quantitative total):
- Mean Dice using torchmetrics.segmentation.DiceScore.
- Distributions of Dice and IoU.
- Counts of one-to-many and many-to-one matches.
- Computational efficiency of the pipeline.

Qualitative (30%): pipeline description, visual quality of masks on H-alpha images, code quality (modularity, documentation).

## Submission format
Single CSV for the whole test set, one row per predicted filament:
- filament_id: image_id + "_" + index, e.g., 20150125172714Mh_1. Index only needs to make rows unique.
- segmentation_rle: RLE counts string only (no size field, no quotes).
- Masks are 2048 x 2048.
- Convert masks/polygons with pycocotools: annToMask, decodeMask, encodeMask.

## Required final deliverables (needed for prize eligibility)
1. 4-page technical report, single PDF, using the Overleaf template from the competition page. Covers preprocessing, model, training, post-processing, and reproduction steps.
2. Public Git repo (GitHub, GitLab, or Bitbucket). Must be accessible without extra permissions and remain public until winners are announced.
3. requirements.txt with all packages and exact versions.
4. A Jupyter notebook that runs the whole pipeline end to end.
5. Model weights committed to the repo, loaded by the notebook. Results must be reproducible without retraining.
6. Google form submission before the deadline.

## Prize
Up to $3,000 total. Prize weight is score^2 (normalized). Team prize is split equally among verified members. Accepting a prize requires releasing full source code and the technical report.

## Planned repo layout (adjust as needed)
```
filament-seg/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/           # YAML configs for experiments
├── src/
│   ├── data.py        # loading, parsing COCO JSON, timestamp-based splits
│   ├── metrics.py     # PQ and Dice implementations
│   ├── models.py      # model definitions
│   ├── train.py
│   ├── postprocess.py # mask cleanup, instance splitting, min-area filter
│   ├── predict.py     # test inference, CSV/RLE writer
│   └── utils.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── pipeline.ipynb # end-to-end reproduction, required for submission
├── weights/           # committed final weights
├── reports/           # 4-page report source and PDF
└── tests/
```

## Working rules for Claude Code
- Validation split: group by timestamp (filename without annotator prefix). Never let the same image appear in both train and validation. Use the same split for all experiments.
- Set random seeds everywhere and log them with each run.
- Keep all hyperparameters in configs/, not hard-coded.
- Every experiment writes its validation PQ, Dice, and config to a results file.
- Before any submission, verify the CSV: row count, filament_id format, RLE decodes to 2048 x 2048, no empty masks unless intentional.
- Do not use test labels or leaderboard feedback to choose post-processing. Tune on validation only.
- Ask before making large decisions, such as switching models, changing the split strategy, or dropping a class.
- Prefer simple, working baselines first. Get the full pipeline producing a valid submission before improving the model.

## Status (update as work progresses)
- Team: solo (with Claude Code)
- Hardware: local laptop, NVIDIA GTX 1650 4 GB, Windows 11, Python 3.12 venv at .venv/
  - Use fp32 (amp: false). fp16 on this GPU gives NaN mask loss and is ~5x slower.
  - ~0.52 s/iter at 1024 crops, ~1.9 GB peak; ~8.5 min/epoch (983 train samples).
- Chosen approach: class-agnostic Mask R-CNN R50-FPN (torchvision, COCO-pretrained),
  trained on 1024 full-res crops, inferred on full 2048 images, greedy score-ordered
  overlap removal + thresholds tuned on validation (src/postprocess.py).
- Split: splits/split_seed42.json, 601 train / 106 val images (171 val samples), grouped by timestamp, stratified by observatory.
- Inter-annotator agreement on multi-annotated train images: PQ ~0.34 (rough ceiling; labels are noisy).
- Gotcha: torchvision sets FrozenBN eps=0 for COCO weights; build_model always does the same, and
  save_weights keeps running_var/mean in fp32 (fp16 underflow -> inf -> no detections).
- Current best validation PQ: 0.4137 tuned (0.3696 default pp), epoch 24 of run maskrcnn_r50_baseline_20261109_193620
  (24 epochs complete). Final weights: weights/maskrcnn_r50_filament.pt + .pp.json (score 0.85, mask 0.5, min_area 0).
- Launching long training: Claude background jobs get killed (low-memory reaper), and Start-Process from the
  PowerShell tool also died with the tool shell. Use WMI so the process is fully detached:
  Invoke-CimMethod Win32_Process Create with `cmd.exe /c cd /d <dir> && .venv\Scripts\python.exe -m src.train ... >> train_stdout.log 2>> train_stderr.log`
  (or the user runs it in their own terminal). Resume command:
  `.venv\Scripts\python -m src.train --config configs/baseline.yaml --resume runs/maskrcnn_r50_baseline_20261109_193620`
- Submissions: v1_ep4 LB 0.33 (val 0.375); v2_ep24 LB 0.36 (val 0.414); v3_ep24_tta LB 0.37 (val 0.419). Final picks: v3 + v2. Log in results/submissions_log.csv. Never tune on LB.
- Public repo: https://github.com/Saggy7276/filament-seg-2026 (GitHub user Saggy7276; gh CLI installed at
  'C:\Program Files\GitHub CLI\gh.exe', logged in). Push with `git push` from the project dir.
- Last completed step: notebooks/pipeline.ipynb executed end to end (configs/baseline_tta.yaml + weights/maskrcnn_r50_filament_tta.pp.json);
  reproduces sub_v3_ep24_tta.csv exactly (val PQ 0.4193). README.md written. Remaining: public repo, report, Google form.
- GPU contention: VALORANT running uses ~4.7 GB GPU memory and makes training ~3x slower.
- TTA study (val, tools/eval_tta.py, cache runs/tta_val_cache.pkl): none 0.4115; none+hflip+vflip 0.4193 (best, used);
  +rot180 0.4179; 4 rotations 0.4154; all 8 dihedral views 0.4169. Rotations do not help.
- Error analysis (final model, val): 472 FN = 266 never detected (small filaments: recall 28% at 200-500 px),
  149 near misses (IoU 0.3-0.5), 57 fragments; 349 FP are large (median 1250 px) and confident (score median 0.92).
  Ruled out: higher-res mask head (28x28 oracle IoU 0.94 >> achieved 0.67), mask dilation (1 px: PQ 0.412 -> 0.392).
- Organisers' scoring code is in src/official_eval.py (verbatim). Their PQ == ours (0.4193 on val). Their secondary
  stats differ: pairwise IoU/Dice over overlapping pairs; 1:n / n:1 counted with ANY overlap (IoU > 0). Notebook reports both.
- Ensemble plan: seed-7 run (configs/seed7.yaml, runs/maskrcnn_r50_seed7_20261111_073534) started 2026-11-11 07:35;
  src/predict.py accepts several --weights and fuses model x view votes with src/tta.fuse.
- Seed-7 run FINISHED 2026-11-11 18:24: best.pt = epoch 16 (default-pp PQ 0.3736; epoch 24 was 0.3676), tuned PQ 0.4058
  (score 0.8 / mask 0.5 / min_area 0). Val caches: runs/tta_val_cache.pkl (seed 42), runs/tta_val_cache_seed7.pkl (seed 7).
- ENSEMBLE RESULT (val, tools/eval_ensemble.py + score-fusion variants): seed42+seed7 x 3 views = 0.4102 (sum/n),
  best variant mean*(k/n)^0.25 = 0.4192 == seed42 alone 0.4193. No gain -> ensemble DROPPED; v3 (seed 42 + flip TTA) stays final.
  seed 7 alone + TTA = 0.4156 (backup only).
- Post-processing test (val, cached seed-42 3-view): merge-touching masks (union of overlapping detections) at score 0.8
  gives PQ 0.4234 vs painting 0.4193 (1:N 43->25, N:1 27->42). Small gain; decide together with the v2 model.
- IMPROVEMENT RUN started 2026-11-11 ~22:30: configs/v2_anchors.yaml = Mask R-CNN R50-FPN **v2** + anchors 16..256
  (src/models.py ARCHS). Smoke: 2.35 GB train, 0.71 s/it, infer 1.8 s/img. Run dir runs/maskrcnn_r50v2_anchors16_*.
  After it finishes: tools/eval_tta.py --config configs/v2_anchors.yaml --weights <run>/best.pt --cache runs/tta_val_cache_v2.pkl
  --combos none none,hflip,vflip ; if val PQ > 0.4193 -> sub_v4, update weights/, notebook, report, repo.
- v2 run (runs/maskrcnn_r50v2_anchors16_20261111_223951) was KILLED by a Windows Update restart 2026-11-12 ~01:26 during
  epoch 6; resumed from ckpt.pt (after epoch 5) on 2026-11-12 07:35 via WMI, logs train_v2_stdout.log / train_v2_stderr.log.
  Val ep 4 default-pp PQ 0.3573 (seed-7 R50 v1 at ep 4: 0.3176). ~0.72 s/it with GPU free -> ~4 h for the 19 remaining epochs.
  Resume: `.venv\Scripts\python -m src.train --config configs/v2_anchors.yaml --resume runs/maskrcnn_r50v2_anchors16_20261111_223951`
  Pause Windows Update before long runs.
- v2 run FINISHED 2026-11-12 16:39: best.pt = epoch 24, default-pp PQ 0.3883, tuned 0.4022 (score 0.8 / mask 0.5 / min_area 0).
  TTA (cache runs/tta_val_cache_v2.pkl): none 0.4022; none+hflip+vflip 0.4169 (score 0.75 / min_area 50).
  0.4169 < 0.4193 -> NO sub_v4; v3 (R50 v1 seed 42 + flip TTA) stays final.
  Ensembles with v2 (val, tools/eval_ensemble.py): v1+v2 x 3 views = 0.4202 (fuse 0.5) / 0.4171 (fuse 0.6); v1+seed7+v2 = 0.4136.
  v1 alone at fuse_iou 0.6 = 0.4207. All within noise of 0.4193 -> v2 and ensembles DROPPED. Only open item: merge-touching pp (0.4234 on v1).
- STAGE-2 CROP REFINER (2026-11-12, idea from public notebook "LB 0.70 Topology-Safe Solar Filaments" by Fususu /
  Nomannic19's Mask R-CNN + U-Net refiner; our own code + weights, their checkpoints NOT used): src/refiner.py,
  configs/refiner.yaml, run runs/refiner_r18_20261112_205302 (ResNet18 U-Net, 256 px crops, context 1.8, 24 epochs,
  batch 16 = 2 GB / 3 min per epoch; batch 32 spills GPU memory -> 25x slower). GT-centred val IoU 0.697.
  Val PQ (tools/eval_refiner.py, v1 seed-42 3-view cache): baseline 0.4193 -> paint 0.4424 / merge 0.4467 / group 0.4456.
  Chosen: score 0.8, mask 0.5, refine thr 0.4, flip-4 refiner TTA, merge touching refined masks (TP 727 FP 368 FN 411).
  Weights weights/refiner_r18_filament.pt; config configs/refined_tta.yaml; predict.py --refiner <pt>.
  Submission submissions/sub_v4_ep24_tta_refined.csv: LB 0.38 (v3 0.37). Notebook re-executed 2026-11-12, reproduces v4
  byte for byte (official PQ 0.4467, pairwise IoU 0.594 / Dice 0.725, torchmetrics Dice 0.6771). README updated.
  TODO: report.tex (still describes v3; regenerate figures from notebook outputs), commit + push (user has not approved yet).
- Tried on top of v4, NO gain (val): refiner-agreement rescue/veto of detections (0.440); v2 detector + refiner 0.4409;
  v1+v2 + refiner 0.4460; v1+seed7+v2 + refiner 0.4411. tools/eval_refiner.py takes several --cache (ensemble).
- Public-repo research 2026-11-12: honest solutions reach LB 0.35-0.41 (Solid-Snake-99 0.41, biconcavelens 0.39-0.40);
  LB scores > ~0.5 are reported to come from leaked test labels (77% of test images are in public MAGFiLO) or a Dice-era
  metric. NEVER train on public MAGFiLO / test-overlap labels.
- cls x3 detector run started 2026-11-12 23:31 via WMI: configs/cls3.yaml (baseline + train.loss_weights
  {loss_classifier: 3.0}), run dir runs/maskrcnn_r50_cls3_*, logs train_cls3_stdout.log / train_cls3_stderr.log, ~4 h.
  After: tools/eval_tta.py --config configs/cls3.yaml --weights <run>/best.pt --cache runs/tta_val_cache_cls3.pkl
  --combos none,hflip,vflip ; then tools/eval_refiner.py --cache runs/tta_val_cache_cls3.pkl --refiner weights/refiner_r18_filament.pt.
  Adopt only if val PQ > 0.4467.
  Run dir runs/maskrcnn_r50_cls3_20261112_233119. KILLED by a system shutdown 2026-11-13 01:14 during epoch 6 (val ep 4
  default-pp PQ 0.3406); resumed from ckpt.pt (after epoch 5) on 2026-11-13 06:38 via WMI, ~0.8 s/it, 19 epochs left (~3-4 h).
  Resume: `.venv\Scripts\python -m src.train --config configs/cls3.yaml --resume runs/maskrcnn_r50_cls3_20261112_233119`
  SLOWDOWN GOTCHA: after each validation pass the training process grew to ~3.75 GB dedicated + 0.85 GB shared GPU memory
  (allocated only 2.6 GB) -> spill into shared memory -> ~3 s/it instead of ~0.6 (epochs 5, 9-12). No other app involved.
  Fix 2026-11-13 10:16: train.gpu_mem_fraction: 0.8 (src/train.py, set in configs/cls3.yaml) + restart from ckpt after epoch 12;
  log now prints `rsv` (reserved). If rsv/speed degrades again after a val pass, kill + resume at an epoch boundary.
  Val default-pp: ep 4 0.3406, ep 8 0.3360, ep 12 0.3614 (best.pt), ep 16 0.3449.
  KILLED again by a reboot 2026-11-13 11:15 during epoch 18; resumed from ckpt.pt (after epoch 17) at 15:44 via WMI
  (7 epochs left, ~1-1.5 h). gpu_mem_fraction 0.8 held after the ep-16 val pass (rsv 2.00G, ~0.56 s/it).
  cls3 run FINISHED 2026-11-13 16:59: best.pt = epoch 24, default-pp PQ 0.3623 (ep 20 0.3603), tuned 0.4040 (score 0.9).
  Flip TTA (runs/tta_val_cache_cls3.pkl): 0.4174 (score 0.85) < v1 0.4193. With refiner: paint 0.4382 / merge 0.4414 /
  group 0.4430, all < 0.4467 -> cls3 DROPPED; v4 (v1 seed 42 + flip TTA + refiner, merge) stays final.
  Report update for v4 was deferred by user (2026-11-13) until this run was done -> now due.
- ZOOM VIEW (2026-11-13, tools/eval_zoom.py, cache runs/tta_val_cache_zoom.pkl): 1024 tiles stride 512 upscaled 2x, NMS,
  combined with the 3-view cache. Val PQ (no refiner): base 0.4193; zoom alone 0.3035; fused as 4th view 0.4036;
  union 0.3856 (alpha 1) / 0.4155 (alpha 0.9). 47 dets/img vs 22, extra ones are confident FPs -> DROPPED.
- FULL-DATA RETRAIN (user approved 2026-11-13; user goal: raise public LB, by valid means only): data.train_on_all: true
  (src/train.py, src/refiner.py) trains on train + val (1154 samples / 707 images). configs/full.yaml (= baseline + that,
  eval_every 24, gpu_mem_fraction 0.8) started 2026-11-13 19:05 via WMI, logs train_full_stdout.log / train_full_stderr.log,
  run dir runs/maskrcnn_r50_full_*, ~4.5 h. Val numbers of this run are NOT valid (val is in training); use last.pt.
  Then: `python -m src.refiner --config configs/refiner_full.yaml` (~1.5 h, use last.pt), then predict test with
  configs/refined_tta.yaml + frozen v4 thresholds (score 0.8 / mask 0.5 / refine 0.4 / merge) -> sub_v5_full.csv.
  Resume: `.venv\Scripts\python -m src.train --config configs/full.yaml --resume runs/maskrcnn_r50_full_<stamp>`
  First attempt (runs/maskrcnn_r50_full_20261113_190514) DIED 2026-11-13 19:28 in epoch 1 (unclean reboot, Kernel-Power 41),
  no checkpoint. RESTARTED from scratch 2026-11-14 22:35 via WMI: runs/maskrcnn_r50_full_20261114_223544 (~4.5 h, done ~03:00).
  Resume: `.venv\Scripts\python -m src.train --config configs/full.yaml --resume runs/maskrcnn_r50_full_20261114_223544`
  Full detector FINISHED 2026-11-15 03:38 (24 epochs; use last.pt). Its val numbers (default 0.3828 / tuned 0.4226) are NOT valid.
  Full refiner started 2026-11-15 07:58 via WMI: runs/refiner_r18_full_20261115_075831, logs train_refiner_full_stdout.log /
  train_refiner_full_stderr.log (~1.5 h; use last.pt). Next: predict test -> submissions/sub_v5_full.csv with frozen v4 thresholds.
  That refiner run DIED 2026-11-15 ~09:07 in epoch 17 (unclean reboot, Kernel-Power 41). src/refiner.py has no resume and
  uses OneCycleLR, so its last.pt (epoch 16) is mid-schedule -> do NOT use. RESTARTED from scratch 2026-11-15 20:04 via WMI:
  runs/refiner_r18_full_20261115_200448 (same logs, appended; ~1.5 h, done ~21:30). Use its last.pt after "ep 24/24".
  Full refiner FINISHED 2026-11-15 21:27 (24 epochs). submissions/sub_v5_full.csv written 23:06 (full detector last.pt +
  full refiner last.pt, configs/refined_tta.yaml, frozen v4 thresholds): 1201 rows, 178/180 images, verified (ids, RLE 2048^2,
  no empty masks). NOT yet uploaded to Kaggle (user uploads). v5 has no valid val score; weights not in weights/, notebook
  reproduces v4. Remaining: upload v5 + choose final 2 picks, commit + push (awaiting user approval), report PDF, Google form.
  sub_v5_full.csv uploaded by user 2026-11-16: public LB 0.39 (v4 0.38) -> best so far. Kaggle clock showed ~10 h left at
  07:30 IST 2026-11-16 (deadline ~17:30 IST, later than the 10:30 IST assumed above).
- (history) Seed-7 run was paused by user 2026-11-11 11:11 after epoch 12 (ckpt.pt saved; epoch 13 partial lost). Resume with:
  `.venv\Scripts\python -m src.train --config configs/seed7.yaml --resume runs/maskrcnn_r50_seed7_20261111_073534`
  (launch via WMI Invoke-CimMethod as above). ~12 epochs left = ~1.8 h with the GPU free.
