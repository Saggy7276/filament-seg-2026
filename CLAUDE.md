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
- (history) Seed-7 run was paused by user 2026-11-11 11:11 after epoch 12 (ckpt.pt saved; epoch 13 partial lost). Resume with:
  `.venv\Scripts\python -m src.train --config configs/seed7.yaml --resume runs/maskrcnn_r50_seed7_20261111_073534`
  (launch via WMI Invoke-CimMethod as above). ~12 epochs left = ~1.8 h with the GPU free.
