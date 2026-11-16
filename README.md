# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha full-disk images
([Kaggle competition](https://www.kaggle.com/competitions/filament-segmentation-2026/), NSF NSO).
One mask per filament, scored by Panoptic Quality (PQ).

**Method:** two stages. (1) A class-agnostic Mask R-CNN (ResNet50-FPN, COCO-pretrained, torchvision),
fine-tuned on full-resolution 1024x1024 crops and run on the native 2048x2048 image with flip TTA,
followed by greedy score-ordered overlap removal with thresholds tuned on a held-out validation
split. (2) A small U-Net (ResNet18 encoder) that re-segments every detected filament on a
native-resolution crop; refined masks that touch are merged.

| | Validation PQ | Public LB |
|---|---|---|
| Epoch 4 checkpoint | 0.375 | 0.33 |
| Epoch 24 | 0.414 | 0.36 |
| Epoch 24 + 3-view flip TTA | 0.419 | 0.37 |
| **Final: + crop refiner (stage 2)** | **0.447** | **0.38** |
| Inter-annotator agreement (reference) | ~0.34 | – |

## Quick start (reproduce the submission without retraining)

```bash
git clone https://github.com/Saggy7276/filament-seg-2026.git && cd filament-seg-2026
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
# put the competition data in MAGFiLO_1.0_Kaggle_2026/ (train/ and test/ as distributed)
jupyter notebook notebooks/pipeline.ipynb               # run all cells
```

The notebook loads `weights/maskrcnn_r50_filament.pt`, the tuned post-processing in
`weights/maskrcnn_r50_filament_tta.pp.json` and the refiner `weights/refiner_r18_filament.pt`,
reports validation metrics and figures, and writes `submissions/submission.csv`.
Equivalent command line:

```bash
python -m src.predict --config configs/refined_tta.yaml --weights weights/maskrcnn_r50_filament.pt \
    --pp weights/maskrcnn_r50_filament_tta.pp.json --refiner weights/refiner_r18_filament.pt \
    --submit --out submissions/submission.csv
```

Retrain (GTX 1650 4 GB: detector about 3.7 h from COCO weights, refiner about 1.2 h from ImageNet weights):

```bash
python -m src.train --config configs/baseline.yaml
python -m src.train --config configs/baseline.yaml --resume runs/<run_dir>   # continue an interrupted run
python -m src.refiner --config configs/refiner.yaml
python tools/eval_refiner.py --refiner runs/<refiner_run>/best.pt            # validation study of stage 2
```

## Pipeline

1. **Data** (`src/data.py`). Images are loaded as single-channel 8-bit grayscale. Only the
   `segmentation` polygons are used as targets; spines, boxes and chirality classes are not used,
   following the competition rules, so all filaments form a single class. Images labelled by
   several annotators give several training samples.
2. **Split.** 15% of the unique images (106) are held out for validation, grouped by image
   timestamp so the same image never appears in both sets, and stratified by observatory.
   The split is stored in `splits/split_seed42.json` and used by every experiment.
3. **Training** (`src/train.py`). Random 1024x1024 crops at native resolution (80% centred near a
   filament), horizontal/vertical flips, 90-degree rotations, brightness/contrast/gamma jitter.
   SGD (lr 0.005, momentum 0.9, weight decay 1e-4), effective batch 4 (batch 1 x 4 accumulation),
   linear warm-up then cosine decay, 24 epochs, fp32. Validation PQ every 4 epochs; best kept.
4. **Inference** (`src/predict.py`). Full 2048x2048 image, about 1 s/image on a GTX 1650.
5. **Post-processing** (`src/postprocess.py`). Detections are painted in descending score order;
   each pixel belongs to at most one filament, and a detection is dropped if less than 50% of it
   survives (mask-level NMS). Score threshold, mask threshold and minimum area are grid-searched
   on the validation split only.
6. **Crop refiner** (`src/refiner.py`, `configs/refiner.yaml`). Mask R-CNN predicts each mask on a
   28x28 grid inside its box, which is coarse for thin filaments. Each instance from step 5 is
   re-segmented by a ResNet18 U-Net on a square native-resolution crop centred on it (side = 1.8 x
   the larger box side, at least 96 px, resized to 256x256, four flip views averaged, threshold
   0.4). Refined masks that touch are merged into one filament; an instance keeps its detector mask
   if the refiner returns nothing. The refiner is trained on the training split only, on one crop
   per ground-truth filament with random context (1.4-2.3) and centre shift to imitate imperfect
   detector boxes (BCE + Dice loss, AdamW, 24 epochs). On validation it raises PQ from 0.419 to
   0.447 (TP 706 -> 727, FP 415 -> 368).
7. **Submission.** One row per filament, `filament_id = <image_id>_<k>`, `segmentation_rle` =
   pycocotools RLE counts string. `validate_submission` checks ids, format, and that every RLE
   decodes to a non-empty 2048x2048 mask.

## Evaluation (`src/metrics.py`)

* **PQ** = sum IoU(TP) / (|TP| + 0.5|FP| + 0.5|FN|), match when IoU > 0.5 (own implementation; pycocotools
  COCOeval computes AP, not PQ).
* **Dice**: binary Dice of all predicted vs all GT filament pixels per image; also computed with
  `torchmetrics.segmentation.DiceScore` in the notebook (identical values).
* **IoU / Dice distributions** per GT filament, and **one-to-many** (a GT covered by >= 2 predictions
  each > 50% inside it) and **many-to-one** (a prediction covering >= 2 GTs each > 50% inside it) counts.

## Repository layout

```
configs/                  all hyperparameters (refined_tta.yaml = final pipeline, refiner.yaml = stage-2 training)
splits/split_seed42.json  fixed train/val split
src/                      data, models, train, refiner, predict, postprocess, tta, metrics, visualize, utils
notebooks/pipeline.ipynb  end-to-end reproduction (generated by tools/build_notebook.py)
weights/                  final weights (fp16, BatchNorm stats fp32) + tuned post-processing
results/                  results.csv (every run), submissions_log.csv
tests/test_pipeline.py    metric / post-processing / RLE tests
tools/                    notebook builder, notebook smoke test, TTA / ensemble / refiner evaluation
```

## Reproducibility notes

* Seeds are fixed (`seed: 42`, re-seeded every epoch so resumed runs match) and logged per run.
* Weights are stored in fp16 to stay under GitHub's 100 MB file limit; BatchNorm running statistics
  are kept in fp32 because torchvision uses `eps=0` for frozen BatchNorm with COCO weights, and tiny
  variances underflow in fp16.
* Mixed precision is disabled: on the GTX 1650 it produced NaN mask losses and was slower than fp32.

## Acknowledgement

The two-stage design (detector followed by a crop-level U-Net refiner, merging masks that overlap)
follows the public Kaggle notebooks "Mask R-CNN + U-Net Refiner" by Nomannic19 and
"LB 0.70 | Topology-Safe Solar Filaments" by Fususu. The code in this repository was written
independently and all weights were trained by us on the competition training split; no
checkpoints from those notebooks are used.

## License / data

Competition data are used for non-commercial research purposes under the competition rules.
