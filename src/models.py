"""Model definitions."""
from __future__ import annotations

import torch
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

NUM_CLASSES = 2  # background + filament (class-agnostic, per competition rules)


def build_model(mcfg: dict, pretrained: bool | None = None) -> torch.nn.Module:
    if mcfg["arch"] != "maskrcnn_resnet50_fpn":
        raise ValueError(f"unknown arch {mcfg['arch']}")
    use_pre = mcfg.get("pretrained", True) if pretrained is None else pretrained
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if use_pre else None,
        weights_backbone=None,
        trainable_backbone_layers=mcfg.get("trainable_backbone_layers", 3),
        min_size=mcfg["train_size"],
        max_size=mcfg["train_size"],
        box_detections_per_img=mcfg.get("box_detections_per_img", 60),
    )
    # torchvision sets FrozenBatchNorm eps=0 when loading COCO weights; do the same when
    # building for inference so train and inference models are numerically identical.
    overwrite_eps(model, 0.0)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, NUM_CLASSES)
    return model


def set_mode(model: torch.nn.Module, mcfg: dict, train: bool) -> None:
    """Switch input size and detection threshold between training and full-res inference."""
    size = mcfg["train_size"] if train else mcfg["infer_size"]
    model.transform.min_size = (size,)
    model.transform.max_size = size
    model.roi_heads.score_thresh = 0.05 if train else mcfg.get("infer_score_thresh", 0.1)
    model.train(train)


def save_weights(model: torch.nn.Module, path) -> None:
    """fp16 weights (~half size, fits GitHub's 100 MB limit); BatchNorm stats kept fp32
    because tiny running_var values underflow to 0 in fp16 (with eps=0 -> inf/NaN)."""
    sd = {}
    for k, v in model.state_dict().items():
        keep32 = k.endswith(("running_var", "running_mean")) or not v.is_floating_point()
        sd[k] = v if keep32 else v.half()
    torch.save(sd, path)


def load_weights(model: torch.nn.Module, path) -> None:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in sd.items()})
