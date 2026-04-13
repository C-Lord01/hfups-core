"""Fine-tune YOLOv8s on flood_v2 — 15-class disaster vocabulary (v2).

Class weights are applied to the BCE classification loss via a custom
DetectionTrainer subclass. The standard v8DetectionLoss computes:

    loss[1] = bce(pred_scores, target_scores).sum() / target_scores_sum

where bce has reduction='none' and output shape (batch, anchors, nc).
ClassWeightedBCE wraps that to multiply each class's per-element loss by
its weight before the caller sums, giving the desired weighted behaviour.

Usage:
    python tools/train_disaster_model.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_YAML = Path(
    "C:/Users/Chris/OneDrive/Documents/Hackerthons/HFUPS/Datasets/flood_v2/data.yaml"
)
MODELS_DIR = Path("C:/Users/Chris/OneDrive/Documents/Hackerthons/HFUPS/HFUPS Repo/models")
OUTPUT_WEIGHTS = MODELS_DIR / "yolov8s_disaster_v2.pt"

import torch as _torch
_DEVICE = 0 if _torch.cuda.is_available() else "cpu"
del _torch

TRAIN_ARGS = dict(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=16 if _DEVICE != "cpu" else 8,
    device=_DEVICE,
    project="disaster_detection",
    name="v2",
    exist_ok=False,
    verbose=True,
)

# 15-class vocabulary (class 11 utility_pole and 16 rescue_boat dropped;
# class 12 carpark kept at original index for label file compatibility)
CLASS_NAMES = {
    0:  "flood",
    1:  "flooded_area",
    2:  "flooded_road",
    3:  "flooded_bridge",
    4:  "flooded_carpark",
    5:  "submerged_car",
    6:  "car",
    7:  "person",
    8:  "person_on_vehicle",
    9:  "person_in_water",
    10: "residential_building",
    12: "carpark",
    13: "ocean",
    14: "waves",
    15: "debris_floating",
}

# Classes with zero annotations (will not be trained — flag as NOT_TRAINED in report)
MISSING_CLASSES = {1, 5, 9, 12, 13, 14, 15}

CLASS_WEIGHTS = {
    0:  2.0,  # flood           — 1,040 annotations
    1:  4.0,  # flooded_area    — MISSING
    2:  1.0,  # flooded_road    — 12,371 ok
    3:  1.5,  # flooded_bridge  — 2,601 ok
    4:  3.0,  # flooded_carpark — 17 SPARSE
    5:  4.0,  # submerged_car   — MISSING
    6:  1.0,  # car             — 9,780 ok
    7:  1.0,  # person          — 24,549 ok
    8:  1.5,  # person_on_vehicle — 1,881 ok
    9:  4.0,  # person_in_water — MISSING
    10: 1.0,  # residential_building — 14,038 ok
    12: 4.0,  # carpark         — MISSING
    13: 4.0,  # ocean           — MISSING
    14: 4.0,  # waves           — MISSING
    15: 4.0,  # debris_floating — MISSING
}

# ---------------------------------------------------------------------------
# Weighted BCE wrapper
# ---------------------------------------------------------------------------

class ClassWeightedBCE:
    """Drop-in replacement for BCEWithLogitsLoss(reduction='none').

    Multiplies each class's per-element loss by a scalar weight before
    returning, so the caller's .sum() accumulates the weighted total.

    Args:
        class_weights: 1-D tensor of shape (nc,). Unweighted classes = 1.0.
    """

    def __init__(self, class_weights: torch.Tensor) -> None:
        self._bce = nn.BCEWithLogitsLoss(reduction="none")
        self.class_weights = class_weights  # shape (nc,)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred / target: (batch, anchors, nc)
        loss = self._bce(pred, target)
        # Broadcast (nc,) across batch and anchor dims
        return loss * self.class_weights.to(pred.device)


# ---------------------------------------------------------------------------
# Picklable criterion factory
# ---------------------------------------------------------------------------

class _WeightedCriterionFactory:
    """Callable assigned to model.init_criterion.

    Defined as a top-level class (not a local function or lambda) so that
    torch.save() can pickle it without error.  save_model() strips it from
    the model's __dict__ before saving and restores it after.
    """

    def __init__(self, model, class_weights_map: dict[int, float]) -> None:
        self._model = model
        self._class_weights_map = class_weights_map

    def __call__(self):
        criterion = v8DetectionLoss(self._model)
        nc = self._model.model[-1].nc
        w = torch.ones(nc, dtype=torch.float32)
        for cls_id, wt in self._class_weights_map.items():
            if cls_id < nc:
                w[cls_id] = wt
        criterion.bce = ClassWeightedBCE(w)
        weighted_ids = [(i, float(wt)) for i, wt in enumerate(w.tolist()) if wt != 1.0]
        print(
            "[WeightedTrainer] criterion.bce patched — class weights: "
            + ", ".join(f"{CLASS_NAMES.get(i, str(i))}({i})={wt:.1f}" for i, wt in weighted_ids)
        )
        return criterion


# ---------------------------------------------------------------------------
# Custom trainer
# ---------------------------------------------------------------------------

class WeightedDetectionTrainer(DetectionTrainer):
    """DetectionTrainer with per-class classification loss weighting.

    Patches model.init_criterion() on the model instance returned by
    get_model() so that when BaseModel.loss() lazily initialises the
    criterion on the first training batch, it gets a v8DetectionLoss whose
    .bce attribute is a ClassWeightedBCE instead of plain BCEWithLogitsLoss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights_map: dict[int, float] = CLASS_WEIGHTS

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        model.init_criterion = _WeightedCriterionFactory(model, self._class_weights_map)
        return model

    def save_model(self) -> None:
        """Save checkpoint after temporarily removing the patched init_criterion.

        _WeightedCriterionFactory holds a reference to the live model, which
        makes the checkpoint larger and can cause pickling issues. Strip it
        before torch.save() and restore immediately after.
        """
        patched = self.model.__dict__.pop("init_criterion", None)
        try:
            super().save_model()
        finally:
            if patched is not None:
                self.model.init_criterion = patched


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train() -> Path:
    """Run training and return path to best.pt."""
    model = YOLO("yolov8s.pt")

    print("Starting fine-tune on flood_v2 (15-class vocabulary)...")
    print(f"  Data:    {DATA_YAML}")
    print(f"  Epochs:  {TRAIN_ARGS['epochs']}")
    print(f"  Imgsz:   {TRAIN_ARGS['imgsz']}")
    print(f"  Batch:   {TRAIN_ARGS['batch']}")
    print(f"  Device:  {TRAIN_ARGS['device']}")
    print(f"  Output:  {OUTPUT_WEIGHTS}")
    print(f"  Weights: {CLASS_WEIGHTS}")

    model.train(
        trainer=WeightedDetectionTrainer,
        **TRAIN_ARGS,
    )

    best_pt = Path("disaster_detection/v2/weights/best.pt")
    if not best_pt.exists():
        raise FileNotFoundError(f"Expected best.pt at {best_pt.resolve()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, OUTPUT_WEIGHTS)
    print(f"\nBest weights copied to: {OUTPUT_WEIGHTS}")
    return OUTPUT_WEIGHTS


# ---------------------------------------------------------------------------
# Validation entry point
# ---------------------------------------------------------------------------

def validate(weights_path: Path) -> None:
    """Run validation on the test split and report per-class mAP50."""
    print("\n--- Validation on test split ---")
    model = YOLO(str(weights_path))

    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=TRAIN_ARGS["imgsz"],
        device=TRAIN_ARGS["device"],
        verbose=False,
    )

    map50_overall = float(results.box.map50)

    # Ultralytics stores per-class AP50 in results.box.ap50
    ap50_per_class = results.box.ap50  # shape (nc,) or list

    print(f"\nmAP50 overall: {map50_overall:.4f}")
    print(f"\n{'ID':<5} {'Class':<24} {'mAP50':>7}  Status")
    print("-" * 48)

    flag_threshold = 0.4
    weak: list[str] = []
    not_trained: list[str] = []

    for cls_id in sorted(CLASS_NAMES.keys()):
        cls_name = CLASS_NAMES[cls_id]
        if cls_id in MISSING_CLASSES:
            print(f"  {cls_id:<4} {cls_name:<24} {'—':>7}  NOT_TRAINED")
            not_trained.append(cls_name)
            continue
        try:
            ap = float(ap50_per_class[cls_id])
        except (IndexError, TypeError):
            ap = float("nan")
        flag = "WEAK" if ap < flag_threshold else "ok"
        print(f"  {cls_id:<4} {cls_name:<24} {ap:>7.4f}  {flag}")
        if ap < flag_threshold:
            weak.append(cls_name)

    print()
    if weak:
        print(f"WEAK classes (mAP50 < {flag_threshold}): {weak}")
    else:
        print(f"All trained classes above mAP50 {flag_threshold} threshold.")
    if not_trained:
        print(f"NOT_TRAINED (no annotations): {not_trained}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    best_weights = train()
    validate(best_weights)
