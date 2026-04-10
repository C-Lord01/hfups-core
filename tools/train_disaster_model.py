"""Fine-tune YOLOv8s on the merged disaster dataset.

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
    "C:/Users/Chris/OneDrive/Documents/Hackerthons/HFUPS/Datasets/merged/data.yaml"
)
MODELS_DIR = Path("C:/Users/Chris/OneDrive/Documents/Hackerthons/HFUPS/HFUPS Repo/models")
OUTPUT_WEIGHTS = MODELS_DIR / "yolov8s_disaster.pt"

TRAIN_ARGS = dict(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project="disaster_detection",
    name="v1",
    exist_ok=False,
    verbose=True,
)

# Per-class weights: index = class id
# 0:flood, 1:flooded_area, 2:car, 3:damaged_building, 4:debris,
# 5:fire, 6:smoke, 7:accident_vehicle, 8:person
CLASS_NAMES = [
    "flood", "flooded_area", "car", "damaged_building", "debris",
    "fire", "smoke", "accident_vehicle", "person",
]
CLASS_WEIGHTS = {
    0: 3.0,  # flood      — thin class
    1: 3.0,  # flooded_area — thin class
    4: 3.0,  # debris      — thinnest class
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
            + ", ".join(f"{CLASS_NAMES[i]}({i})={wt:.1f}" for i, wt in weighted_ids)
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

    print("Starting fine-tune on merged disaster dataset...")
    print(f"  Data:    {DATA_YAML}")
    print(f"  Epochs:  {TRAIN_ARGS['epochs']}")
    print(f"  Imgsz:   {TRAIN_ARGS['imgsz']}")
    print(f"  Batch:   {TRAIN_ARGS['batch']}")
    print(f"  Device:  {TRAIN_ARGS['device']}")
    print(f"  Weights: {CLASS_WEIGHTS}")

    model.train(
        trainer=WeightedDetectionTrainer,
        **TRAIN_ARGS,
    )

    best_pt = Path("disaster_detection/v1/weights/best.pt")
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
    maps_per_class = results.box.maps  # list of per-class mAP50-95; use ap_class_index

    # Ultralytics stores per-class AP50 in results.box.ap50
    ap50_per_class = results.box.ap50  # shape (nc,) or list

    print(f"\nmAP50 overall: {map50_overall:.4f}")
    print(f"\nPer-class mAP50:")
    flag_threshold = 0.4
    flagged: list[str] = []
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        try:
            ap = float(ap50_per_class[cls_id])
        except (IndexError, TypeError):
            ap = float("nan")
        flag = "  <-- BELOW 0.4" if ap < flag_threshold else ""
        print(f"  {cls_id}  {cls_name:<20}: {ap:.4f}{flag}")
        if ap < flag_threshold:
            flagged.append(cls_name)

    if flagged:
        print(f"\nFlagged classes (mAP50 < {flag_threshold}): {flagged}")
    else:
        print(f"\nAll classes above mAP50 {flag_threshold} threshold.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    best_weights = train()
    validate(best_weights)
