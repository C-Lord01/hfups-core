"""BLIP-based scene captioner for disaster-response image understanding."""

from __future__ import annotations


class SceneCaptioner:
    MODEL_ID = "Salesforce/blip-image-captioning-base"

    def __init__(self, device: str = "cpu"):
        # lazy load — do not import transformers at module level
        # load processor and model on first call to caption()
        self._processor = None
        self._model = None
        self.device = device

    def _load(self):
        from transformers import BlipForConditionalGeneration, BlipProcessor
        import torch  # noqa: F401 — needed for .to() and device placement
        self._processor = BlipProcessor.from_pretrained(self.MODEL_ID)
        self._model = BlipForConditionalGeneration.from_pretrained(self.MODEL_ID)
        self._model.to(self.device)
        self._model.eval()

    def caption(self, image_path: str) -> str:
        """Return a single caption string for the image."""
        if self._model is None:
            self._load()
        from PIL import Image
        import torch
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=50)
        return self._processor.decode(output[0], skip_special_tokens=True)
