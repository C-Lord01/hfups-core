# HFUPS — Project Continuity Update
**Date:** April 3, 2026  
**Status:** Active Passion Project — Integration Phase  
**Author:** Chris 'C-Lord' Kulpa

---

## 1. Session Summary (April 3, 2026)

This session resolved three architectural issues with the prompt reconstruction 
pipeline and integrated the HuggingFace/FLUX.1 backend to replace Nova Canvas 
(which hit legacy status March 30, 2026).

### What was done

**Issue 1 — Object detection ceiling**
- Per-class deduplication cap raised from 2 to 4 instances in `keyframe_builder.py`
- `KeyframeBuilder` now explicit `max_objects=12` in `demo/run.py`
- Default confidence threshold lowered from 0.15 to 0.10 to improve recall on 
  partially occluded objects (common in disaster imagery)

**Issue 2 — Grid coordinates poisoning image generation**
- All prompt templates updated to remove `(grid cx,cy)` coordinate output
- All confidence percentage strings removed from prompt output
- `grid_to_bucket_phrase` rewritten to return natural language spatial descriptions:
  e.g. "mid center of the frame", "upper left side of the frame"
- URGENT prefix removed from disaster_response template (belongs in notification 
  layer, not image generation prompt)

**Issue 3 — UPS prompt reconstruction**
- New `_ups_template` added to `nova/prompt_templates.py` with all four UPS layers:
  - A: Realism booster ("Ultra-realistic documentary photograph...")
  - B: Subject/scene from detected objects with natural spatial language
  - C: Camera and lighting ("full-frame DSLR, 24-70mm lens, f/4...")
  - D: Imperfections ("Subtle motion blur, atmospheric haze...")
- `ups` is now the default template. `disaster_response` retained with deprecation warning.

**New feature — FLUX.1 / HuggingFace backend**
- `nova/hf_client.py` added: `invoke_hf_image(prompt, token, model, width, height)`
- Model: `black-forest-labs/FLUX.1-schnell` (default)
- Reads `HF_API_TOKEN` from environment, fails loudly if missing
- `demo/run.py` updated: `--backend huggingface` (default) / `bedrock`
- `--hf-model` flag for model override
- `--caption` flag for manual operator scene description (bypasses BLIP entirely)
- `requests>=2.28` added to default dependencies

**Bug fixes**
- `yolov8m.pt` → `yolov8n.pt` in `demo/run.py` (wrong model was being downloaded)
- BLIP/SceneCaptioner removed from hot path entirely; `scene_captioner.py` retained 
  for future optional use

**Test results:** 113 passed, 1 xfailed, 0 failed

### Verified prompt output (image 1.jpg, 3 objects detected)
```
Ultra-realistic documentary photograph, photojournalism style, natural color 
science, high dynamic range, no CGI. Scene showing: a medium person in the lower 
left side of the frame, a medium car in the mid right side of the frame, a small 
car in the upper left side of the frame. Shot on a full-frame DSLR, 24-70mm lens, 
f/4, natural available light, wide establishing shot. Subtle motion blur, 
atmospheric haze, dust particles, realistic shadows, minor lens distortion.
```

No grid coordinates. No confidence percentages. No URGENT prefix. Reads like a 
photography brief.

---

## 2. Current Architecture

### A. Edge — Transmitter
- Image → YOLOv8n detection (conf=0.10) → class mapping via Open Images V7
- Objects ranked, deduplicated (up to 4 per class), capped at 12
- Positions quantized to 8×8 grid (internal only, not exposed in prompts)
- Confidence quantized 0–15 (internal only)
- Packed into deterministic bit-level `KeyframePacket`
- Worst case: 46 bytes (12 objects), best case: 5 bytes (1 object)
- Optional `DeltaPacket` for motion tracking between frames

### B. Transport Layer
Three modes implemented and tested:
- **Shim** — in-process for testing
- **TCP** — live streaming with optional ACK/resend
- **VARA HF Text Bridge** — chunked Base64 ASCII with CRC32 per chunk

### C. Receiver / Reconstruction
- Decodes semantic packet → object list with spatial positions
- UPS prompt reconstruction via `_ups_template` (4-layer structure)
- Prompt fed to FLUX.1-schnell via HuggingFace Inference API
- CLI driven, outputs saved to `outputs/` folder

---

## 3. Open Issues / Known Constraints

### Object count on sparse scenes
Image 1.jpg (overturned truck on highway at night) only produced 3 detected objects.
YOLOv8n nano has limited recall in low-light and fog conditions. This is expected.
The 0.10 confidence threshold is already at the practical floor for this model.

**Options not yet explored:**
- YOLOv8s (small) — next size up, ~22M params vs 3M, better recall
- Custom fine-tuned model on disaster imagery (noted as long-term goal)
- The BLIP/scene_captioner.py path is still available as an optional `--caption` 
  supplement if the operator wants to add context the detector missed

### FLUX.1-schnell quality ceiling
FLUX.1-schnell is the fast/free tier. FLUX.1-dev or FLUX.1-pro will produce 
significantly better photorealistic output. The `--hf-model` flag supports 
swapping the model without code changes:
```bash
python -m hfups.demo.run --image <image> --nova \
  --hf-model black-forest-labs/FLUX.1-dev
```
Note: FLUX.1-dev requires accepting terms at huggingface.co/black-forest-labs/FLUX.1-dev

### UPS Layer C is static
Camera and lighting in Layer C is currently hardcoded:
"Shot on a full-frame DSLR, 24-70mm lens, f/4, natural available light, wide 
establishing shot."

This should eventually be derived from scene context (night → high ISO, indoor → 
artificial light, etc.). Not a blocker for demo but worth noting.

### UPS Layer B caps at 6 objects for prompt
`_ups_template` uses `items[:6]` to avoid token overflow. With 12 objects encoded 
in the packet, 6 are dropped from the prompt. This is intentional for now — 
diffusion models don't handle 12-item lists well. May revisit with structured 
prompt injection.

---

## 4. Immediate Next Session

- [ ] Set `HF_API_TOKEN` if not already permanent in User environment scope
- [ ] Accept FLUX.1-schnell terms at huggingface.co/black-forest-labs/FLUX.1-schnell
- [ ] Run first live end-to-end: 
```bash
  python -m hfups.demo.run --image data/demo/1.jpg --nova --backend huggingface
```
- [ ] Evaluate `recon.png` quality
- [ ] Run against all 10 demo images, collect `summary.json` outputs
- [ ] Compare reconstruction quality across scene types
- [ ] Decide whether to upgrade to YOLOv8s for demo images

---

## 5. Short Term (This Week)

- [ ] Record 60–90 second demo video
  - Show input image (MB scale)
  - Show 5–46 byte packet
  - Show airtime calculation
  - Show FLUX.1 reconstruction side by side
- [ ] Write arXiv white paper draft
- [ ] Open source release — push to GitHub public, post to r/amateurradio
- [ ] Get General class ham radio license (HamStudy.org)

---

## 6. Long Term

- [ ] Real two-station HF demo (requires license)
- [ ] ARES/RACES community outreach
- [ ] NTT / NICT contact via white paper
- [ ] Custom disaster imagery fine-tuned detection model
- [ ] Build Pi + (tr)uSDX demo unit

---

## 7. Repository State

**GitHub:** https://github.com/C-Lord01/hfups-core  
**Branch:** main  
**Test count:** 113 passed, 1 xfailed  
**Python:** 3.14  
**Entry points:** `hfups-demo`, `hfups-rx`, `hfups-tx`, `hfups-demo-run`

### Key File Map
```
src/hfups/
  vision/           # YOLO adapter, keyframe builder, class mapping, tracker
                    # scene_captioner.py retained but not in hot path
  transport/        # VARA text bridge, TCP, serial, shim
  nova/             # UPS prompt templates (ups default), HF client, bedrock client
  demo/             # Full pipeline CLI runner (--backend huggingface default)
  integrations/     # TX/RX keyframe wrappers
  packets.py        # Legacy IFrame, MFPacket, ClipParams bit packing
  framing.py        # COBS framing
  crc.py            # CRC helpers
  streaming.py      # Frame stream decoder
tests/              # 113 tests
tools/              # OpenImages dict builder
data/
  demo/             # 10 disaster scenario test images (1.jpg–10.jpg)
  mappings/         # YOLO→OpenImages class maps
  openimages/       # Class descriptions CSV
models/
  yolov8n.pt        # YOLO nano model (confirmed correct)
```

---

## 8. Core Message (Unchanged)

HFUPS demonstrates that:
- HF radio can carry situational awareness, not just text
- Semantic compression enables image reconstruction over ultra-low bandwidth
- Generative AI can reconstruct scenes from structured meaning

**This is not a toy wrapper — it is a new transmission paradigm.**
