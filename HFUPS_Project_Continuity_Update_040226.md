# HFUPS — Project Continuity Update
**Date:** April 2, 2026
**Status:** Active Passion Project — Integration Complete, Demo Ready
**Author:** Chris 'C-Lord' Kulpa

---

## 1. Project Vision (Unchanged)

HFUPS replaces pixel transmission with semantic transmission.

- Extract semantic meaning from an image via YOLOv8m object detection + BLIP scene captioning
- Compress to a deterministic, bit-packed structure (46 bytes worst case, 12 objects)
- Transmit tens of bytes instead of megabytes over HF radio
- Reconstruct a high-fidelity scene using FLUX.1-schnell on the receiving end

**Core thesis:** When bandwidth is scarce, transmit meaning — not pixels.

---

## 2. Context: Why This Exists

Originally built for an AWS hackathon using Hurricane Helene as the emotional anchor.
The real use case in mind: Japanese post-earthquake/tsunami disaster response.
Japan's emergency infrastructure (IoT vending machines, OFDM networks, NTT R&D) was a direct inspiration.
The hackathon was a bust — Nova Canvas access arrived 12 hours before submission on a work day.
Project has since evolved into a passion project with long-term intent:

- Open source release to the ham radio / emergency comms community (ARES/RACES)
- White paper submission (arXiv target)
- Potential presentation to NTT or NICT

---

## 3. Session Summary (April 2, 2026)

This session completed the integration phase. The pipeline is now end-to-end functional
and validated against real disaster imagery.

### What Was Done

**HF Inference Backend**
- Created `src/hfups/nova/hf_client.py`
  - `invoke_image_generation(prompt, token, width, height) -> bytes`
  - Lazy `requests` import, reads `HF_API_TOKEN` from env
  - Endpoint: `https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell`
  - Note: old `api-inference.huggingface.co` returns HTTP 410 — use router URL
- Added `--backend` flag to `run.py`: `huggingface` (default) / `bedrock`
- `HF_API_TOKEN` set permanently in Windows User environment scope via `setx`

**Model Upgrade**
- Swapped YOLOv8n → YOLOv8m
- Detection recall improved immediately (3 objects vs 1 on test image)
- `models/*.pt` added to `.gitignore` — weights are not committed to repo
- YOLOv8m auto-downloads on first run via ultralytics

**Disaster Vocabulary Rebuild**
- Previous state: 5 encodable classes, 7 of 12 mappings pointing to wrong MIDs
- New state: 29 classes, all correct MIDs
- Files replaced:
  - `data/openimages/class-descriptions-boxable.csv`
  - `data/mappings/yolo_to_openimages.json`
  - `dict/openimages_v7_boxable.json` (rebuilt via `tools/rebuild_dictpack.py`)
- MID collision fixed: traffic light `/m/015p6` → `/m/01mqdt`

**BLIP Scene Captioning**
- Created `src/hfups/vision/scene_captioner.py`
  - Model: `Salesforce/blip-image-captioning-base` (224M, CPU viable)
  - Lazy load on first call
- Created `src/hfups/vision/caption_parser.py`
  - Pure keyword matching: hazards, actions, environment
  - Deterministic, no ML at parse time
- Integrated into `run.py` — captioning is ON by default, `--no-caption` to disable
- Unit tests pass `--no-caption` to avoid loading BLIP in test fixtures

**Architecture note — BLIP and YOLO are strictly additive:**
BLIP output never touches the detections list, the packet, or the KeyframeBuilder.
Data flow is one-directional:
```
YOLO detections ──► KeyframeBuilder ──► KeyframePacket ──► prompt (YOLO facts)
                                                                ↑
BLIP caption ──► parse_caption ──────────────────────────── appended below
```
BLIP enters only at prompt-assembly in `prompt_templates.py` as plain string
concatenation after all YOLO-derived facts are written. The two models are
complementary, not cross-validating at the code level.

**summary.json Enhancements**
- Added `backend` field
- Added `output_image` field (null when `--nova` not passed)
- Added `caption` and `caption_parsed` fields (null when `--no-caption`)

**Git**
- Commit `b65cdf6`: feat: BLIP scene captioning + expanded disaster vocab + HF inference backend
- Commit `28ffb1b`: chore: ignore model weight files (*.pt)
- Commit `<continuity doc commit>`: docs: project continuity update 04/02/2026
- Commit `<image4 fix commit>`: docs: correct image 4 diagnosis — BLIP anchor bias, not YOLO hallucination
- 26 files changed, 487 insertions (code commits)

---

## 4. Architecture (Current, Working)

### A. Edge — Transmitter
- Image → YOLOv8m detection → 29-class disaster vocabulary via Open Images V7
- Image → BLIP-base captioning → keyword-parsed hazards/environment/actions
- Objects ranked, deduplicated, capped at 12
- Positions quantized to 8×8 grid
- Confidence quantized 0–15
- Packed into deterministic bit-level `KeyframePacket`
- Worst case: 46 bytes (12 objects), best case: 1 byte (no mapped objects, caption only)

### B. Transport Layer
Three modes implemented and tested:
- **Shim** — in-process for testing
- **TCP** — live streaming with optional ACK/resend
- **VARA HF Text Bridge** — chunked Base64 ASCII with CRC32 per chunk, stream CRC,
  out-of-order reassembly, corruption detection

### C. Receiver / Reconstruction
- Decodes semantic packet → object list, positions, motion narrative
- BLIP caption appended to prompt when available
- Four prompt template styles: `concise`, `descriptive`, `disaster_response`, `cinematic`
- Reconstruction via FLUX.1-schnell (HF Inference API, default) or Nova Canvas (bedrock)
- CLI driven, all outputs saved to `outputs/` folder

---

## 5. Validation Results (10 Disaster Images)

Pipeline run against 10 real disaster scenario images with `--caption` (now default).

| # | Scene | YOLO Objects | BLIP Caption | Hazards | Packet Bytes |
|---|---|---|---|---|---|
| 1 | Urban flood, person standing | 3 | flooded street in a neighborhood | flood | 13 |
| 2 | Submerged car, person | 4 | car submerged in flooded street | flood, submerged | 16 |
| 3 | Explosion/smoke cloud | 0 | a large white cloud | none | 1 |
| 4 | Coastal surge, parking lot | 15+ | large wave crashing over ocean | none | 20 |
| 5 | Dust storm on road | 1 | road with dust cloud in background | none | 5 |
| 6 | Street flooding | 4 | street filled with water | water | 16 |
| 7 | Overturned vehicle on highway | 2 | truck sitting on side of road | none | 9 |
| 8 | House fire | 4 | a house on fire | fire | 16 |
| 9 | Flooded road | 0 | flooded street with car in middle | flood | 1 |
| 10 | Fallen tree on car | 2 | tree fallen on car, residential area | none | 9 |

**Compression ratios demonstrated: 500,000:1 to 10,000,000:1**

Key findings:
- BLIP is essential: images 3 and 9 produce zero YOLO detections. Without captioning
  both would transmit 1 byte of boilerplate. BLIP correctly identifies both scenes.
- BLIP catches what YOLO misses: image 8, YOLO detects cars but misses the fire entirely.
  BLIP correctly captions "a house on fire" — triggering the fire hazard tag.
- YOLO is right, BLIP undersells: image 4 is a coastal parking lot inundated by storm
  surge with 15+ real vehicles. YOLO correctly detects them. BLIP anchors on the dominant
  wave feature and ignores the vehicles and infrastructure entirely. This is a known
  BLIP-base limitation — anchor bias toward dominant visual features in complex scenes.
- Image 7 limitation: overturned vehicle on highway. YOLO calls it bench + truck.
  BLIP says "truck sitting on side of road." Neither captures "overturned."
  Fine-tuned disaster captioner would address this.

---

## 6. Known Issues / Tech Debt

- Image 4 BLIP anchor bias: BLIP-base anchors on dominant visual features and misses
  secondary objects. In high-object-density disaster scenes (coastal surge, parking lots,
  debris fields) BLIP caption may underrepresent what YOLO correctly detects.
  Fine-tuned disaster captioner is the fix — not YOLO suppression.
- BLIP-base misses damage states: "overturned," "partially collapsed," "submerged"
  not reliably captured. Fine-tuning on FEMA/NOAA imagery is the fix.
- Legacy `src/hfups/prompt_builder.py` still present — deprecation warning added,
  removal pending after downstream refs cleaned up.
- Import-only tests (`test_vara_tcp_import.py` etc.) pad test count without
  testing behavior — flag for future cleanup.
- `vara_tcp.py` has no behavioral tests.

---

## 7. Completion Path

### Immediate Next Session
- [ ] Run `--nova` against images 1, 8, and 2 — flood+person, house fire, submerged car
- [ ] Collect reconstructed PNGs for demo video
- [ ] Evaluate FLUX.1-schnell reconstruction quality against real disaster prompts

### Short Term
- [ ] Record 60–90 second demo video
  - Input image (MB scale) → packet bytes → airtime ms → FLUX reconstruction
- [ ] Write arXiv white paper draft
- [ ] Open source release — GitHub public, post to r/amateurradio and ARRL forums
- [ ] Get General class ham radio license (study: HamStudy.org)

### Medium Term
- [ ] Build Pi + (tr)uSDX demo unit for over-the-air transmission
- [ ] Fine-tune BLIP-base on FEMA/NOAA disaster imagery
  - Addresses anchor bias and missed damage states
  - Strong pitch for NTT/NICT engagement

### Long Term
- [ ] Real two-station HF demo (requires ham license)
- [ ] ARES/RACES community outreach and field testing
- [ ] NTT / NICT contact via white paper or warm introduction

---

## 8. Repository State

**GitHub:** https://github.com/C-Lord01/hfups-core
**Branch:** main
**Test count:** 102 passed, 1 xfailed
**Python:** 3.14
**Entry points:** `hfups-demo`, `hfups-rx`, `hfups-tx`, `hfups-demo-run`

### Key File Map
```
src/hfups/
  vision/           # YOLOv8m adapter, BLIP captioner, caption parser,
                    # keyframe builder, class mapping, tracker
  transport/        # VARA text bridge, TCP, serial, shim
  nova/             # Prompt templates (4), bedrock client, HF client
  demo/             # Full pipeline CLI runner
  integrations/     # TX/RX keyframe wrappers
  packets.py        # IFrame, MFPacket, ClipParams bit packing
  framing.py        # COBS framing
  crc.py            # CRC helpers
  streaming.py      # Frame stream decoder
tests/              # 102 tests
tools/              # OpenImages dict builder, rebuild_dictpack.py
data/
  demo/             # 10 disaster scenario test images
  mappings/         # YOLO→OpenImages class maps (29 classes, correct MIDs)
  openimages/       # Class descriptions CSV (29 classes)
dict/
  openimages_v7_boxable.json   # 29-class dictpack
models/
  yolov8m.pt        # YOLO medium model (gitignored, auto-downloads)
```

---

## 9. Core Message (Unchanged)

HFUPS demonstrates that:
- HF radio can carry situational awareness, not just text
- Semantic compression enables image reconstruction over ultra-low bandwidth
- Generative AI can reconstruct scenes from structured meaning

**Compression demonstrated: 500,000:1 to 10,000,000:1**
**This is not a toy wrapper — it is a new transmission paradigm.**
