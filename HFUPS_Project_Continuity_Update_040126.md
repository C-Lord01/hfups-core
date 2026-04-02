# HFUPS — Project Continuity Update
**Date:** April 1, 2026  
**Status:** Active Passion Project — Integration Phase  
**Author:** Chris 'C-Lord' Kulpa

---

## 1. Project Vision (Unchanged)

HFUPS replaces pixel transmission with semantic transmission.

- Extract semantic meaning from an image via YOLOv8 object detection
- Compress to a deterministic, bit-packed structure (46 bytes worst case, 12 objects)
- Transmit tens of bytes instead of megabytes over HF radio
- Reconstruct a high-fidelity scene using a generative AI image model on the receiving end

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

## 3. Architecture (Current, Working)

### A. Edge — Transmitter
- Image → YOLOv8n detection → class mapping via Open Images V7
- Objects ranked, deduplicated, capped at 12
- Positions quantized to 8×8 grid
- Confidence quantized 0–15
- Packed into deterministic bit-level `KeyframePacket`
- Worst case: 46 bytes (12 objects), best case: 5 bytes (1 object)
- Optional `DeltaPacket` for motion tracking between frames

### B. Transport Layer
Three modes implemented and tested:
- **Shim** — in-process for testing
- **TCP** — live streaming with optional ACK/resend  
- **VARA HF Text Bridge** — chunked Base64 ASCII with CRC32 per chunk, stream CRC, out-of-order reassembly, corruption detection

### C. Receiver / Reconstruction
- Decodes semantic packet → object list, positions, motion narrative
- Four prompt template styles: `concise`, `descriptive`, `disaster_response`, `cinematic`
- Prompt fed to image generation model for visual reconstruction
- CLI driven, all outputs saved to `outputs/` folder

---

## 4. Work Completed This Session (April 1, 2026)

### Code Review Findings (Pre-Session)
Full senior developer review of repo identified:
- Duplicate prompt builders (legacy vs current)
- `cli_demo.py` using old prompt path
- No `.gitignore`
- `__pycache__` committed to repo
- No Bedrock/Nova integration despite project doc claiming it as next step
- Import-only tests padding test count
- Author field set to "HFUPS Team"

### Phase 1 — Housekeeping ✅
- `.gitignore` created (covers `__pycache__`, `.env`, `outputs/`, dist, etc.)
- Author updated in `pyproject.toml` to `Chris 'C-Lord' Kulpa`
- `nova = ["boto3>=1.34"]` added as optional dependency
- All 91 tests passing confirmed before proceeding

### Phase 2 — Legacy Prompt Builder Deprecated ✅
- Deprecation warning added to `src/hfups/prompt_builder.py`
- Points users to `hfups.nova.prompt_templates.build_nova_prompt`
- File retained (not deleted) — still referenced by existing code

### Phase 3 — CLI Demo Fixed ✅
- `cli_demo.py` rebuilt to use `KeyframePacket` pipeline end-to-end
- Now imports from `hfups.nova.prompt_templates` not legacy builder
- Result dict schema updated: `encoded_bytes`, `airtime_10kbps_seconds`, `prompt`, `object_count`, `template`
- Legacy imports removed
- `test_cli_demo.py` and `test_cli_demo_output_file.py` updated to match new schema

### Phase 4 — Nova Bedrock Client ✅
- `src/hfups/nova/bedrock_client.py` created
- `invoke_nova_canvas(prompt, region, profile, width, height) -> bytes`
- Lazy boto3 import with clear install instructions on failure
- All boto3/botocore exceptions wrapped in RuntimeError
- 3 new tests: returns-bytes (mocked), missing-boto3, ClientError handling

### Phase 5 — Full Demo Runner ✅
- `src/hfups/demo/__init__.py` created
- `src/hfups/demo/run.py` created — full CLI pipeline:
  - `--image` (required)
  - `--nova` (flag)
  - `--template` (concise/descriptive/disaster_response/cinematic, default: disaster_response)
  - `--out` (output folder, default: outputs/)
  - `--profile` (AWS CLI profile)
  - `--conf` (YOLO confidence threshold, default: 0.15)
- Saves: `recon.png`, `prompt.txt`, `vara_out.txt`, `summary.json`
- Fails loudly if `--nova` used without valid credentials
- `hfups-demo-run` entry point added to `pyproject.toml`
- 3 new tests added

### Phase 6 — Final Verification ✅
- **97 tests passing, 1 xfailed, 0 failed** (up from 91+1)
- All new files confirmed present
- All entry points confirmed in `pyproject.toml`

### Live Testing ✅
- AWS CLI installed and configured (us-east-1, IAM user: HFUPS)
- `aws sts get-caller-identity` confirmed working
- `python -m hfups.demo.run --image <path>` confirmed working
- Sample output on duck/apple test image:
  - Objects detected: 1 (Person, 85% confidence, middle-center)
  - Encoded packet: 5 bytes
  - Airtime at 10kbps: 4.0ms
  - Template: disaster_response

### Blocker Encountered — Nova Canvas Legacy ✅ (Workaround in Progress)
- `amazon.nova-canvas-v1:0` hit legacy status March 30, 2026 (literally today)
- No active text-to-image model available in AWS Bedrock account
- Stability AI models in account are editing/upscaling only, not text-to-image
- **Decision: Swap reconstruction backend to Hugging Face Inference API**
- Model selected: `black-forest-labs/FLUX.1-schnell`
- HF account confirmed: huggingface.co/CMK83
- Implementation pending next session

---

## 5. Completion Path

### Immediate Next Session
- [ ] Set `HF_API_TOKEN` environment variable (permanent, User scope)
- [ ] Accept FLUX.1-schnell terms at huggingface.co/black-forest-labs/FLUX.1-schnell
- [ ] Claude Code: Create `src/hfups/nova/hf_client.py`
  - `invoke_image_generation(prompt, token, width, height) -> bytes`
  - Reads token from `HF_API_TOKEN` env var by default
  - Same interface contract as `bedrock_client.py`
- [ ] Claude Code: Update `src/hfups/demo/run.py`
  - Add `--backend` flag: choices `huggingface` (default) / `bedrock`
  - Route to appropriate client based on flag
  - Read `HF_API_TOKEN` from environment, fail loudly if missing when backend=huggingface
- [ ] Claude Code: Add 3 tests for `hf_client.py` (mocked requests)
- [ ] Run live end-to-end: image → YOLO → 5 bytes → FLUX.1 → `recon.png`
- [ ] Confirm `recon.png` opens and contains a plausible scene reconstruction

### Short Term (This Week)
- [ ] Curate 6–10 compelling demo images (disaster/emergency scenarios work best)
- [ ] Run pipeline against each, collect `summary.json` outputs
- [ ] Evaluate reconstruction quality across templates
- [ ] Identify best template per scenario type
- [ ] Get General class ham radio license (study materials: HamStudy.org)

### Medium Term
- [ ] Record 60–90 second demo video
  - Show input image (MB scale)
  - Show 5-byte packet
  - Show 4ms airtime
  - Show VARA bridge output
  - Show FLUX.1 reconstruction
- [ ] Write arXiv white paper draft
- [ ] Open source release — push to GitHub public, post to r/amateurradio and ARRL forums
- [ ] Build Pi + (tr)uSDX demo unit for over-the-air transmission
- [ ] Get ham radio license (General class minimum for HF)

### Long Term
- [ ] Real two-station HF demo (requires license)
- [ ] ARES/RACES community outreach and field testing
- [ ] NTT / NICT contact via white paper or warm introduction

---

## 6. Repository State

**GitHub:** https://github.com/C-Lord01/hfups-core  
**Branch:** main  
**Test count:** 97 passed, 1 xfailed  
**Python:** 3.14 (per pycache artifacts)  
**Entry points:** `hfups-demo`, `hfups-rx`, `hfups-tx`, `hfups-demo-run`

### Key File Map
```
src/hfups/
  vision/           # YOLO adapter, keyframe builder, class mapping, tracker
  transport/        # VARA text bridge, TCP, serial, shim
  nova/             # Prompt builder, 4 templates, bedrock client (HF client pending)
  demo/             # Full pipeline CLI runner
  integrations/     # TX/RX keyframe wrappers
  packets.py        # IFrame, MFPacket, ClipParams bit packing
  framing.py        # COBS framing
  crc.py            # CRC helpers
  streaming.py      # Frame stream decoder
tests/              # 97 tests
tools/              # OpenImages dict builder
data/
  demo/             # Test images
  mappings/         # YOLO→OpenImages class maps
  openimages/       # Class descriptions CSV
models/
  yolov8n.pt        # YOLO nano model
```

---

## 7. Decisions Made This Session

| Decision | Rationale |
|---|---|
| AWS CLI profile for credentials | Safest for open source — no secrets in code |
| us-east-1 as Nova region | Nova Canvas only available in us-east-1/us-west-2 |
| FLUX.1-schnell via HF API | Nova Canvas hit legacy status; FLUX is superior quality anyway |
| HF backend as default | More portable, less AWS-dependent for open source release |
| `--conf` default 0.15 | YOLOv8n nano misses objects at 0.25; 0.15 improves recall |
| disaster_response as default template | Best fit for stated use case |
| Fail loudly on missing credentials | Explicit over silent for a CLI tool |

---

## 8. Known Issues / Tech Debt

- Legacy `src/hfups/prompt_builder.py` still present — deprecation warning added, full removal pending after downstream references cleaned up
- Import-only tests (`test_vara_tcp_import.py`, `test_vara_probe_import.py`, `test_vara_smoke_import.py`) pad test count without testing behavior — flag for future cleanup
- `vara_tcp.py` has no behavioral tests
- `recon.png` reconstruction quality untested against real disaster imagery
- YOLOv8n misses small/occluded objects — consider YOLOv8s upgrade for demo images

---

## 9. Core Message (Unchanged)

HFUPS demonstrates that:
- HF radio can carry situational awareness, not just text
- Semantic compression enables image reconstruction over ultra-low bandwidth
- Generative AI can reconstruct scenes from structured meaning

**This is not a toy wrapper — it is a new transmission paradigm.**
