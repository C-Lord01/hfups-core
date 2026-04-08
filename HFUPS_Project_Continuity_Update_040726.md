# HFUPS — Project Continuity Update
**Date:** April 7, 2026
**Status:** Patent Pending — Pre-Demo Development Phase
**Author:** Chris 'C-Lord' Kulpa (C-Lord)

---

## 1. Project Vision (Expanded)

HFUPS began as a semantic compression codec for disaster imagery over HF radio. It has since expanded into a broader paradigm: semantic reconstruction pipelines driven by vector embeddings and generative models.

**Original thesis:** When bandwidth is scarce, transmit meaning — not pixels.

**Expanded thesis:** When both sides share the same understanding of the world, transmission becomes nearly unnecessary. Send intent. Reconstruct reality.

The long-term target is not disaster response alone. It is semantic video calling at dialup-class speeds — a capability that has never existed. Video presence over HF radio, satellite, or any constrained channel, without pixel transmission of any kind.

---

## 2. Why This Exists

Hurricane Helene struck Greenville, SC. All communications infrastructure failed. Chris could not reach his wife's elderly parents. Roads were blocked. The inability to transmit even basic visual situational awareness — over infrastructure that already existed (HF radio) — was the direct inspiration.

HFUPS is the answer to that failure. Not a better compression algorithm. A fundamentally different transmission paradigm.

**Stated long-term targets:**
- ARES / RACES emergency communications community
- White paper submission (arXiv: cs.CV or cs.NI)
- NTT / NICT — Japan's disaster communications R&D apparatus
- Defense and government emergency response contractors
- Patent licensing to 5G/6G semantic communications implementers

---

## 3. Patent Status

**US Provisional Application #64/029,991**
- Filed: April 5, 2026 at 3:44 PM ET via USPTO EFS-Web
- Title: *System and Method for Deterministic Semantic Encoding and Generative Reconstruction Using Shared Vocabulary Contracts*
- Entity: Micro entity — $65 filing fee paid
- Confirmation #: 8054
- Patent Center #: 75158994
- **Conversion deadline: April 5, 2027**

**13 claims filed covering:**
1. Core method — extract, encode, transmit, reconstruct
2. Shared vocabulary with version identifier and hash verification
3. No pixel-level data transmitted
4. Conditioning via vocabulary expansion to structured descriptors
5. Bandwidth constraint — ≤10 kbps channel
6. Receiver gating — reconstruction withheld on vocabulary mismatch
7. No natural language transmitted
8. Channel independence — HF radio through 6G and fiber
9. Delta encoding between semantic frames
10. Temporal coherence post-processing
11. Embedding-based generative conditioning
12. Identity persistence via latent vectors across frames
13. Audio integration via low-bitrate codec

**Strategic position:** The core defensible novelty is the vocabulary contract mechanism — versioned, hash-verified, shared between endpoints. Neither pixel data nor natural language is transmitted. The receiver expands symbolic identifiers into generative conditioning inputs. This combination does not exist in prior art.

**Empirical compression data on record:**
- 10 disaster scenario images tested (4800×3584px, 8–19MB JPEG)
- Conservative floor: 250,000:1 vs JPEG (9 objects, 35-byte packet, 8.8MB source)
- Range: 250,000:1 to 18,900,000:1 vs JPEG
- Range: 1,400,000:1 to 51,600,000:1 vs raw uncompressed

**Licensing market context:** The global 5G SEP licensing market is approximately $15B annually. A single foundational method patent with a working demo realistically targets $500K–$10M per licensee in the near term, with standard-essential territory possible if semantic communications becomes incorporated into 6G standardization — which current 3GPP research trajectories suggest is likely.

---

## 4. Current Architecture

### A. Edge — Transmitter
- Image → YOLOv8n detection (conf=0.10) → class mapping via Open Images V7
- Objects ranked, deduplicated (up to 4 per class), capped at 12
- Positions quantized to 8×8 grid (internal only — not exposed in prompts)
- Confidence quantized 0–15 (internal only)
- Packed into deterministic bit-level `KeyframePacket`
- Worst case: 46 bytes (12 objects), best case: 5 bytes (1 object)
- `DeltaPacket` implemented for motion tracking between frames

### B. Transport Layer
Three modes implemented and tested:
- **Shim** — in-process for testing
- **TCP** — live streaming with optional ACK/resend
- **VARA HF Text Bridge** — chunked Base64 ASCII with CRC32 per chunk, out-of-order reassembly, corruption detection

### C. Receiver / Reconstruction
- Decodes semantic packet → object list with spatial positions
- UPS prompt reconstruction via `_ups_template` (4-layer: realism, subject/scene, camera/lighting, imperfections)
- Natural language spatial descriptions — no grid coordinates, no confidence percentages in output
- Prompt fed to FLUX.1-schnell via HuggingFace Inference API (endpoint: router.huggingface.co)
- CLI driven, outputs saved to `outputs/` folder

### D. Embedding Research Branch (hfups-embed-lab)

**Phase 1 — CLIP ViT-B/32 Coherence (Complete)**
- Separation ratio: 2.12x (intra-group distance 0.1463 vs inter-group 0.3102)
- Nearest-neighbor recall: 4/6 = 0.67 (above 0.5 threshold)
- Tightest pair: coastal surge images at cosine distance 0.0369
- Conclusion: embedding space is coherent enough for delta compression

**Phase 2 — FAISS Vocabulary Index + Delta Magnitude Measurement (Complete)**
- `VocabIndex` class implemented: `IndexFlatL2` over L2-normalized CLIP embeddings
- k=1 self-query returns correct label for all 10 images (test_vocab_index_builds passing)
- Delta magnitude measurement across all 45 image pairs (10 choose 2):

| Metric | Value |
|--------|-------|
| Intra-group mean delta | 0.4879 |
| Inter-group mean delta | 0.7816 |
| Compression advantage ratio | **1.60x** |
| Tightest intra-group pair | coastal surge (imgs 4+6): 0.2718 |
| All-pairs min | 0.2718 |
| All-pairs max | 0.9202 |
| All-pairs mean | 0.7620 |
| All-pairs std | 0.1339 |

- **Conclusion:** 1.60x separation ratio confirms delta compression is viable. Semantically similar scenes cluster significantly closer in embedding space than dissimilar scenes. Claims 9–12 have empirical foundation.
- Implementation: `open_clip` ViT-B/32 (OpenAI pretrained weights), consistent across Phase 1 and Phase 2
- All 3 Phase 2 tests passing: `test_vocab_index_builds`, `test_delta_magnitudes`, `test_delta_magnitude_distribution`

---

## 5. Work Completed Since Last Update (April 6–7, 2026)

### Codebase Cleanup ✅
- `invoke_image_generation` and unused `_HF_API_URL` deleted from `nova/hf_client.py`
- `cli_rx_keyframe.py` updated to import from `nova.prompt_templates` (was using old builder with grid coord leakage)
- Committed `__pycache__` directories removed from git tracking
- `test_placeholder.py` deleted
- `nova/__init__.py` dual aliasing resolved — `build_nova_prompt` export removed, `build_nova_template_prompt` is canonical
- TODO comment added to 6-object cap in `_ups_template` with rationale
- All cleanup committed and pushed to `hfups-core` main

### FLUX End-to-End Pipeline Validated ✅
- HuggingFace endpoint updated from `api-inference.huggingface.co` (deprecated, 410 Gone) to `router.huggingface.co`
- Full pipeline confirmed working: image → YOLO → KeyframePacket → UPS prompt → FLUX.1-schnell → recon.png
- All 10 demo images processed; pipeline stable

### FLUX Reconstruction Quality Assessment ✅
- Reconstruction coherence is low for current demo images
- Root cause: YOLOv8n detects generic objects (car, person) but misses disaster-specific semantics (overturned vehicle, floodwater, smoke plume)
- Images 3, 5, 9: zero detections — fallback to generic "A disaster scene" prompt
- Image 7 (overturned truck): YOLO detected "truck + boat" (guardrail/undercarriage misclassified) — no road, night, fog, or rollover state transmitted
- **Diagnosis confirmed:** pipeline is correct; detection layer is the bottleneck
- Manual `--caption` flag validates that accurate semantic input produces coherent reconstructions
- **Decision:** disaster-specific fine-tuning is the correct and necessary next step

### Embedding Lab Phase 2 ✅
- `hfups-embed-lab` repo rebuilt (Phase 1 source was untracked, reconstructed from test file)
- Phase 2 implementation complete: `src/hfups_embed_lab/vocab.py` (VocabIndex), `tests/test_phase2_faiss.py`
- All 3 Phase 2 tests passing
- Empirical delta magnitude data recorded above — white paper and patent prosecution record
- Embed lab committed locally (not yet pushed to remote)

**Test count (hfups-core):** 113 passed, 1 xfailed, 0 failed

---

## 6. Demo Roadmap (Priority Order)

### 1. ~~FLUX End-to-End Live Generation~~ ✅ Complete
- Pipeline working, quality bottleneck identified as detection layer

### 2. ~~Embedding Lab Phase 2~~ ✅ Complete
- FAISS index built, delta magnitudes measured, 1.60x compression advantage confirmed

### 3. Disaster-Specific Detection Model — Next Priority (2–4 Weeks)
- YOLOv8n fails on images 3, 5, 9 (haboob, aerial flood, urban flood)
- Source labeled flood/tsunami/wildfire dataset (NOAA, FEMA, Roboflow Universe, academic)
- Fine-tune YOLOv8s or YOLOv8m on disaster imagery
- Target semantic classes: floodwater, overturned vehicle, smoke plume, downed power lines, collapsed structure, person in distress
- Target: zero 0-detection scenes across all 10 demo images
- Claude Code handles training pipeline

### 4. Facial Biometrics Model — 2–3 Weeks (after Phase 2 ✅)
- Identity-specific embedding vectors established at session init
- Transmitted out-of-band, used by receiver to lock subject identity
- Enables semantic video calling with consistent face reconstruction
- Directly implements Claim 12
- Prerequisite (Phase 2) now complete — can begin after disaster model

### 5. Pi 5 + HF Hardware Demo — Requires Ham License
- Study General class license: HamStudy.org
- Hardware: Raspberry Pi 5 + Coral USB accelerator (transmitter)
- Radio: (tr)uSDX HF transceiver
- Receiver: any machine capable of running FLUX inference
- This is the cinematic demo moment — 35 bytes transmitted over radio, scene reconstructed on the other side

### 6. Video + Lip Sync Receiver — Stretch Goal
- AnimateDiff or FLUX video for temporal coherence between frames
- Lip sync layer conditioned on audio semantic stream
- Full semantic video call demonstration
- Implements Claims 10, 11, 12, 13 simultaneously

### 7. White Paper — Write After Demo Exists
- Document what was built, not what is planned
- Target: arXiv cs.CV or cs.NI
- Include empirical benchmark data, compression ratios, embedding coherence results, delta magnitude measurements
- Do not publish before demo is running

### 8. Open Source Release — After Physical Demo
- Push GitHub repo public
- Post to r/amateurradio and ARRL forums
- Community release with working demo video is the correct sequencing
- Do not release early — a working HF demo is the stake in the ground

---

## 7. Strategic Context

**What this becomes if executed:**

The demo roadmap culminates in a working semantic video call over HF radio at 2–10 kbps. That capability does not exist anywhere. It has never been demonstrated. A 30-second video call over radio — even low fidelity, even with latency — transmitted at dialup-class speeds using 35 bytes per frame, is a story no one in communications research can ignore.

The social value: 3.5 billion people with no reliable internet access gain a path to visual communication over existing HF infrastructure that already has global reach.

The commercial value: licensing to semantic communications implementers in the 5G/6G space, government emergency response contractors, and potentially NTT/NICT for disaster infrastructure. Single foundational patent licensing deals in this space range from $500K to $10M per licensee.

The nonprovisional must be filed before April 5, 2027. The demo, the white paper, and the embedding work done between now and then are what determine whether the nonprovisional converts to a granted patent with enforcement teeth or sits as a pending application with limited leverage.

---

## 8. Repository State

**GitHub:** https://github.com/C-Lord01/hfups-core (private — do not open until physical demo complete)
**Embedding lab:** hfups-embed-lab (local only — remote not yet configured)
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
                    # hf_client.py: router.huggingface.co endpoint (updated April 7)
  demo/             # Full pipeline CLI runner (--backend huggingface default)
  integrations/     # TX/RX keyframe wrappers (cli_rx_keyframe.py uses prompt_templates)
  packets.py        # Legacy IFrame, MFPacket, ClipParams bit packing
  framing.py        # COBS framing
  crc.py            # CRC helpers
  streaming.py      # Frame stream decoder
tests/              # 113 tests
tools/
  benchmark_compression.py  # Empirical compression ratio measurement
  build_openimages_dict.py
data/
  demo/             # 10 disaster scenario test images (1.jpg–10.jpg)
  mappings/         # YOLO→OpenImages class maps
  openimages/       # Class descriptions CSV
models/
  yolov8n.pt        # YOLO nano model (confirmed correct)

hfups-embed-lab/ (separate repo)
  src/hfups_embed_lab/
    vocab.py        # VocabIndex: FAISS IndexFlatL2 over CLIP embeddings
  tests/
    test_embedding_coherence.py  # Phase 1: separation ratio, recall
    test_phase2_faiss.py         # Phase 2: FAISS index, delta magnitudes
```

---

## 9. Known Issues / Tech Debt

- YOLOv8n fails on images 3, 5, 9 — atmospheric occlusion, aerial perspective, water surfaces — addressed by disaster-specific fine-tuning (next roadmap item)
- Legacy `src/hfups/prompt_builder.py` still present — deprecation warning added, removal pending after downstream references cleaned up
- Import-only VARA tests pad test count without behavioral assertions — flagged for future cleanup
- `vara_tcp.py` has no behavioral tests
- UPS Layer C (camera/lighting) is static — not yet derived from scene context (night/indoor/outdoor)
- UPS Layer B caps at 6 objects in prompt — TODO added; 6 of 12 encoded objects dropped at generation time; revisit with priority-ranked truncation after disaster model work
- hfups-embed-lab not yet pushed to remote GitHub repo

---

## 10. Core Message

HFUPS demonstrates that:
- HF radio can carry situational awareness, not just text
- Semantic compression enables image reconstruction over ultra-low bandwidth
- Generative AI can reconstruct scenes from structured meaning
- The vocabulary contract between transmitter and receiver is the invention — not the models on either end
- Embedding space coherence (1.60x delta compression advantage) provides the mathematical foundation for Claims 9–12

**The long game:** Video calling at dialup-class speeds. Completely unheard of. That is what we are building toward.

**This is not a toy wrapper — it is a new transmission paradigm.**
