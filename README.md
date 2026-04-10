# HFUPS Protocol Core

## Setup and run (Windows)

```powershell
python -m pip install -e .
python -m hfups.cli_demo
hfups-demo
```

## Models

Model weights are not stored in this repository.

To generate `models/yolov8s_disaster.pt`, run:

```
python tools/train_disaster_model.py
```

Expected training time: ~1.25 hours on RTX 3080.
See "Training the Disaster Detection Model" section for full instructions.

Pre-trained weights can be shared separately via direct transfer if needed.

## Run tests

```powershell
python -m pytest
```

Or, if you use Make:

```bash
make test
```
