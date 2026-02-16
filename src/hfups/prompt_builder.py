"""Build render-friendly scene specs and Nova prompt text from HFUPS state."""

CAM_MOVE_LABELS = {
    0: "static",
    1: "slow pan",
    2: "slow tilt",
    3: "dolly in",
    4: "dolly out",
    42: "tracking sweep",
}

FRAMING_LABELS = {
    0: "wide",
    1: "medium",
    2: "close-up",
    3: "overhead",
}

SUBJECT_PATH_LABELS = {
    0: "stationary",
    1: "left-to-right",
    2: "right-to-left",
    3: "toward camera",
    4: "away from camera",
}

FPS_BY_INDEX = {0: 6, 1: 10, 2: 15, 3: 24}


def _label(mapping: dict[int, str], value: int | None, prefix: str) -> str:
    if value is None:
        return f"unknown_{prefix}"
    return mapping.get(value, f"{prefix}_{value}")


def build_scene_spec(state_dict: dict) -> dict:
    """Convert HFUPS decoded state dict into a renderable JSON-safe scene spec."""
    iframe = state_dict.get("iframe")
    mf = state_dict.get("mf")
    clip_params = state_dict.get("clip_params")

    warnings: list[str] = []

    cam_move = mf.get("cam_move") if isinstance(mf, dict) else None
    framing = mf.get("framing") if isinstance(mf, dict) else None
    subject_path = mf.get("subject_path") if isinstance(mf, dict) else None
    if not isinstance(mf, dict):
        warnings.append("Missing MF packet; camera defaults applied")

    fps_idx = clip_params.get("fps_idx") if isinstance(clip_params, dict) else 0
    frame_count = clip_params.get("frame_count") if isinstance(clip_params, dict) else 0
    if not isinstance(clip_params, dict):
        warnings.append("Missing ClipParams; clip defaults applied")

    fps = FPS_BY_INDEX.get(fps_idx, FPS_BY_INDEX[0])
    if fps_idx not in FPS_BY_INDEX:
        warnings.append(f"Unknown fps_idx={fps_idx}; defaulted to {FPS_BY_INDEX[0]}")

    duration_s = float(frame_count) / float(fps) if fps > 0 else 0.0

    quality = {
        "psnr_idx": iframe.get("psnr_idx") if isinstance(iframe, dict) else None,
        "quant_idx": iframe.get("quant_idx") if isinstance(iframe, dict) else None,
        "q_m": iframe.get("q_m") if isinstance(iframe, dict) else None,
        "q_u": iframe.get("q_u") if isinstance(iframe, dict) else None,
        "q_v": iframe.get("q_v") if isinstance(iframe, dict) else None,
    }
    if not isinstance(iframe, dict):
        warnings.append("Missing I-frame; quality/timestamps incomplete")

    scene_spec = {
        "summary": (
            f"A { _label(FRAMING_LABELS, framing, 'framing') } shot with "
            f"{ _label(CAM_MOVE_LABELS, cam_move, 'cam_move') } movement at {fps} fps."
        ),
        "camera": {
            "movement": _label(CAM_MOVE_LABELS, cam_move, "cam_move"),
            "framing": _label(FRAMING_LABELS, framing, "framing"),
            "subject_path": _label(SUBJECT_PATH_LABELS, subject_path, "subject_path"),
        },
        "clip": {
            "fps": fps,
            "frame_count": int(frame_count),
            "duration_s": duration_s,
        },
        "quality": quality,
        "timestamps": {
            "timestamp_s": state_dict.get("timestamp_s"),
            "shot_id": state_dict.get("shot_id"),
        },
    }

    if warnings:
        scene_spec["warnings"] = warnings

    return scene_spec


def build_nova_prompt(scene_spec: dict) -> str:
    """Build a compact, deterministic Nova-friendly prompt string."""
    camera = scene_spec.get("camera", {})
    clip = scene_spec.get("clip", {})

    movement = camera.get("movement", "unknown_cam_move")
    framing = camera.get("framing", "unknown_framing")
    subject_path = camera.get("subject_path", "unknown_subject_path")
    fps = clip.get("fps")
    duration_s = clip.get("duration_s")

    parts = [
        "Cinematic scene",
        f"camera movement: {movement}",
        f"framing: {framing}",
        f"subject path: {subject_path}",
    ]

    if fps is not None:
        parts.append(f"{fps} fps")
    if duration_s is not None:
        parts.append(f"duration {duration_s:.2f}s")

    return "; ".join(parts)
