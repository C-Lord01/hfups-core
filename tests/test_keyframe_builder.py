import json
from pathlib import Path

from hfups.vision.class_mapping import ClassMapper
from hfups.vision import keyframe_builder as keyframe_builder_module
from hfups.vision.keyframe_builder import KeyframeBuilder
from hfups.vision.openimages_dict import OpenImagesClass, OpenImagesDict
from hfups.vision.yolo_adapter import Detection


def _build_openimages_dict(labels: list[str]) -> OpenImagesDict:
    classes = [
        OpenImagesClass(id=idx, label=label, name=f"Class {idx}")
        for idx, label in enumerate(labels)
    ]
    return OpenImagesDict(
        classes=classes,
        by_id={entry.id: entry for entry in classes},
        by_label={entry.label: entry for entry in classes},
    )


def _write_mapping(path: Path, mapping: dict[str, str]) -> Path:
    path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    return path


def test_mapping_drop_unmapped_detections(tmp_path: Path) -> None:
    openimages = _build_openimages_dict(["/m/person", "/m/car", "/m/bus"])
    mapping_path = _write_mapping(
        tmp_path / "mapping.json",
        {
            "person": "/m/person",
            "car": "/m/car",
            "bus": "/m/bus",
        },
    )
    mapper = ClassMapper(openimages, mapping_path=mapping_path)
    builder = KeyframeBuilder(mapper)

    detections = [
        Detection("person", 0.90, 10, 10, 100, 100),
        Detection("car", 0.80, 50, 50, 140, 140),
        Detection("bus", 0.70, 150, 150, 300, 300),
        Detection("cat", 0.95, 10, 10, 20, 20),
        Detection("truck", 0.95, 30, 30, 40, 40),
    ]

    packet, stats = builder.build_with_stats(detections, image_width=800, image_height=600)

    assert stats["mapped_detections"] == 3
    assert len(packet.objects) == 3


def test_quantization_correctness_for_grid_extremes(tmp_path: Path) -> None:
    openimages = _build_openimages_dict(["/m/person", "/m/car"])
    mapping_path = _write_mapping(
        tmp_path / "mapping.json",
        {
            "person": "/m/person",
            "car": "/m/car",
        },
    )
    mapper = ClassMapper(openimages, mapping_path=mapping_path)
    builder = KeyframeBuilder(mapper)

    detections = [
        Detection("person", 0.95, 0.0, 0.0, 0.0, 0.0),
        Detection("car", 0.90, 799.0, 599.0, 799.0, 599.0),
    ]
    packet = builder.build(detections, image_width=800, image_height=600)

    assert packet.objects[0].cx == 0
    assert packet.objects[0].cy == 0
    assert packet.objects[1].cx == 7
    assert packet.objects[1].cy == 7


def test_selection_ordering_is_deterministic(tmp_path: Path) -> None:
    openimages = _build_openimages_dict(["/m/person", "/m/car", "/m/bus"])
    mapping_path = _write_mapping(
        tmp_path / "mapping.json",
        {
            "person": "/m/person",
            "car": "/m/car",
            "bus": "/m/bus",
        },
    )
    mapper = ClassMapper(openimages, mapping_path=mapping_path)
    builder = KeyframeBuilder(mapper)

    detections = [
        Detection("car", 0.90, 0.0, 0.0, 10.0, 10.0),
        Detection("person", 0.90, 20.0, 0.0, 30.0, 10.0),
        Detection("bus", 0.90, 40.0, 0.0, 45.0, 5.0),
    ]
    packet = builder.build(detections, image_width=800, image_height=600)

    assert [obj.class_id for obj in packet.objects] == [0, 1, 2]


def test_max_objects_cap_and_encoded_size(tmp_path: Path) -> None:
    labels = [f"/m/{idx:06d}" for idx in range(20)]
    mapping = {f"class_{idx}": label for idx, label in enumerate(labels)}
    openimages = _build_openimages_dict(labels)
    mapping_path = _write_mapping(tmp_path / "mapping.json", mapping)

    mapper = ClassMapper(openimages, mapping_path=mapping_path)
    builder = KeyframeBuilder(mapper)

    detections = [
        Detection(
            class_name=f"class_{idx}",
            confidence=0.70,
            x1=float(idx),
            y1=float(idx),
            x2=float(idx + 10),
            y2=float(idx + 10),
        )
        for idx in range(20)
    ]

    packet = builder.build(detections, image_width=800, image_height=600)

    assert len(packet.objects) == 12
    assert len(packet.encode()) == 46


def test_build_keyframe_from_image_includes_debug_fields_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dict_path = tmp_path / "dict.json"
    dict_path.write_text(
        json.dumps(
            {
                "schema": "hfups.dict.openimages.boxable.v1",
                "source": {
                    "dataset": "openimages",
                    "version": "v7",
                    "subset": "boxable",
                    "input_file": "data/openimages/class-descriptions-boxable.csv",
                },
                "created_utc": "2026-02-26T00:00:00Z",
                "classes": [
                    {"id": 0, "label": "/m/person", "name": "Person"},
                    {"id": 1, "label": "/m/car", "name": "Car"},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    mapping_path = _write_mapping(
        tmp_path / "mapping.json",
        {
            "person": "/m/person",
            "car": "/m/car",
        },
    )

    detections = [
        Detection("zebra", 0.80, 10, 10, 40, 40),
        Detection("car", 0.40, 50, 50, 120, 120),
        Detection("person", 0.90, 200, 100, 260, 300),
        Detection("apple", 0.95, 5, 5, 15, 15),
    ]

    class FakeRunner:
        def __init__(self, model_path):
            del model_path

        def detect(self, image_path, conf=0.25):
            del image_path, conf
            return detections

    monkeypatch.setattr(keyframe_builder_module, "YoloRunner", FakeRunner)
    monkeypatch.setattr(keyframe_builder_module, "_read_image_size", lambda path: (800, 600))

    packet, debug = keyframe_builder_module.build_keyframe_from_image(
        image_path=tmp_path / "image.jpg",
        model_path=tmp_path / "model.pt",
        dict_path=dict_path,
        mapping_path=mapping_path,
        include_raw=True,
        include_unmapped=True,
        debug_top_n=3,
    )

    assert len(packet.objects) == 2
    assert debug["mapped_detections"] == 2
    assert debug["unmapped_class_names"] == ["apple", "zebra"]
    assert [item["name"] for item in debug["top_raw_detections"]] == [
        "apple",
        "person",
        "zebra",
    ]
