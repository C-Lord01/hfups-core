# HFUPS Nova Prompt Samples

Quick judge playbook:
1. Run `python -m hfups.cli_keyframe --image data/demo/some_image.jpg --model models/yolov8n.pt --nova-template cinematic --out-prompt demo_prompt.txt`
2. Open `demo_prompt.txt`.
3. Copy prompt text and paste into Nova Canvas.

Each sample below shows a compact semantic input summary and four deterministic prompt styles.

## 1) Traffic head-on collision

Semantic input (Keyframe-like):
```python
{
  "objects": [
    (101, 1, 6, 2, 14),  # sedan
    (102, 3, 5, 2, 13),  # pickup truck
    (201, 2, 6, 1, 11),  # person
    (301, 1, 5, 1, 10),  # smoke
  ],
  "deltas": [(0, 1, 0)]
}
```

- concise:
`Overturned sedan lower-left; red pickup middle-left; 1 person nearby; smoke visible.`

- descriptive:
`A scene showing the following elements. A sedan is in the bottom-left (grid 2,7) with confidence 95%. A pickup truck is in the bottom-center (grid 4,6) with confidence 85%. A person is in the bottom-left (grid 3,7) with confidence 75%. Smoke is in the middle-left (grid 2,6) with confidence 65%. Over the next second the sedan moves slightly right.`

- disaster_response:
`URGENT: possible head-on collision with rollover risk. Hazard observed: smoke in the lower-left roadway corridor. A sedan appears overturned near grid 2,7 and a pickup is stopped nearby, indicating a blocked lane. One person is visible close to the impact area and may need immediate medical assessment.`

- cinematic:
`Night rain reflects off broken asphalt as an overturned sedan dominates the lower-left while a battered pickup sits near the center-left lane. A lone person stands close to the wreck and smoke lifts into the wet air behind them. Emphasize emergency lights, reflective puddles, and tense stillness before responders arrive. Over the next second the sedan drifts slightly right, implying unstable motion. Art direction: photorealistic, high dynamic range, shallow depth of field, gritty documentary style.`

## 2) Wildfire smoke plume

Semantic input (Keyframe-like):
```python
{
  "objects": [
    (301, 4, 1, 3, 14),  # smoke
    (401, 3, 4, 2, 12),  # tree
    (401, 5, 4, 2, 11),  # tree
  ],
  "deltas": [(0, 0, 1)]
}
```

- concise:
`Heavy smoke upper-center; tree line below; plume rising upward.`

- descriptive:
`A scene showing the following elements. Smoke is in the top-center (grid 5,2) with confidence 95%. A tree is in the middle-center (grid 4,5) with confidence 80%. A second tree is in the middle-right (grid 6,5) with confidence 75%. Over the next second the smoke moves slightly down, indicating expansion across the canopy.`

- disaster_response:
`URGENT: possible wildfire spread. Dense smoke is visible above a tree line and may indicate active fire beyond the ridge. Vegetation sits directly under the plume, raising risk of fast lateral spread and reduced evacuation visibility.`

- cinematic:
`A dense smoke column blooms over dark trees as wind pulls the plume across the skyline. The foreground tree line appears dry and vulnerable, with ash haze softening the horizon and compressing depth. Keep the image wide, atmospheric, and tense, with layered smoke texture and subtle ember glow. Over the next second the plume expands and shifts across frame. Art direction: photorealistic, high dynamic range, shallow depth of field, gritty documentary style.`

## 3) Weather/tree-block

Semantic input (Keyframe-like):
```python
{
  "objects": [
    (401, 3, 5, 3, 13),  # fallen tree
    (101, 5, 6, 2, 12),  # car
    (501, 5, 6, 2, 11),  # fire
  ],
  "deltas": []
}
```

- concise:
`Fallen tree blocks road; car lower-right; fire visible near vehicle.`

- descriptive:
`A scene showing the following elements. A tree is in the bottom-center (grid 4,6) with confidence 85%. A car is in the bottom-right (grid 6,7) with confidence 80%. Fire is in the bottom-right (grid 6,7) with confidence 75%. The road appears obstructed by the fallen tree.`

- disaster_response:
`URGENT: possible storm damage and vehicle fire. A large fallen tree blocks the roadway while a car is stopped at the obstruction point. Fire is visible near the vehicle position, indicating escalating danger and potential secondary ignition.`

- cinematic:
`After severe weather, a massive fallen tree sprawls across the road while a stranded car burns at the lower-right edge. Wet pavement, drifting smoke, and fractured branches create a chaotic disaster tableau. Emphasize tangled debris geometry, flame reflections, and emergency urgency in a grounded documentary perspective. Art direction: photorealistic, high dynamic range, shallow depth of field, gritty documentary style.`

## 4) Military convoy

Semantic input (Keyframe-like):
```python
{
  "objects": [
    (601, 1, 4, 1, 13),  # APC 1
    (601, 2, 4, 1, 12),  # APC 2
    (601, 3, 4, 1, 12),  # APC 3
    (601, 4, 4, 1, 11),  # APC 4
    (601, 5, 4, 1, 11),  # APC 5
  ],
  "deltas": [(0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0)]
}
```

- concise:
`Five APCs aligned mid-frame; convoy moving to the right.`

- descriptive:
`A scene showing the following elements. Five APC vehicles are aligned across the middle corridor from left to right, each with medium confidence. The formation suggests coordinated convoy movement. Over the next second the APC line moves slightly right in unison.`

- disaster_response:
`URGENT: armed convoy movement detected. Multiple APCs are traveling in formation across the center lane, suggesting coordinated transit and potential access control operations. Movement is rightward and synchronized.`

- cinematic:
`Five armored APCs cut across the frame in a disciplined line, engines low and heavy, dust drifting behind the column. Keep composition lateral and deliberate to emphasize coordination, mass, and tactical spacing. The convoy advances rightward as one unit, projecting controlled force and momentum. Art direction: photorealistic, high dynamic range, shallow depth of field, gritty documentary style.`

## 5) Breach/trespass

Semantic input (Keyframe-like):
```python
{
  "objects": [
    (201, 2, 5, 1, 13),  # person 1
    (201, 3, 5, 1, 12),  # person 2
    (201, 4, 5, 1, 12),  # person 3
    (201, 5, 5, 1, 11),  # person 4
    (701, 3, 4, 2, 10),  # fence
  ],
  "deltas": [(0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0)]
}
```

- concise:
`Four people near fence line; movement rightward suggests active trespass.`

- descriptive:
`A scene showing the following elements. Four people occupy the lower-middle corridor near a fence in the middle-center (grid 4,5). Confidence values indicate clear human detection and probable coordinated movement. Over the next second the group moves moderately right.`

- disaster_response:
`URGENT: possible perimeter breach. Multiple people are clustered near a fence line and moving in the same direction, suggesting active trespass. Recommend immediate verification, route blocking, and responder safety protocols.`

- cinematic:
`A fence line divides the scene as four figures surge across the lower-middle corridor in a coordinated run. Emphasize urgency, motion blur at foot level, and tense low-angle framing that highlights intent and direction of escape. Keep the environment realistic and surveillance-like, with hard shadows and practical lighting. Art direction: photorealistic, high dynamic range, shallow depth of field, gritty documentary style.`

## 6) Coastal flooding

Semantic input (Keyframe-like):
```python
{
  "objects": [
    (801, 4, 6, 2, 13),  # boat
    (901, 4, 5, 3, 12),  # floodwater
    (101, 2, 6, 1, 10),  # car
  ],
  "deltas": [(0, -1, 0)]
}
```

- concise:
`Floodwater over road; boat partially submerged; stranded car left side.`

- descriptive:
`A scene showing the following elements. Floodwater covers the lower-middle roadway (grid 5,6) with confidence 80%. A boat sits partially submerged in the bottom-center (grid 5,7) with confidence 85%. A car remains stranded in the bottom-left (grid 3,7) with confidence 65%. Over the next second the boat moves slightly left.`

- disaster_response:
`URGENT: coastal flooding is impacting road access. Floodwater has overtopped the roadway and a partially submerged boat indicates strong water movement. A stranded car remains within the flood zone, suggesting possible occupant risk and blocked evacuation routes.`

- cinematic:
`Brackish floodwater swallows the road as a partially submerged boat drifts through the lower frame and a stranded car waits at the edge of inundation. Push a storm-muted palette, reflective water texture, and heavy coastal atmosphere with distant haze. The boat shifts leftward as current pressure builds, reinforcing ongoing flood motion. Art direction: photorealistic, high dynamic range, shallow depth of field, gritty documentary style.`
