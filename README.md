# AI-Powered Real-Time Archery Simulator

> An end-to-end computer-vision project that detects arrow hits on a physical archery target, calculates scores, and displays results on a projector.

This repository documents the complete development workflow for an AI-based archery simulator: dataset preparation, annotation, YOLO model experiments, camera integration, arrow-tip localization, scoring logic, game-state management, Arduino triggering, projector rendering, testing, and deployment.

**Project context:** Developed as a freelance AI/ML engineering project for FunTime, Bangalore.  
**Dataset size:** Approximately 4,000 images used during model development.  
**Primary stack:** Python, OpenCV, Ultralytics YOLO, Arduino serial communication, and projector output.

---

## Table of Contents

1. [Project Goals](#project-goals)
2. [System Architecture](#system-architecture)
3. [How the Full Pipeline Works](#how-the-full-pipeline-works)
4. [Hardware Setup](#hardware-setup)
5. [Software Requirements](#software-requirements)
6. [Dataset Collection and Preparation](#dataset-collection-and-preparation)
7. [Annotation Strategy](#annotation-strategy)
8. [Training YOLO Models](#training-yolo-models)
9. [Model Evaluation and Selection](#model-evaluation-and-selection)
10. [Camera Pipeline](#camera-pipeline)
11. [Detection and Arrow-Tip Localization](#detection-and-arrow-tip-localization)
12. [Scoring Engine](#scoring-engine)
13. [Camera-to-Projector Alignment](#camera-to-projector-alignment)
14. [Game-State Management](#game-state-management)
15. [Arduino Integration](#arduino-integration)
16. [Performance and Reliability](#performance-and-reliability)
17. [Testing and Debugging](#testing-and-debugging)
18. [Suggested Repository Structure](#suggested-repository-structure)
19. [Configuration Reference](#configuration-reference)
20. [Reproducibility Checklist](#reproducibility-checklist)
21. [Limitations and Future Improvements](#limitations-and-future-improvements)
22. [Skills Demonstrated](#skills-demonstrated)

---

## Project Goals

The goal was to build a real-world archery experience in which a player shoots physical arrows at a target board while a camera and AI model determine where an arrow has landed.

The system should:

- Detect the target board and the arrow-related objects.
- Locate the arrow tip as accurately as possible.
- Convert the detected hit location into a score.
- Display the target and score through a projector.
- Support a round containing up to five arrows.
- Start a round through a hardware button connected to an Arduino, with a keyboard trigger available for development/testing.
- Prevent false hits, repeated scoring, and unwanted detections before a shot.
- Keep the final score visible until the next round begins.

## System Architecture

```text
                         ┌─────────────────────┐
                         │ Camera captures view│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ OpenCV frame capture│
                         │ and preprocessing   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Trained YOLO model  │
                         │ target / arrow / tip│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Detection filtering │
                         │ and tip localization│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Hit validation and  │
                         │ duplicate protection│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Scoring engine      │
                         │ target geometry     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Game state and score│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Projector rendering │
                         └─────────────────────┘

       Arduino button ───────► Round-start/game controller
       Keyboard test key ────► Round-start/game controller
```

Keep these responsibilities separate in the code. The model should detect objects; the scoring engine should calculate points; the game controller should decide when a hit counts; and the projector renderer should display the current game state.

## How the Full Pipeline Works

1. **Initialize the application:** Load the trained model, open the camera, initialize the serial connection if an Arduino is available, and create the projector and debug windows.
2. **Wait for a start signal:** A player presses the Arduino button or uses the development keyboard trigger.
3. **Capture camera frames:** OpenCV reads frames from the configured camera.
4. **Run inference:** YOLO predicts the target, arrow, and arrow-tip locations.
5. **Validate detections:** Apply confidence thresholds, class checks, geometry checks, and temporal stability checks.
6. **Determine a hit point:** Use the arrow-tip detection or a verified fallback method, if one was implemented.
7. **Validate the hit:** Ensure the point is within the valid target area and is not a duplicate of an already-counted arrow.
8. **Calculate the score:** Compare the hit point with the target center and ring boundaries.
9. **Update the game state:** Record the score and increment the arrow count only once per accepted hit.
10. **Render the result:** Draw the target, hit marker, score, arrow count, and game messages on the projector output.
11. **Finish the round:** Stop accepting new hits after five accepted arrows and keep the final score visible until a new round starts.

The exact order of hit confirmation and the use of any fallback detection should be documented according to the implementation in the repository.

## Hardware Setup

The physical system includes:

- A Windows laptop with an NVIDIA GPU for inference.
- A camera aimed at the target and arrow-impact area.
- A projector connected as a second display.
- An Arduino board and a push button to trigger the next round.
- A physical target board and arrows.

### Camera placement

Camera placement is a major part of the system's accuracy. The camera must see the relevant target area clearly and consistently. Test the setup for:

- Focus and exposure.
- Motion blur during arrow impact.
- Reflections and changing lighting.
- Occlusion of the arrow tip by the arrow shaft or target.
- Perspective distortion and target alignment.
- Stable mounting so the target does not shift in the camera frame.

Use a camera index and resolution that have been tested on the actual machine. A camera may open successfully but still return black frames, incorrect exposure, or a resolution different from the requested setting. Check the captured frames rather than assuming that `VideoCapture` opening successfully means the camera is working correctly.

### Projector setup

Configure the projector as an extended display where possible. Keep the projector rendering independent from the camera debug window so the player sees the clean game interface while the developer can inspect detections on the laptop.

A typical arrangement is:

- Laptop display: camera feed, bounding boxes, confidence values, and diagnostics.
- Projector display: target graphics, hit markers, score, arrow count, and game status.

The projector resolution and display position should be configurable rather than hard-coded throughout the program.

## Software Requirements

The project was developed around Python, OpenCV, and YOLO. Exact package versions should be recorded from the working environment to support reproducibility.

Typical dependencies include:

- Python
- `ultralytics`
- `opencv-python`
- `numpy`
- `pyserial` (if Arduino serial communication is enabled)
- PyTorch with a compatible CUDA build for NVIDIA GPU inference

Install only the dependencies used by the implementation. For a fresh environment, create a virtual environment first:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project's pinned dependencies after adding a tested `requirements.txt`:

```bash
pip install -r requirements.txt
```

Verify that PyTorch can access the GPU in the environment used to run inference. CUDA availability depends on the installed PyTorch build, NVIDIA driver, and compatible hardware.

## Dataset Collection and Preparation

Approximately 4,000 images were used during model development. The dataset should represent the real deployment environment as closely as possible.

Include variation in:

- Arrow positions and orientations.
- Arrow-tip visibility and difficult tip locations.
- Target position and apparent size.
- Lighting, shadows, and reflections.
- Camera angle and distance.
- Empty target frames with no arrow impact.
- Frames that could cause false detections.

### Dataset quality workflow

1. Gather images from the intended camera setup.
2. Remove exact and near-duplicate images where they add no useful variation.
3. Inspect blurred, corrupted, or incorrectly captured images.
4. Annotate all required objects consistently.
5. Review labels for missing objects, wrong classes, and inaccurate boxes or keypoints.
6. Split data into training, validation, and test sets.
7. Keep images from the same recording burst or physical scene in a single split when possible, reducing near-duplicate leakage between splits.
8. Save a versioned dataset manifest and class definitions.

Do not use the test set to repeatedly tune the model. Use the validation set for model selection and preserve a final test set for a less biased estimate of performance.

## Annotation Strategy

The object categories used in the detection pipeline include:

- **Target:** The target board or target region needed to establish scoring geometry.
- **Arrow:** The arrow object, where used by the chosen model.
- **Arrow tip:** The arrow's point of impact, which is especially important for scoring.

The annotation format must match the model task.

- **Detection model:** Uses bounding boxes and class IDs.
- **Segmentation model:** Uses object masks/polygons.
- **Pose/keypoint model:** Uses keypoint coordinates and visibility labels according to the dataset format.

Do not mix labels from different tasks without a deliberate conversion process. Confirm that class IDs, keypoint definitions, and label coordinates are consistent throughout the dataset. If arrow-tip precision is the main objective, review tip annotations carefully because small label errors can cause large scoring errors.

## Training YOLO Models

Several model approaches were evaluated during development, including detection, segmentation, and pose/keypoint approaches. The best task type depends on how the arrow tip is represented and how reliably it can be localized in real camera frames.

### Example dataset YAML

Create a dataset configuration file that matches the actual folder paths and annotation classes. Example only:

```yaml
path: /path/to/archery_dataset
train: images/train
val: images/val
test: images/test

names:
  0: target
  1: arrow_tip
```

If the model uses an arrow class, additional class entries must be included and all label IDs must match. For pose/keypoint training, use the appropriate pose dataset configuration rather than treating it as a detection dataset.

### Example detection training command

The following is an illustrative command; use a model checkpoint and image size appropriate to the selected task and available GPU memory.

```bash
yolo detect train model=yolo11s.pt data=archery.yaml imgsz=1280 epochs=100 batch=2 workers=0
```

For segmentation, use a compatible segmentation checkpoint and the `segment` task. For pose/keypoint work, use a compatible pose checkpoint and the `pose` task. Do not use a detection checkpoint for a different task without verifying compatibility.

### Training workflow

1. Establish a baseline with a small, suitable model.
2. Check that the dataset loads and labels display correctly.
3. Train the baseline and inspect validation metrics.
4. Review failure cases, especially arrow-tip localization and false detections.
5. Improve labels or dataset diversity where appropriate.
6. Experiment with model size, image resolution, confidence thresholds, and task type one change at a time.
7. Compare candidate models using the same held-out validation data.
8. Test promising candidates on real camera footage before selecting a deployment model.
9. Save the selected weights, dataset version, training settings, and evaluation results.

Higher image resolution may help with a small arrow tip but increases memory use and inference time. Larger models can improve accuracy but may not be suitable for a low-latency application. Select based on measured end-to-end performance, not only training metrics.

## Model Evaluation and Selection

Track more than a single overall metric. Useful measurements include:

- Precision and recall for each class.
- mAP at IoU 0.50 and across IoU thresholds.
- Arrow-tip localization error in pixels or normalized target coordinates.
- Percentage of true hits accepted correctly.
- False-hit rate during idle/no-shot frames.
- Duplicate-hit rate.
- End-to-end time from captured frame to score update.
- Performance under the actual projector and camera setup.

A model with a strong overall mAP can still be unsuitable if arrow-tip localization is unstable. Inspect per-class results and review false positives and missed detections. The final model should be selected using both validation metrics and real-world scoring reliability.

## Camera Pipeline

The camera pipeline should manage capture, preprocessing, inference, and debug output.

Recommended checks:

- Confirm the selected camera index.
- Read a frame and verify that it is non-empty.
- Log the actual frame width and height.
- Set supported camera properties and verify the returned resolution.
- Handle camera read failures without crashing the whole game.
- Keep the debug view separate from the projector display.
- Avoid unnecessary image copies and expensive drawing operations in the inference path.

### Basic camera smoke test

```python
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open camera")

ok, frame = camera.read()

if not ok or frame is None:
    raise RuntimeError("Camera opened, but no valid frame was received")

print("Frame shape:", frame.shape)

cv2.imshow("Camera Test", frame)
cv2.waitKey(0)

camera.release()
cv2.destroyAllWindows()
```

This is only a camera diagnostic example, not the full simulator pipeline. Change the camera index and resolution based on the hardware that has been tested.

## Detection and Arrow-Tip Localization

Scoring quality depends on converting model output into a reliable hit point.

A robust implementation should:

1. Select detections by the expected class ID.
2. Apply class-specific confidence thresholds.
3. Check that the target detection is plausible and sufficiently stable.
4. Obtain the arrow tip from the relevant detection, keypoint, or other verified method used by the model.
5. Convert the point into the same coordinate system used by the scoring engine.
6. Reject invalid points, including points outside the target where appropriate.
7. Require a suitable confirmation condition before recording a hit.

Bounding-box centers are not automatically accurate arrow-tip coordinates. If a box is used as a fallback, its limitations should be measured and documented. Avoid treating every frame-level prediction as a new arrow hit; the same arrow may appear in many consecutive frames.

## Scoring Engine

Keep scoring calculations independent from model inference. This makes the scoring logic easier to test without a camera or GPU.

### Distance from target center

Let the target center be \((c_x, c_y)\), and the detected hit point be \((x, y)\).

The Euclidean distance from the center is:

\[
d = \sqrt{(x-c_x)^2 + (y-c_y)^2}
\]

If the scoring geometry uses a circular target with effective radius \(R\), normalize the distance:

\[
r = \frac{d}{R}
\]

A normalized distance helps keep scoring consistent when the target appears at different pixel sizes. For an oval or perspective-distorted target, use calibrated geometry rather than assuming a perfect circle.

### Example ring mapping

The project configuration uses the following score bands:

| Normalized distance \(r\) | Score |
|---|---:|
| \(0 \le r \le 0.20\) | 10 |
| \(0.20 < r \le 0.40\) | 8 |
| \(0.40 < r \le 0.60\) | 6 |
| \(0.60 < r \le 0.80\) | 4 |
| \(0.80 < r \le 1.00\) | 2 |
| Outside the valid target radius | 0 or reject, according to game rules |

Treat these as the project's configured bands; calibrate them to the actual projected/physical target geometry. Decide explicitly how points exactly on a boundary are handled, and use the same rule everywhere.

### Scoring implementation example

```python
from math import sqrt

def calculate_score(hit_x, hit_y, center_x, center_y, radius):
    if radius <= 0:
        raise ValueError("Target radius must be positive")

    dx = hit_x - center_x
    dy = hit_y - center_y
    distance = sqrt(dx * dx + dy * dy)
    ratio = distance / radius

    if ratio <= 0.20:
        return 10
    elif ratio <= 0.40:
        return 8
    elif ratio <= 0.60:
        return 6
    elif ratio <= 0.80:
        return 4
    elif ratio <= 1.00:
        return 2
    else:
        return 0
```

This example assumes a circular target and that the hit point and target geometry use the same coordinate system. It should be adapted if the production implementation uses ellipse correction, a perspective transform, or a different rule for outside hits.

### Scoring engine test cases

Test at least:

- Target center returns the highest score.
- Points just inside and outside each ring boundary.
- Points on the outer boundary.
- Points outside the target.
- Invalid radius.
- Target geometry at different resolutions.
- Coordinates transformed from the camera to the target/projector coordinate system.

## Camera-to-Projector Alignment

Camera coordinates and projector coordinates are not necessarily interchangeable. The camera sees the physical target from its own viewpoint, while the projector draws graphics in the projector display's coordinate system.

A reliable alignment procedure is:

1. Fix the camera and projector physically.
2. Define the target center and ring geometry in the scoring coordinate system.
3. Determine whether scoring occurs in camera pixels, normalized target coordinates, or calibrated target coordinates.
4. If necessary, use calibration points to map camera coordinates to target/projector coordinates.
5. Validate alignment with known points across the entire target—not just the center.
6. Recalibrate if the camera, projector, target, or display resolution changes.

For a perspective mapping, a homography can map points between planes when sufficient corresponding calibration points are available. For an oval or non-circular view, use a suitable geometric correction rather than relying on a single radius. Keep calibration parameters in configuration rather than scattering constants across the code.

## Game-State Management

Treat the simulator as a state machine rather than a collection of unrelated conditions.

Possible states:

- **WAITING:** No active round; wait for a start trigger.
- **READY:** Reset round counters and prepare to accept hits.
- **ACTIVE:** Accept valid, confirmed hits.
- **ROUND_COMPLETE:** Stop accepting new hits after the configured arrow count.
- **RESULTS:** Keep the final score visible until the next round begins.

A state machine helps prevent scoring before a round starts, accepting extra arrows after the round ends, and resetting the score unexpectedly. Make state transitions explicit and test them independently from the model.

### Hit confirmation and duplicate protection

A single arrow can be detected in many adjacent frames. Use a combination of appropriate techniques, such as:

- A short confirmation window or temporal stability rule.
- A cooldown between accepted hits.
- A minimum distance from previously recorded hit points.
- An explicit state transition after a hit is accepted.
- A clear reset at the start of a new round.

Thresholds should be tuned with real footage. A cooldown alone can suppress legitimate fast shots, while a distance threshold alone can incorrectly merge two nearby arrows. Log why a detection was accepted or rejected so issues can be diagnosed.

## Arduino Integration

An Arduino button can send a serial signal to the Python application to start a round. Keep serial reading separate from inference so a slow or missing serial message does not block camera processing.

Document the actual serial protocol used by the project, including:

- Serial port configured on the deployment computer.
- Baud rate.
- Message format sent by the Arduino.
- Button debouncing behavior.
- Reconnection and error handling.
- How the application handles a start signal during an active round.

Do not assume a COM port is the same on every computer. Make it configurable and check that the serial device opens successfully. A keyboard shortcut can remain available for local testing if the production application supports it.

## Performance and Reliability

Measure performance on the actual deployment machine and camera. A GPU may accelerate inference, but capture, preprocessing, drawing, projector rendering, and serial handling can still add latency.

Recommended practices:

- Warm up the model before measuring latency.
- Measure capture, inference, scoring, and rendering separately.
- Use a stable camera resolution and avoid unnecessary resizing.
- Avoid loading model weights inside the main frame loop.
- Keep one owner for each camera and display resource.
- Keep model confidence thresholds and game timing values configurable.
- Log model errors, invalid frames, accepted hits, rejected hits, and round transitions.
- Ensure camera and serial resources are released on exit.
- Test idle scenes to measure false-positive behavior.

## Testing and Debugging

### Model tests

- Validate each class independently.
- Review confusion and missed detections.
- Inspect arrow-tip localization on difficult frames.
- Test frames with no arrow present.
- Compare model candidates on the same validation set.

### Scoring tests

- Test all ring boundaries.
- Test target-center and outside-target points.
- Test coordinate scaling and calibration.
- Confirm that score calculations are deterministic.

### Integration tests

- Confirm that the correct camera feed appears.
- Confirm that the debug window and projector window show the intended content.
- Start a round using the configured trigger.
- Confirm that one physical arrow produces at most one accepted score.
- Confirm that five accepted arrows end the round.
- Confirm that the final score remains visible.
- Start a new round and verify that counters reset correctly.
- Disconnect or disable the Arduino and verify that the application fails gracefully.

### Common issues

| Symptom | Possible areas to inspect |
|---|---|
| Black camera window | Camera index, driver, frame-read status, exposure, and another application using the camera |
| False hit before shooting | Idle-frame false positives, confidence thresholds, round state, and temporal confirmation |
| Same arrow scores more than once | Hit state transition, cooldown, duplicate-distance logic, and frame-to-frame tracking |
| Incorrect score | Arrow-tip localization, target center/radius, coordinate transforms, and ring boundaries |
| Projector output appears on laptop | Extended-display selection and projector window placement |
| Slow inference | Model size, input resolution, GPU use, frame copies, and rendering overhead |
| Different results after moving hardware | Camera/projector calibration and target geometry |

## Suggested Repository Structure

Adapt the layout to match the actual source code. Do not create empty modules just to match this example.

```text
archery-ai-simulator/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── game_config.yaml
│   └── dataset.yaml
├── datasets/
│   └── README.md
├── models/
│   └── README.md
├── src/
│   ├── camera/
│   │   ├── camera_capture.py
│   │   └── calibration.py
│   ├── detection/
│   │   ├── model_loader.py
│   │   └── detection_pipeline.py
│   ├── scoring/
│   │   └── scoring_engine.py
│   ├── game/
│   │   └── game_controller.py
│   ├── display/
│   │   └── projector_renderer.py
│   ├── hardware/
│   │   └── arduino_serial.py
│   └── run_game.py
├── tests/
│   ├── test_scoring.py
│   └── test_game_state.py
├── docs/
│   ├── dataset_and_training.md
│   ├── hardware_setup.md
│   └── calibration.md
└── assets/
    └── demo/
```

## Configuration Reference

Keep adjustable values in one configuration file where practical.

| Setting | Purpose |
|---|---|
| Camera index | Selects the camera device |
| Camera resolution | Sets the capture dimensions |
| Model path | Selects the trained weights |
| Confidence thresholds | Controls class-specific detection acceptance |
| Target geometry | Defines center, radius, and any calibration values |
| Ring boundaries | Defines normalized scoring bands |
| Maximum arrows | Number of accepted hits per round; configured as five |
| Duplicate-hit distance | Helps reject repeated hit detections |
| Hit cooldown/confirmation | Controls when a detection can count as a shot |
| Projector display position | Selects the second display/window placement |
| Arduino port and baud rate | Configures the serial trigger |

Record actual deployed values in the project configuration or a hardware-specific setup guide. Avoid exposing private credentials or machine-specific details unnecessarily.

## Reproducibility Checklist

Before publishing a model or release, record:

- [ ] Dataset version and image count.
- [ ] Annotation format and class definitions.
- [ ] Train/validation/test split method.
- [ ] Model task and checkpoint.
- [ ] Image size, batch size, epochs, and relevant training settings.
- [ ] Validation metrics for each class.
- [ ] Arrow-tip localization and false-hit results.
- [ ] Tested camera model, resolution, and placement.
- [ ] Tested GPU and package versions.
- [ ] Scoring geometry and calibration procedure.
- [ ] Arduino protocol and configuration.
- [ ] A short demo video or representative screenshots.
- [ ] Known limitations and steps to reproduce the result.

## Limitations and Future Improvements

Possible next steps include:

- Improve arrow-tip localization in difficult or partially occluded frames.
- Add more representative images for difficult lighting and camera positions.
- Use a repeatable calibration process for target geometry.
- Improve confidence and temporal filtering using recorded failure cases.
- Add automated tests for scoring and game-state transitions.
- Track inference latency and false-hit rates across releases.
- Provide a setup utility that validates the camera, GPU, projector, and Arduino before a game starts.

## Skills Demonstrated

- Python application development
- Computer vision with OpenCV
- YOLO model training and evaluation
- Dataset collection, annotation, and cleaning
- Object detection and arrow-tip localization
- Geometric scoring algorithms
- Real-time inference and latency debugging
- Camera and projector integration
- Arduino serial communication
- State-machine design and duplicate-hit handling
- Hardware-in-the-loop testing
- Iterative debugging and model selection

## Acknowledgements

Developed as a freelance AI/ML engineering project for **FunTime, Bangalore**.

---

**Repository note:** This README explains the architecture and workflow. Update commands, class IDs, model paths, hardware details, and configuration values to match the code and trained weights you actually publish. Do not commit private data, large raw datasets, or model weights unless you have permission and have chosen an appropriate Git LFS or release strategy.
