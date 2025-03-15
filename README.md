# Blink Detection Using Eye Aspect Ratio (EAR)

This project implements a blink detection system using the Eye Aspect Ratio (EAR) and a dynamic threshold based on a baseline EAR. The system is designed to be robust to individual differences in eye shape and size, using facial landmarks to detect blinks in real time.

## Table of Contents

- [How It Works](#how-it-works)
  - [Eye Aspect Ratio (EAR)](#eye-aspect-ratio-ear)
  - [Baseline EAR](#baseline-ear)
  - [Dynamic Threshold](#dynamic-threshold)
  - [Smoothing EAR Values](#smoothing-ear-values)
  - [Cooldown Period](#cooldown-period)
  - [Eye Movement Detection](#eye-movement-detection)
  - [EAR Consistency Check](#ear-consistency-check)
  - [Reflection Analysis](#reflection-analysis)
- [Code Overview](#code-overview)
- [Key Variables](#key-variables)
- [Usage](#usage)
- [Dependencies](#dependencies)

## How It Works

### Eye Aspect Ratio (EAR)

The EAR is a metric used to determine whether the eyes are open or closed. It is calculated using the vertical and horizontal distances between specific facial landmarks around the eyes.

Formula:

![img.png](img.png)

Where:

- \( P1, P2, P3, P4, P5, P6 \) are the facial landmarks around the eye.
- \( \|A - B\| \) represents the Euclidean distance between two points.

The EAR is calculated for both the left and right eyes, and the average EAR is used for blink detection.

### Baseline EAR

The baseline EAR represents the average EAR when the eyes are open and not blinking. It is calculated during the first `BASELINE_FRAMES` (e.g., 30 frames).

![img_1.png](img_1.png)

This running average smooths out small variations and establishes a reliable baseline EAR.

### Dynamic Threshold

The dynamic threshold determines whether a blink has occurred. It is calculated as a percentage of the baseline EAR:

![img_2.png](img_2.png)

If the average EAR drops below 85% of the baseline EAR, a blink is detected.

### Smoothing EAR Values

The EAR values can fluctuate due to small facial movements. To mitigate noise, the EAR values are smoothed over a small window of frames.

Smoothing formula:

![img_3.png](img_3.png)

Where `EAR_history` is a list containing the last few EAR values.

### Cooldown Period

A cooldown period prevents multiple detections for a single blink. After detecting a blink, the system waits for a short period before registering another blink.

### Eye Movement Detection

To account for eye movements, we track the center of the eye over time. The eye movement detection method calculates the variance in eye position over recent frames:

![img_4.png](img_4.png)

If the normalized variance exceeds a threshold (e.g., 0.15), significant movement is detected.

### EAR Consistency Check

A consistency check ensures that sudden, irregular EAR changes do not lead to false positives. It compares differences between recent EAR values:

![img_5.png](img_5.png)

If the largest EAR change exceeds a predefined threshold (e.g., 0.15), a blink is flagged.

### Reflection Analysis

To check for potential obstructions (e.g., glare or reflections), the eye region is processed with thresholding:

![img_6.png](img_6.png)

If reflections are minimal, the detection remains valid.

## Code Overview

The code consists of the following key components:

- **EAR Calculation:** `calculate_ear` computes the EAR for a given eye.
- **Baseline EAR Calculation:** `calculate_baseline_ear` establishes the baseline EAR over the first 30 frames.
- **Smoothing:** `smooth_ear` averages the last few EAR values to reduce noise.
- **Blink Detection:** The main loop captures video frames, detects faces, calculates the EAR, and checks against the dynamic threshold.
- **Eye Movement Detection:** `check_eye_movement` tracks eye position history and detects significant movement.
- **EAR Consistency Check:** `check_ear_consistency` prevents erratic EAR changes from falsely triggering blinks.
- **Reflection Analysis:** `analyze_reflections` checks for excessive light reflections in the eye region.
- **Display:** The system overlays the EAR values, blink count, baseline EAR, and dynamic threshold on the video feed.

## Key Variables

- `baseline_ear`: The average EAR when the eyes are open.
- `dynamic_threshold`: 85% of the baseline EAR, used to detect blinks.
- `ear_history`: A list of the last 3 EAR values for smoothing.
- `cooldown_counter`: Prevents multiple detections for a single blink.
- `eye_position_history`: Tracks recent eye positions to monitor movement.
- `EYE_MOVEMENT_THRESHOLD`: Defines the threshold for significant eye movement.
- `ear_history_blink`: Stores recent EAR values for consistency checks.
- `EAR_CHANGE_THRESHOLD`: Defines the threshold for EAR consistency validation.

## Usage

1. Install the required dependencies:

```bash
pip install opencv-python dlib scipy numpy
```

2. Download the `shape_predictor_68_face_landmarks.dat` file from dlib's website and place it in the project directory.

3. Run the script:

```bash
python blink_detection.py
```

4. Follow the on-screen instructions to calculate the baseline EAR and start detecting blinks.

## Dependencies

- `opencv-python`: For video capture and frame processing.
- `dlib`: For face detection and facial landmark detection.
- `scipy`: For distance calculations.
- `numpy`: For numerical operations and EAR smoothing.

