# Dynamic Eye Blink Analysis as an Anti-Spoofing Strategy in Facial Recognition Technology

This bachelor's thesis project implements a comprehensive security system that combines robust blink detection with deepfake detection techniques. The system uses multiple biometric and behavioral cues to verify liveness and detect potential spoofing attacks in facial recognition technology (FRT) systems.

## Implementation Details

The applet architecture is implemented as a modular pipeline with four key layers:

1. **Sensor Layer**: Utilizes OpenCV's VideoCapture interface for real-time video acquisition from webcam.

2. **Processing Layer**: Prepares input data through grayscale conversion and employs dlib's HOG-based face detector.

3. **Analysis Layer**: Uses dlib's 68-point facial landmark predictor for precise feature mapping and implements dynamic blink detection through EAR calculation.

4. **Decision Layer**: Implements multi-modal spoofing detection including microsaccade analysis, photometric analysis, and interactive challenge-response.

### Eye Aspect Ratio (EAR) Formula Implementation

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

where p1-p6 are facial landmarks around the eye (landmarks 36-41 for left eye, 42-47 for right eye).

### Interactive Verification Challenges

| Challenge Type | Parameters | Security Purpose |
|---------------|------------|------------------|
| Single Blink | 1 blink in 12s | Baseline liveness check |
| Double Quick Blink | 2 blinks 12s max time | Defeats video splicing |
| Triple Slow Blink | 3 blinks 15s max time | Defeats video splicing |
| Eyes Open | Random duration (3-5s) | Detects printed photos |
| Random Count Blinks | 3-5 blinks (random), self-paced | Verifies natural blink variability |
| Rhythm Blinks | 3 blinks synced to rhythm, 0.4s tolerance | Detects algorithmic blink generation |

## Table of Contents

- [Research Results Summary](#research-results-summary)
- [System Components](#system-components)
  - [Blink Detection](#blink-detection)
  - [Microsaccade Analysis](#microsaccade-analysis)
  - [Interactive Tests](#interactive-tests)
  - [Photo Attack Detection](#photo-attack-detection)
  - [Deepfake Detection](#deepfake-detection)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Implementation Details](#implementation-details)
- [Challenges and Future Work](#challenges-and-future-work)
- [Dependencies](#dependencies)

## System Components

### Blink Detection
- Uses dynamic EAR thresholds adapted to individual users
- Tracks both eye closure and reopening phases
- Implements cooldown periods to prevent false positives
- Detects double blinks and verifies blink symmetry

### Microsaccade Analysis
- Detects and analyzes natural eye movements
- Calibrates to individual movement patterns
- Identifies unnatural movement patterns indicative of spoofing
- Tracks movement velocity and direction variability

### Interactive Tests
- Presents randomized challenge-response tests:
  - Single/double/triple blink tests
  - Eyes-open duration tests
  - Rhythm-based blink tests
  - Precise timing challenges
- Provides real-time feedback and instructions

### Photo Attack Detection
- Analyzes texture variations in eye regions
- Detects screen artifacts and reflections
- Monitors face movement patterns
- Checks for unnatural blink timing and duration

### Deepfake Detection
- Verifies eye reflections and corneal patterns
- Analyzes eyelid movement dynamics
- Checks for unnatural EAR patterns
- Detects inconsistencies in microsaccades

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VladMalosev/Blink-Analysis-FRT.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the facial landmark predictor:
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   mv shape_predictor_68_face_landmarks.dat dat/
   ```

## Usage

Run the system with:
```bash
python frt.py
```

## Configuration

Key configuration parameters (in `config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| BASELINE_FRAMES | 30 | Frames to calculate initial EAR baseline |
| BLINK_COOLDOWN | 0.5s | Minimum time between blink detections |
| SMOOTHING_WINDOW | 5 | Frames for EAR smoothing |
| EAR_CHANGE_THRESHOLD | 0.25 | Threshold for abrupt EAR changes |
| MIN_BLINK_DURATION | 0.07s | Minimum valid blink duration |
| MAX_BLINK_DURATION | 0.5s | Maximum valid blink duration |
| DOUBLE_BLINK_THRESHOLD | 0.8s | Max time between double blinks |

## Technical Details

### Eye Aspect Ratio (EAR) Calculation

The EAR is calculated as:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

where p1-p6 are facial landmarks around the eye.

### Microsaccade Detection

Uses velocity thresholding with adaptive thresholds based on:

- Median absolute deviation of eye positions
- Directional consistency checks
- Amplitude-duration relationships

### Photo Attack Detection

Combines multiple indicators:

- Texture analysis using FFT
- Blink pattern analysis
- Face movement variance
- Screen reflection detection

## Dependencies

- Python 3.7+
- OpenCV
- dlib
- numpy
- scipy
- scikit-image