# AERIS — Acoustic Event Recognition & Interval Segmentation

> **Spectrogram-based acoustic event detection and temporal localization**

AERIS is a machine learning and computer vision pipeline for detecting and temporally localizing drone acoustic signatures from spectrogram images.

The project demonstrates how an acoustic detection problem can be transformed into a visual recognition problem by representing frequency-domain information as images and applying image-based feature extraction, temporal classification, and event segmentation.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Problem Definition](#problem-definition)
- [System Architecture](#system-architecture)
- [Dataset Structure](#dataset-structure)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Temporal Context](#temporal-context)
- [Machine Learning Model](#machine-learning-model)
- [Temporal Post-Processing](#temporal-post-processing)
- [Evaluation](#evaluation)
- [Memory-Efficient Design](#memory-efficient-design)
- [Output Format](#output-format)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Stack](#technical-stack)
- [Engineering Challenges](#engineering-challenges)
- [Security & Privacy](#security--privacy)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)

---

## Overview

Modern security and monitoring environments increasingly rely on multiple sensing modalities.

In a restricted security environment, conventional visual surveillance and radar systems may not always provide sufficient information for detecting small, low-signature aerial platforms. Acoustic sensing can provide an additional source of information.

Small rotorcraft generate characteristic acoustic patterns through their propulsion systems. Even when these signals are weak and partially masked by environmental noise, their frequency-domain structure can contain useful information.

AERIS explores a practical approach:

**convert acoustic information into spectrogram images and treat the detection problem as a computer vision and temporal localization task.**

The system estimates:

1. **The number of detected drone events**
2. **The temporal interval of each event**

---

## Motivation

The engineering problem is more challenging than ordinary image classification.

A conventional classifier might answer:

> Is there a drone in this spectrogram?

A temporal detection system must answer:

> How many drone events are present, and exactly when does each event start and end?

This distinction is important whenever event timing is part of the evaluation objective.

Example:

```text
Time
0s                    5s                    10s                   15s
|---------------------|---------------------|---------------------|

Event A
      ███████████

Event B
                                  ███████████

Event C
                                             █████████
```

The desired output is therefore a set of temporal intervals rather than a single class label.

---

## Problem Definition

Each input sample is a spectrogram representing a period of acoustic observation.

The model must estimate:

- Number of events
- Start time of each event
- End time of each event

For example:

```text
drone_count = 3

intervals =
0.52-1.75;
13.70-15.96;
16.42-17.62
```

The resulting prediction can be represented as:

```csv
image_file,drone_count,intervals
sample.png,3,0.52-1.75;13.70-15.96;16.42-17.62
```

This turns the task into a **temporal event detection and localization problem**.

---

## System Architecture

The complete pipeline is:

```text
                Spectrogram Image
                        |
                        v
              Image Preprocessing
                        |
                        v
            Spectral Feature Extraction
                        |
                        v
             Temporal Context Features
                        |
                        v
              Machine Learning Model
                        |
                        v
             Probability over Time
                        |
                        v
             Temporal Post-Processing
                        |
                        v
             Start / End Intervals
                        |
                        v
                 submission.csv
```

The architecture is modular so that feature extraction, model selection, and temporal post-processing can be improved independently.

---

## Dataset Structure

The original competition dataset is **not included** in this repository.

Expected local structure:

```text
dataset/
├── train_labels.csv
├── train/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── test/
    ├── image_101.png
    ├── image_102.png
    └── ...
```

The dataset contains training spectrograms, temporal annotations, and unlabeled test spectrograms.

The original dataset is not redistributed with this repository.

---

## Methodology

AERIS follows five major stages:

1. Represent the acoustic signal as a spectrogram.
2. Extract compact spectral features.
3. Construct local temporal context.
4. Predict drone probability for each temporal position.
5. Convert the probability curve into temporal intervals.

```text
Acoustic Information
        |
        v
    Spectrogram
        |
        v
Feature Engineering
        |
        v
Temporal Classifier
        |
        v
Probability Curve
        |
        v
Event Segmentation
        |
        v
Temporal Intervals
```

---

## Feature Engineering

A spectrogram is not treated as an ordinary photograph.

Its axes represent physical properties of the acoustic signal:

- Horizontal axis → time
- Vertical axis → frequency
- Pixel intensity → spectral energy

For each temporal position, AERIS extracts compact descriptors including:

- Mean spectral energy
- Spectral standard deviation
- Maximum spectral energy
- Magnitude statistics
- Spectral contrast characteristics
- Energy across multiple frequency bands
- Spectral centroid
- Spectral spread
- First temporal derivatives

These features provide both local spectral information and information about how the spectrum changes over time.

---

## Temporal Context

A single spectrogram column may not be sufficient to distinguish a drone signature from environmental noise.

AERIS therefore incorporates neighboring temporal positions.

```text
          Temporal Context

    t-2     t-1      t      t+1     t+2
     |       |       |       |       |
     v       v       v       v       v
   +-------------------------------------+
   |          Feature Vector             |
   +-------------------------------------+
```

This allows the model to learn temporal continuity rather than relying exclusively on instantaneous spectral appearance.

---

## Machine Learning Model

The current implementation uses:

**`HistGradientBoostingClassifier`**

The classifier estimates:

```text
P(drone | spectral + temporal features)
```

for each temporal position.

The output is a one-dimensional probability signal:

```text
Probability
1.0 |             ████
    |            ██████
    |           ███████
0.5 |───────████████████──────
    |
0.0 +--------------------------------
          Time -------------------->
```

The probability signal is then converted into temporal event intervals.

---

## Temporal Post-Processing

Raw predictions are not directly used as the final submission.

AERIS applies:

### 1. Gaussian Smoothing

Small frame-to-frame fluctuations are reduced.

### 2. Thresholding

The probability curve is converted into an active/inactive temporal mask.

### 3. Gap Bridging

Very small inactive gaps inside an otherwise continuous event can be merged.

### 4. Minimum Duration Filtering

Very short detections are removed as likely noise.

```text
Model Probability
       |
       v
Gaussian Smoothing
       |
       v
Thresholding
       |
       v
Temporal Segmentation
       |
       v
Gap Bridging
       |
       v
Minimum Duration Filter
       |
       v
Final Intervals
```

---

## Evaluation

The competition evaluates temporal localization using **Temporal Intersection over Union (tIoU)**.

For predicted interval `P` and ground-truth interval `G`:

```text
             Temporal Intersection
tIoU = -------------------------------
               Temporal Union
```

A perfect match has:

```text
tIoU = 1.0
```

A completely non-overlapping prediction has:

```text
tIoU = 0.0
```

The overall competition score accounts for:

- True Positives
- False Positives
- False Negatives
- Temporal overlap quality

Conceptually:

```text
Score =
100 × Σ matched tIoU
      ----------------
        TP + FP + FN
```

This creates an important trade-off:

- Missing an event produces a false negative.
- Inventing an event produces a false positive.
- Detecting an event at the wrong time reduces its tIoU.

Therefore, the system must balance **sensitivity, precision, and temporal accuracy**.

---

## Memory-Efficient Design

Feature expansion can make a relatively small image dataset consume several gigabytes of RAM.

A naïve implementation may attempt:

```text
All Images
    |
    v
All Features
    |
    v
One Huge X Matrix
    |
    v
RAM Exhaustion
```

AERIS instead uses a streaming strategy:

```text
Image 1
   |
Feature Extraction
   |
Sample Training Columns
   |
Release Memory
   |
Image 2
   |
Feature Extraction
   |
Sample Training Columns
   |
Release Memory
   |
...
```

Only a limited number of positive and negative temporal samples are retained from each training image.

Validation and test inference are also performed one image at a time.

This design is intended to make the pipeline practical on machines with **8GB RAM**.

Current memory-control parameters include:

```python
MAX_POS_PER_IMAGE = 120
MAX_NEG_PER_IMAGE = 180
CONTEXT = 2
```

---

## Training Strategy

The validation split is performed at the **image level**, rather than randomly splitting individual temporal columns.

Incorrect:

```text
One spectrogram
    |
    +---- columns -> Training
    |
    +---- columns -> Validation
```

Preferred:

```text
Spectrogram A ─────────> Training
Spectrogram B ─────────> Training
Spectrogram C ─────────> Validation
Spectrogram D ─────────> Validation
```

All temporal samples originating from the same spectrogram remain in the same partition.

This provides a more realistic estimate of generalization.

---

## Hyperparameter Optimization

Temporal post-processing parameters are evaluated on the validation set.

The search includes:

| Parameter | Purpose |
|---|---|
| Threshold | Controls classification sensitivity |
| Sigma | Controls temporal smoothing |
| Minimum duration | Removes very short detections |
| Maximum gap | Controls small-gap merging |

The configuration producing the highest validation score is selected for final inference.

---

## Output Format

The final submission contains exactly three columns:

| Column | Description |
|---|---|
| `image_file` | Test image filename |
| `drone_count` | Number of detected events |
| `intervals` | Semicolon-separated start-end intervals |

Example:

```csv
image_file,drone_count,intervals
sample_a1b2c3d4.png,1,10.99-12.84
sample_b2c3d4e5.png,1,2.41-4.53
sample_c3d4e5f6.png,0,
sample_d4e5f7g8.png,3,0.52-1.75;13.70-15.96;16.42-17.62
sample_e5f6g7h8.png,2,0.34-1.35;11.03-12.19
```

The notebook also validates the generated CSV before finishing.

---

## Project Structure

```text
aeris/
├── README.md
├── drone.ipynb
├── requirements.txt
├── submission.csv
└── src/
    └── ...
```

| File | Purpose |
|---|---|
| `README.md` | Project documentation |
| `drone.ipynb` | Complete training and inference pipeline |
| `requirements.txt` | Python dependencies |
| `submission.csv` | Example generated submission |
| `src/` | Optional reusable source modules |

---

## Installation

```bash
git clone <repository-url>
cd aeris

pip install -r requirements.txt
```

Place the dataset in:

```text
dataset/
├── train_labels.csv
├── train/
└── test/
```

Then open:

```text
drone.ipynb
```

and run all cells.

---

## Usage

The complete workflow is contained in the notebook.

Run:

```text
Run All
```

The notebook performs:

1. Dataset validation
2. Label parsing
3. Dataset inspection
4. Image-level train/validation split
5. Feature extraction
6. Memory-efficient training sample construction
7. Model training
8. Validation inference
9. Temporal parameter optimization
10. Final model training
11. Test inference
12. Submission generation
13. Submission format validation

At the end:

```text
submission.csv
```

is generated automatically.

---

## Technical Stack

### Programming

- Python
- NumPy
- Pandas

### Computer Vision / Image Processing

- Pillow
- Spectrogram image analysis
- Frequency-band statistics
- Temporal image feature extraction

### Signal Processing

- Frequency-domain representation
- Spectral energy analysis
- Spectral centroid
- Spectral spread
- Temporal smoothing

### Machine Learning

- Scikit-learn
- Histogram-based Gradient Boosting
- Supervised temporal classification

### Optimization

- Validation-based parameter search
- Temporal threshold optimization
- Post-processing optimization

### Evaluation

- Temporal IoU
- TP / FP / FN analysis
- Competition-style temporal scoring

---

## Engineering Challenges

### Weak Acoustic Signatures

Drone-related spectral patterns may be partially hidden by:

- Wind
- Birds
- Environmental noise
- Other mechanical sources
- Background acoustic activity

The classifier must distinguish meaningful spectral structures from background variation.

### Temporal Localization

Detecting an event is only part of the problem.

The model must also estimate:

```text
Start ───────────── End
```

with sufficient temporal precision.

### Multiple Events

A single observation window may contain multiple independent events.

The output therefore supports multiple temporal intervals rather than assuming one event per image.

### False Positives

Over-sensitive detection can create artificial intervals.

These false positives directly affect the competition metric.

### Memory Constraints

Feature expansion can make a relatively small image dataset consume several gigabytes of RAM.

The streaming design addresses this constraint explicitly.

---

## Security & Privacy

This repository intentionally excludes the original dataset and sensitive operational information.

The public repository contains:

- Source code
- Notebook
- Documentation
- Non-sensitive example output

It does **not** contain:

- Raw acoustic recordings
- Restricted imagery
- Real-world deployment coordinates
- Sensor deployment configurations
- Operational security parameters
- Credentials or access tokens
- Sensitive infrastructure information

The project is presented as a research and engineering implementation of spectrogram-based acoustic event detection.

---

## Broader Applications

The underlying methodology is not limited to drone detection.

The same architecture can be adapted to:

- Civilian drone monitoring
- Industrial acoustic monitoring
- Machinery fault detection
- Wildlife and ecological monitoring
- Rotorcraft detection
- Smart sensor networks
- Remote acoustic monitoring
- General acoustic event segmentation

The central idea remains:

```text
Acoustic Signal
      |
      v
Spectral Representation
      |
      v
Visual Feature Extraction
      |
      v
Machine Learning
      |
      v
Temporal Event Localization
```

---

## Limitations

The current implementation has several limitations.

### Feature Representation

The model uses engineered spectral statistics rather than a deep neural network trained directly on spectrogram images.

### Temporal Resolution

The final temporal resolution is constrained by the number of temporal columns in the input spectrogram.

### Domain Shift

Changes in microphone characteristics, environmental conditions, background noise, recording setup, or spectrogram generation parameters may affect generalization.

### Model Capacity

Gradient boosting provides an efficient baseline, but more expressive temporal architectures may capture complex acoustic patterns more effectively.

---

## Future Work

Potential extensions include:

- CNN-based spectrogram encoders
- CRNN architectures
- 1D temporal convolution
- Vision Transformers
- Transformer-based temporal segmentation
- Self-supervised acoustic representation learning
- Spectrogram-domain augmentation
- Noise-robust feature extraction
- Confidence calibration
- Ensemble models
- Higher-resolution temporal localization
- More advanced event-matching strategies

A promising direction is a hybrid architecture combining a visual encoder with an explicit temporal model:

```text
Spectrogram
    |
    v
CNN / Vision Encoder
    |
    v
Temporal Representation
    |
    v
Transformer / TCN
    |
    v
Temporal Event Segmentation
```

---

## Key Takeaway

AERIS demonstrates an important engineering principle:

> **A difficult signal-processing problem can sometimes become a tractable computer-vision problem by choosing the right representation.**

Instead of processing large volumes of raw audio directly, the acoustic information is represented as spectrogram images.

The resulting system combines:

```text
Signal Processing
        +
Computer Vision
        +
Feature Engineering
        +
Machine Learning
        +
Temporal Modeling
        +
Optimization
```

to produce an end-to-end temporal detection pipeline.

The final objective is not simply:

> **"Is there a drone?"**

but:

> **"How many events are present, and exactly when does each event occur?"**

That distinction turns the task from conventional classification into a temporal detection and localization problem.

---

## Author

**Sepehr**

Bachelor's Student — Mechatronics Engineering

Interested in:

- Robotics
- Machine Learning
- Computer Vision
- Signal Processing
- Autonomous Systems
- Programming
- Engineering Problem Solving

---

## License

This repository is intended for educational and research purposes.

See `LICENSE` for the applicable terms.
