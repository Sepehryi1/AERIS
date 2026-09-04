\# Drone Acoustic Detection from Spectrograms

\> \*\*Computer Vision · Machine Learning · Signal Processing · Temporal Detection\*\*

An end-to-end machine learning pipeline for detecting and temporally localizing drone acoustic signatures from spectrogram images.

The project demonstrates how an acoustic detection problem can be transformed into a computer vision problem by representing frequency-domain information as images and applying image-based feature extraction and temporal machine learning.

\---

\#\# 1\. Background

Modern security environments increasingly rely on multi-modal sensing systems.

Consider a highly restricted military security zone where conventional visual surveillance and radar-based monitoring may be insufficient for detecting small, low-signature aerial platforms. In such environments, acoustic sensing can provide an additional sensing modality.

Small rotorcraft generate characteristic acoustic signatures through their propulsion systems. Although these signatures can be extremely weak compared with environmental noise, their frequency patterns can contain useful information.

Instead of processing large volumes of raw audio directly with computationally expensive audio-processing pipelines, this project explores an alternative approach:

\*\*convert acoustic information into spectrogram images and treat the problem as a computer vision and temporal detection task.\*\*

The objective is not only to determine whether a drone is present, but also to estimate:

1\. \*\*How many drone signatures are present\*\*  
2\. \*\*When each signature starts\*\*  
3\. \*\*When each signature ends\*\*

This makes the problem substantially more challenging than ordinary image classification.

\---

\# 2\. Problem Definition

Each input sample is a spectrogram representing a period of acoustic observation.

The model must transform the spectrogram into a set of temporal intervals:

\`\`\`text  
Drone 1:  0.52 ───────── 1.75  
Drone 2:                 13.70 ─────── 15.96  
Drone 3:                              16.42 ─── 17.62

The final prediction for each image is represented as:

image\_file,drone\_count,intervals  
sample.png,3,0.52-1.75;13.70-15.96;16.42-17.62

Therefore, the task is fundamentally a combination of:

* Signal representation  
* Image processing  
* Feature engineering  
* Temporal classification  
* Event segmentation  
* Post-processing  
* Multi-object temporal localization

---

# **3\. Why Spectrograms?**

A spectrogram represents how the frequency content of a signal changes over time.

Conceptually:

Raw Audio  
    │  
    ▼  
Frequency Analysis  
    │  
    ▼  
Spectrogram  
    │  
    ├──────────────► Time  
    │  
    ▼  
Frequency

Once the signal is represented as a spectrogram, acoustic events become visual structures.

A drone may produce characteristic patterns such as:

* Persistent frequency components  
* Harmonic structures  
* Energy concentration in specific frequency bands  
* Temporal continuity  
* Characteristic changes in spectral energy

This enables the use of computer vision and machine learning techniques without requiring the entire solution to operate directly on raw audio.

---

# **4\. Core Challenge**

The problem is not simple image classification.

A conventional classifier could answer:

> "Does this image contain a drone?"

This project requires a much more precise answer:

> "At exactly which points in time does each drone signature exist?"

For example:

Time  
0s                    5s                    10s                   15s  
│─────────────────────│─────────────────────│─────────────────────│

Drone A  
      ███████████

Drone B  
                                  ███████████

Drone C  
                                             █████████

The system therefore performs **temporal segmentation** rather than simple classification.

---

# **5\. System Architecture**

The complete pipeline can be summarized as:

                Spectrogram Image  
                         │  
                         ▼  
                Image Preprocessing  
                         │  
                         ▼  
              Spectral Feature Extraction  
                         │  
                         ▼  
              Temporal Feature Construction  
                         │  
                         ▼  
               Machine Learning Model  
                         │  
                         ▼  
             Probability over Time  
                         │  
                         ▼  
              Temporal Post-processing  
                         │  
                         ▼  
             Drone Start / End Intervals  
                         │  
                         ▼  
                  submission.csv  
---

# **6\. Dataset Structure**

The original competition dataset is intentionally **not included** in this repository.

The expected local dataset structure is:

dataset/  
│  
├── train\_labels.csv  
│  
├── train/  
│   ├── image\_001.png  
│   ├── image\_002.png  
│   └── ...  
│  
└── test/  
    ├── image\_101.png  
    ├── image\_102.png  
    └── ...

The dataset contains:

* Training spectrogram images  
* Training labels  
* Drone counts  
* Temporal intervals  
* Unlabeled test spectrograms

No original dataset files are redistributed with this repository.

---

# **7\. Label Representation**

Each training sample contains temporal annotations.

For example:

drone\_count \= 3

intervals \=  
0.52-1.75;  
13.70-15.96;  
16.42-17.62

These annotations are converted into a temporal binary representation.

If the spectrogram contains `W` temporal columns:

0  1  2  3  4  5  6  7  8  9 ...  
│  │  │  │  │  │  │  │  │  │

the intervals are mapped onto these temporal positions:

0  1  1  1  0  0  0  1  1  0 ...

This converts the original event-detection problem into a supervised temporal classification problem.

---

# **8\. Feature Engineering**

One of the important aspects of the project is that the spectrogram is not treated as an ordinary RGB photograph.

The image contains physical information about the frequency distribution of the acoustic signal.

For every temporal position, the system extracts compact spectral descriptors such as:

* Mean spectral energy  
* Spectral variance  
* Maximum energy  
* Magnitude statistics  
* Spectral saturation / contrast characteristics  
* Energy in multiple frequency bands  
* Spectral centroid  
* Spectral spread  
* Temporal derivatives

These features describe the local acoustic state of the system.

---

# **9\. Temporal Context**

A single spectrogram column may not contain enough information to distinguish a drone signature from environmental noise.

Therefore, neighboring temporal positions are incorporated into the feature vector.

Conceptually:

            Temporal Context

       t-2     t-1      t      t+1     t+2  
        │       │       │       │       │  
        ▼       ▼       ▼       ▼       ▼  
      ┌───────────────────────────────┐  
      │        Feature Vector         │  
      └───────────────────────────────┘

This allows the classifier to recognize not only the spectral structure at a specific instant, but also how that structure evolves over time.

---

# **10\. Machine Learning Model**

The implementation uses a gradient-boosted tree classifier:

HistGradientBoostingClassifier

The classifier receives a temporal feature vector and estimates:

P(drone | spectral features)

for every temporal position.

The result is therefore a probability curve:

Probability  
1.0 │             ████  
    │            ██████  
    │           ███████  
0.5 │───────████████████──────  
    │  
0.0 └──────────────────────────────  
       Time ──────────────────────►

This probability signal is subsequently converted into temporal intervals.

---

# **11\. Temporal Post-processing**

Raw model predictions are not directly used as the final answer.

The probability curve is processed using several temporal operations:

### **Smoothing**

Small fluctuations are reduced using Gaussian smoothing.

### **Thresholding**

The probability curve is converted into an active/inactive temporal mask.

Probability  
     │  
1.0  │       █████████  
     │      ███████████  
0.5  │──────████████████──────  
     │  
0.0  └─────────────────────────

### **Gap Bridging**

Very small gaps inside an otherwise continuous event can be merged.

### **Minimum Duration**

Very short detections are rejected as probable noise.

The final result becomes:

Probability Curve  
       │  
       ▼  
Smoothing  
       │  
       ▼  
Threshold  
       │  
       ▼  
Temporal Segmentation  
       │  
       ▼  
\[Start, End\] Intervals  
---

# **12\. Validation Strategy**

A major concern in temporal detection problems is data leakage.

If individual temporal columns from the same spectrogram are randomly distributed between training and validation sets, the evaluation can become unrealistically optimistic.

Therefore, the split is performed at the **image level**.

Image A ───────────────► Train

Image B ───────────────► Train

Image C ───────────────► Validation

Image D ───────────────► Validation

All temporal samples belonging to an image remain in the same partition.

This provides a more realistic estimate of generalization to unseen spectrograms.

---

# **13\. Evaluation Metric**

The competition evaluates temporal localization using:

**Temporal Intersection over Union (tIoU)**

For two intervals:

Prediction:  ────────────────  
Ground Truth:     ────────────────

the temporal IoU is:

                Intersection  
tIoU \= ─────────────────────────────  
                 Union

A perfect prediction produces:

tIoU \= 1.0

while completely non-overlapping intervals produce:

tIoU \= 0.0

The competition metric additionally penalizes:

* False Positives  
* False Negatives  
* Poor temporal alignment

This makes the task substantially more demanding than simply counting detections.

---

# **14\. Why Temporal Precision Matters**

Consider two predictions:

Ground Truth:  
10.00 ───────────────── 12.00

Prediction A:  
10.01 ───────────────── 11.99

Prediction B:  
 8.00 ───────────────── 14.00

Both predictions identify the same general event.

However, Prediction A has significantly better temporal localization.

This is important in any system where the timing of an event matters.

Therefore, the project focuses not only on:

> **Detection**

but also on:

> **Temporal localization**

---

# **15\. Memory-Efficient Implementation**

The original dataset can be hundreds of megabytes in size, while spectrogram feature matrices can become substantially larger than the image files themselves.

A naïve implementation might attempt:

All Images  
    ↓  
All Features  
    ↓  
One Giant X Matrix  
    ↓  
RAM

This can easily cause memory exhaustion on an 8GB machine.

The implemented pipeline instead uses a streaming strategy:

Image 1  
   ↓  
Extract Features  
   ↓  
Train Samples  
   ↓  
Release Memory

Image 2  
   ↓  
Extract Features  
   ↓  
Train Samples  
   ↓  
Release Memory

...

Only a limited number of temporal samples are retained from each image for training.

Validation and test inference are also performed **one image at a time**.

This dramatically reduces peak memory consumption.

---

# **16\. 8GB RAM Design**

The memory-aware implementation specifically avoids:

X\_all \= np.vstack(all\_features)

for the complete dataset.

Instead:

                   ┌──────────────┐  
Image ─────────────►│ Feature      │  
                    │ Extraction   │  
                    └──────┬───────┘  
                           │  
                           ▼  
                    Small Sample Set  
                           │  
                           ▼  
                         Model  
                           │  
                           ▼  
                    Release Memory

This design makes the pipeline significantly more practical on machines with limited RAM.

---

# **17\. Hyperparameter Optimization**

The temporal post-processing parameters are optimized on the validation set.

The search includes parameters such as:

* Classification threshold  
* Gaussian smoothing strength  
* Minimum event duration  
* Maximum temporal gap

Conceptually:

Model Probability  
       │  
       ├── Threshold  
       ├── Smoothing  
       ├── Min Duration  
       └── Gap Bridging  
               │  
               ▼  
          tIoU Score

The configuration producing the strongest validation score is then used for final test inference.

---

# **18\. Final Training Pipeline**

After validation and parameter selection:

Training Dataset  
       │  
       ▼  
Feature Extraction  
       │  
       ▼  
Sampled Training Data  
       │  
       ▼  
Final Model  
       │  
       ▼  
Unseen Test Spectrograms  
       │  
       ▼  
Probability Curves  
       │  
       ▼  
Temporal Segmentation  
       │  
       ▼  
submission.csv  
---

# **19\. Output Format**

The final output follows the required competition format:

image\_file,drone\_count,intervals  
sample\_a1b2c3d4.png,1,10.99-12.84  
sample\_b2c3d4e5.png,1,2.41-4.53  
sample\_c3d4e5f6.png,0,  
sample\_d4e5f7g8.png,3,0.52-1.75;13.70-15.96;16.42-17.62  
sample\_e5f6g7h8.png,2,0.34-1.35;11.03-12.19

The required columns are:

image\_file  
drone\_count  
intervals  
---

# **20\. Repository Contents**

.  
├── README.md  
├── drone.ipynb  
├── submission.csv  
├── requirements.txt  
└── src/

### **`drone.ipynb`**

The complete reproducible machine learning pipeline:

* Dataset inspection  
* Data validation  
* Label parsing  
* Feature engineering  
* Train/validation split  
* Model training  
* Temporal post-processing  
* Validation  
* Hyperparameter search  
* Final training  
* Test inference  
* Submission generation  
* Output validation

### **`submission.csv`**

Example output generated by the pipeline.

### **`requirements.txt`**

Python dependencies required to reproduce the experiment.

---

# **21\. Reproducibility**

The notebook is designed to run from start to finish using:

Run All

The expected workflow is:

git clone \<repository\>  
cd drone-spectrogram-detection

pip install \-r requirements.txt

Then place the competition dataset in:

dataset/  
├── train\_labels.csv  
├── train/  
└── test/

and execute:

drone.ipynb

The final submission will be generated automatically as:

submission.csv  
---

# **22\. Technical Stack**

The project combines several areas of computer science and engineering:

### **Programming**

* Python  
* NumPy  
* Pandas

### **Image Processing**

* Pillow  
* Spectrogram feature extraction  
* Spatial and spectral statistics

### **Machine Learning**

* Scikit-learn  
* Gradient Boosting  
* Supervised temporal classification

### **Signal Processing**

* Frequency-domain representation  
* Spectral energy analysis  
* Temporal smoothing  
* Spectral centroid and spread

### **Optimization**

* Hyperparameter search  
* Temporal threshold optimization  
* Post-processing optimization

### **Evaluation**

* Temporal IoU  
* TP / FP / FN analysis  
* Competition-style scoring

---

# **23\. Engineering Challenges**

Several engineering challenges make this project more interesting than a standard classification problem.

### **23.1 Weak Signals**

Drone acoustic signatures can be hidden inside environmental noise.

### **23.2 Temporal Precision**

Detecting an event is not enough.

The beginning and end of the event must also be estimated accurately.

### **23.3 False Positives**

Environmental acoustic structures can resemble drone signatures.

Excessive sensitivity can therefore produce hallucinated detections.

### **23.4 Multiple Events**

A single observation window may contain multiple drone events.

The model therefore needs to produce multiple independent temporal intervals.

### **23.5 Memory Constraints**

Large-scale feature extraction can create matrices significantly larger than the original dataset.

The implementation therefore uses streaming and sampling strategies to operate within an 8GB RAM environment.

---

# **24\. Security and Privacy Considerations**

The original dataset and any potentially sensitive operational information are intentionally excluded from this repository.

This public repository contains only:

* Source code  
* Notebook  
* Generic documentation  
* Non-sensitive example outputs

No operational coordinates, sensor deployment information, raw acoustic recordings, restricted imagery, credentials, or sensitive infrastructure information are included.

The project is presented as a research and engineering demonstration of acoustic event detection using machine learning and computer vision techniques.

---

# **25\. Broader Applications**

Although the motivating scenario involves security monitoring, the underlying technology is not limited to military environments.

The same architecture can be adapted to:

* Civilian drone detection  
* Critical infrastructure monitoring  
* Industrial acoustic monitoring  
* Wildlife and ecological monitoring  
* Rotorcraft detection  
* Machinery fault detection  
* Acoustic event segmentation  
* Smart sensor networks  
* Remote monitoring systems

The core idea remains the same:

Acoustic Signal  
      ↓  
Spectral Representation  
      ↓  
Visual Feature Extraction  
      ↓  
Machine Learning  
      ↓  
Temporal Event Detection  
---

# **26\. Key Takeaway**

This project demonstrates a useful engineering principle:

> **A difficult signal-processing problem can sometimes become a tractable computer-vision problem by choosing the right representation.**

Instead of processing thousands of hours of raw audio directly, the acoustic information is represented as spectrogram images.

From there, the system combines:

Signal Processing  
        \+  
Computer Vision  
        \+  
Feature Engineering  
        \+  
Machine Learning  
        \+  
Temporal Modeling  
        \+  
Optimization

to produce a complete end-to-end detection pipeline.

The final objective is not simply to answer:

> **"Is there a drone?"**

but:

> **"How many events are present, and exactly when does each event occur?"**

That distinction is what turns the project from a conventional classification problem into a temporal detection and localization problem.

---

# **27\. Future Improvements**

Potential future directions include:

* CNN-based spectrogram encoders  
* Vision Transformers  
* CNN \+ temporal sequence models  
* CRNN architectures  
* 1D temporal convolution  
* Transformer-based temporal segmentation  
* Self-supervised audio representation learning  
* Data augmentation in the spectrogram domain  
* Noise-robust feature extraction  
* Confidence calibration  
* Ensemble models  
* Sub-frame temporal localization  
* More sophisticated event matching strategies

These approaches could potentially improve temporal precision and robustness, especially in low signal-to-noise conditions.

---

# **28\. Author**

**Sepehr**

Bachelor's Student — Mechatronics Engineering

Areas of interest:

* Robotics  
* Machine Learning  
* Computer Vision  
* Signal Processing  
* Autonomous Systems  
* Programming  
* Engineering Problem Solving

---

## **License**

This repository contains an educational and research-oriented implementation.

See `LICENSE` for details.