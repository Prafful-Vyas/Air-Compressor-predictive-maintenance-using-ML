# Predictive Maintenance of Industrial Air Compressor

## Overview

This project implements a predictive maintenance (PdM) pipeline for an industrial air compressor system using sensor data. The objective is to identify early signs of bearing degradation and predict impending failures before they lead to downtime or costly damage.

The project focuses on **early warning**, not post-failure diagnosis, and follows realistic industry practices used in condition-based maintenance systems.

---

## Dataset Description

The dataset consists of 1000 observations and 17 sensor-based features collected from an air compressor supplying air to a factory production line.

Key sensor categories include:

* Mechanical: RPM, torque, vibration (ground and head acceleration)
* Electrical: motor power, oil pump power, water pump power
* Thermal: outlet temperature, oil tank temperature, cooling water temperatures
* Acoustic: noise levels
* Hydraulic: air pressure, air flow, water pressure and flow

### Target Variable

* `Bearings Status`: Binary indicator of bearing condition

  * 0 → OK
  * 1 → Noise / Faulty

An additional **predictive label** was engineered:

* `bearing_failure_soon`: Indicates whether a bearing failure is expected in the near future based on future sensor behavior.

---

## Project Goals

1. Identify sensor signals correlated with bearing degradation
2. Engineer meaningful time-domain and frequency-domain features
3. Build a predictive model that prioritizes early failure detection
4. Evaluate model performance with a focus on failure recall

---

## Feature Engineering

### Time-Domain Features

* Noise levels
* Motor power consumption
* Vibration magnitude derived from ground acceleration (X, Y, Z)

### Frequency-Domain Features

To capture subtle mechanical degradation patterns:

* Windowed Fast Fourier Transform (FFT) applied to vibration signals
* Extracted spectral features such as high-frequency energy and spectral spread

These features help model instability and irregular energy distribution caused by early bearing wear.

---

## Modeling Approach

* Supervised classification using Logistic Regression
* Time-aware train/test split (no shuffling) to simulate real-world deployment
* Class imbalance handled through metric selection rather than oversampling

### Evaluation Metrics

* Primary focus: Recall for failure class (minimizing missed failures)
* Secondary metrics: Precision, F1-score, overall accuracy

---

## Results

* Achieved ~89% recall for impending bearing failures
* FFT-based features provided marginal but stable improvements and added domain interpretability
* Demonstrated realistic predictive maintenance behavior with subtle early-warning signals

---

## Tools and Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Signal processing (FFT)

---

## Key Learnings

* Early failure signals are subtle and require careful feature analysis
* Frequency-domain features complement, rather than replace, time-domain signals
* High failure recall is more important than raw accuracy in predictive maintenance systems

---

## Future Improvements

* Remaining Useful Life (RUL) estimation
* Advanced models (Random Forest, XGBoost)
* Deployment using PySpark and AWS Glue for large-scale streaming data
* Real-time monitoring and alerting integration

---

## Disclaimer

This project is intended for educational and demonstration purposes and uses a simulated industrial dataset.