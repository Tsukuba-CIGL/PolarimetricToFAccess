# Polarimetric ToF Material Classification

This repository contains code and instructions for training and evaluating a **polarimetric Time-of-Flight (ToF) material classification** pipeline. By leveraging polarization features, this system aims to improve material classification performance compared to standard intensity or depth‐only methods.

## Overview

- **Goal**: Classify different materials using polarimetric ToF signals.
- **Techniques**: 
  - Polarimetric ToF imaging
  - Feature extraction across various frequencies
  - Dimensionality reduction (LDA, PCA, etc.)
  - Classifiers (e.g., Linear Discriminant Analysis, SVM, etc.)

This pipeline can serve as a reference implementation for researchers working with ToF‐based material classification. It includes data loading, preprocessing, classification, and visualization steps.

## Dataset

The dataset used in this project is available on Zenodo:

- **Zenodo Link**: [Insert Zenodo URL Here]  
- **Data Contents**:  
  - Training examples (e.g., `.npy` format)  
  - Corresponding labels (`label.npy`)  
  - Calibrated test data  
  - Additional metadata (if applicable)

Place or symlink the dataset under a structure like:

```css
dataset/
  └── <train_id>/
      ├── some_data.npy
      └── label.npy
  └── <test_id>/
      └── some_data_calibrated.npy
```


## Requirements

- Python 3.8+  
- NumPy  
- SciPy  
- scikit-learn  
- Matplotlib  
- Pandas (if you need to export confusion matrices or results in CSV)

Install dependencies:

```bash
pip install -r requirements.txt
```
## Repository Structure

```css
Polarimetric-ToF-Material-Classification/
├── README.md
├── requirements.txt
├── analysis.py
├── dataset/
└── results/
```

- `analysis.py`: Functions for reading .npy data and labels.
- `dataset/`: Data from Zenodo (ignored by version control).
- `results/`: Figures, CSV outputs, etc.

## Usage

### 1. Clone & Install

```bash
git clone https://github.com/Tsukuba-CIGL/PolarimetricToFAccess.git
cd polarimetric-tof-access
pip install -r requirements.txt
```

### 2. Download Dataset
- Acquire from [Zenodo](https://zenodo.org/records/15038366) 
- Place the data under `dataset/<train_id>` and `dataset/<test_id>`

### 3. Example of Running the Code

```bash
python analysis.py
```

