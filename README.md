# Polarimetric ToF Material Classification

This repository contains code and instructions for training and evaluating a **polarimetric Time-of-Flight (ToF) material classification** .

## Dataset

The dataset used in this project is available on Zenodo:

- **Zenodo Link**: [https://zenodo.org/records/15038366]  
- **Data Contents**:  
  - Training examples (e.g., `.npy` format)  
  - Corresponding labels (`label.npy`)  

```css
dataset/
  └── <train_id>/
      ├── some_data.npy
      └── label.npy
```


## Requirements

- Python 3.8+  
- NumPy  
- scikit-learn  
- Matplotlib  
- Pandas

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

