# ga-pls-cellulose-solubility
Reproducible GA-PLS variable selection and PLS modeling workflow used for descriptor-based solubility analysis in cellulose-derived systems.

# GA-PLS Modeling for Cellulose-Based Systems

This repository contains Python scripts and datasets used for GA-PLS modeling, nested cross-validation, final PLS model construction, multicollinearity assessment, and y-randomization tests.

The study focuses on modeling logS (and related response variables) for cellulose-based materials (CMC, HPMC, TOCN).

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                # Original descriptor datasets
│   └── final/              # Descriptor sets after GA selection and correlation filtering
│
├── src/
│   ├── gapls_nested_loocv.py
│   ├── final_pls_loocv_vip.py
│   ├── y_randomization_loocv.py
│   └── multicollinearity_check.py
│
├── README.md
└── requirements.txt

```

---

# Data Description

- dataset_*.csv
  Original descriptor sets before GA-based variable selection.
- final_model_dataset_*.csv
  Descriptor sets after:
  1. GA-PLS variable selection (nested LOOCV)
  2. Manual removal of redundant descriptors based on correlation analysis.

All final modeling, VIP calculation, and y-randomization tests were performed using the corresponding final_model_dataset_*.csv files.

# Data Format
Each CSV file follows the structure:
- Column 1: response variable (y)
- Columns 2 onward: descriptor variables (X)

# Scripts
## 1. GA-PLS (Nested LOOCV)
gapls_nested_loocv.py
- Outer CV: LOOCV
- Inner CV: 5-fold
- GA individual: binary bitstring
- Outputs:
  - nested_loocv_predictions.csv
  - variable_selection_frequency.csv

## 2. Final PLS Model + VIP
final_pls_loocv_vip.py
- Autoscaling of X and y
- LOOCV component selection (maximize R2cv on original y-scale)
- VIP calculation
- Outputs:
  - CV predictions
  - VIP table
  - Model summary

## 3. Y-Randomization Test
y_randomization_loocv.py
- 200 permutations (default)
- Empirical p-value calculation
- Histogram visualization
- Outputs:
  - permutation results
  - summary statistics
  - p-value metadata

## 4. Multicollinearity Check
multicollinearity_check.py
- Pearson correlation matrix
- Heatmap visualization
- Extraction of strongly correlated descriptor pairs (|r| >= 0.85)

# Software Environment
Tested with:
- Python 3.9+
- scikit-learn 1.0+
- numpy
- pandas
- matplotlib
- seaborn
- deap

Install dependencies:
```
pip install -r requirements.txt
```

# Reproducibility
- Random seeds are fixed where applicable.
- Autoscaling is performed using training data only in nested CV.
- Y-randomization uses reproducible random number generation.
- All scripts are designed for full reproducibility of the reported results.

# Citation
If you use this repository, please cite the corresponding publication.

# Author
Yoshikuni Teramoto, Kyoto University

