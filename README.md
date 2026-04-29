# Chem277a_Project_II
# Alzheimer's Disease Risk Classification


**Authors:** Zander Rothering, Shivani Tijare, Seungho Yoo, Girish Krishna

## Overview

This project builds and compares classification models for predicting Alzheimer's disease risk from a global demographic and clinical-features dataset. We walk through a full machine-learning pipeline: exploratory data analysis, feature encoding, feature selection, dimensionality analysis (PCA), and model training and tuning across four classifiers — Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine.

Early identification of at-risk individuals is clinically meaningful because Alzheimer's disease has a preclinical phase during which intervention may slow cognitive decline. We treat this as a binary classification problem and place particular emphasis on minimizing false negatives, since missing a true case carries a higher clinical cost than a false alarm.

## Dataset

- **Source:** [Alzheimer's Prediction Dataset (Global)](https://doi.org/10.34740/KAGGLE/DSV/10618775) — Ankit (2025), Kaggle
- **File:** `alzheimers_prediction_dataset.csv`
- **Size:** 74,283 records × 25 features
- **Target:** `Alzheimer's Diagnosis` (Yes / No)
- **Class balance:** ~58.7% No, ~41.3% Yes
- **Missing values:** None

### Feature groups

| Group | Features |
|---|---|
| Numeric | Age, BMI, Education Level, Cognitive Test Score |
| Binary / nominal | Gender, Smoking Status, Alcohol Consumption, Diabetes, Hypertension, Cholesterol Level, Family History, Sleep Quality, Dietary Habits, Employment Status, Marital Status, APOE-ε4 allele, Urban vs Rural |
| Ordinal | Physical Activity Level, Depression Level, Air Pollution Exposure, Social Engagement Level, Income Level, Stress Levels |
| High-cardinality | Country |

## Repository structure

```
.
├── Project_II.ipynb                      # Main analysis notebook
├── alzheimers_prediction_dataset.csv     # Input dataset
└── README.md                             # This file
```

## Requirements

Python 3.9 or newer. Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn category_encoders jupyter
```

## How to run

1. Clone the repository:
   ```bash
   git clone https://github.com/Zander-Rothering/Chem277a_Project_II.git
   cd Chem277a_Project_II
   ```
2. Make sure `alzheimers_prediction_dataset.csv` is in the project root (same folder as the notebook).
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Project_II.ipynb
   ```
4. Run all cells in order (Cell → Run All).

Expected total runtime: roughly 10–20 minutes on a modern laptop. The Random Forest GridSearchCV cell is the slowest (~5–15 minutes); all other cells finish in under a minute each.

## Pipeline summary

1. **EDA** — distribution checks, target balance, numeric correlations, categorical countplots, mutual information, and chi-square tests for feature–target relationships.
2. **Encoding** — one-hot for binary/nominal features (with `drop_first=True`), explicit ordinal encoding for level-based features (Low/Medium/High), and binary encoding for the high-cardinality `Country` column. Column names are normalized to handle curly apostrophes in the source CSV.
3. **Feature selection** — correlation heatmap, target correlation, and mutual information scoring.
4. **Train/test split and scaling** — stratified 80/20 split (`stratify=y`, `test_size=0.2`); `StandardScaler` fit on training data only, applied to numeric columns.
5. **PCA** — exploratory only; the encoded feature space does not concentrate variance, so the full feature set is retained.
6. **Baseline modeling** — all four classifiers fit with reasonable defaults, evaluated by accuracy and confusion matrix.
7. **Hyperparameter tuning** — `GridSearchCV` with 5-fold stratified cross-validation on the training set, scored on accuracy. The chosen model is evaluated once on the held-out test set. `LinearSVC` is used in place of `SVC(kernel='linear')` to keep SVM tuning fast on the ~74k-row dataset.

## Results

All four tuned models cluster around **70% test accuracy**, with Random Forest and Logistic Regression slightly ahead of Decision Tree and SVM. `Age`, the `APOE-ε4 allele` indicator, and `Family History` emerge consistently as the strongest individual predictors across mutual information, target correlation, and Random Forest feature importance — in line with established clinical literature on Alzheimer's risk. The full results table, confusion matrices, and per-model best parameters are in the notebook.

> **Note on metrics:** Accuracy is reported as the primary metric for consistency across models. In a clinical screening setting, false negatives carry the higher cost — the confusion matrices in the notebook show how each model trades off false positives vs false negatives, which is the more relevant view than accuracy alone.

## Known limitations and future work

- Categorical encoders are currently fit on the full dataset rather than inside a scikit-learn `Pipeline` with `ColumnTransformer` fit only on training folds; this is a mild form of leakage worth tightening.
- Accuracy is the only metric reported; precision, recall, F1, and ROC-AUC would give a fuller picture, particularly for the false-negative-sensitive use case.
- Gradient-boosted trees (XGBoost, LightGBM) typically outperform Random Forest on tabular data of this size and would be a natural next baseline.
- Threshold tuning on predicted probabilities would let the screening trade-off be set explicitly rather than left at the default 0.5.
- Additional features capturing biomarker data (e.g., CSF amyloid levels, MRI volumetrics) would likely produce a meaningful step up in performance over the current demographic and lifestyle features.

## References

1. Ankit. (2025). *Alzheimer's Prediction Dataset (Global)* [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/10618775
2. Ballard, C., Gauthier, S., Corbett, A., Brayne, C., Aarsland, D., & Jones, E. (2011). Alzheimer's disease. *The Lancet*, 377(9770), 1019–1031. https://doi.org/10.1016/S0140-6736(10)61349-9
3. Lane, C. A., Hardy, J., & Schott, J. M. (2018). Alzheimer's disease. *European Journal of Neurology*, 25(1), 59–70. https://doi.org/10.1111/ene.13439
4. Jack, C. R., et al. (2018). NIA-AA research framework: Toward a biological definition of Alzheimer's disease. *Alzheimer's & Dementia*, 14(4), 535–562. https://doi.org/10.1016/j.jalz.2018.02.018

## License

Academic project for Chem 277A coursework. Dataset is redistributed under its original Kaggle license.