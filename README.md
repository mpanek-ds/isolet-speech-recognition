# ISOLET Audio Signal Classification (Speech Recognition)

## 📌 Project Overview
This project focuses on building an advanced multiclass classification system to recognize 26 spoken English letters based on audio signal features. The analysis utilizes the **ISOLET** (Isolated Letter Speech Recognition) dataset, containing 617 predictors representing extracted audio features (e.g., spectral coefficients). The core challenge lies in the high dimensionality and the subtle phonetic boundaries between certain letters (e.g., "M" vs. "N" or "P" vs. "B").

## 🎯 Business & Engineering Objective
With 617 "black box" features, traditional linear models are prone to overfitting and the curse of dimensionality. The objective was to design a robust pipeline that aggressively filters noise, normalizes features, and identifies the optimal non-linear algorithm capable of capturing complex phonetic interactions.

## 🧠 Machine Learning Pipeline
The entire machine learning pipeline was developed using the `tidymodels` framework in R:
* **Preprocessing (NZV & Normalization):** Applied a Near Zero Variance (NZV) filter to eliminate 20 predictors with zero or near-zero variance, preventing numerical instability in kernel-based algorithms. All numeric predictors were then strictly normalized.
* **Model Evolution:** 
  * Started with baseline models: **Logistic Regression** and **Random Forest** (which achieved ~92% accuracy).
  * Explored ensemble methods: **Bagging** and **Stacking** (SVM + RF via the `stacks` package).
  * Concluded that a single, precisely tuned non-linear classifier outperformed complex ensembles.
* **SVM Tuning (The "Manual" Sniper Approach):** 
  * Initially used Auto Grid Search with 5-fold CV to narrow the parameter space.
  * Applied manual grid tuning to pinpoint the exact optimal hyperparameters: $C = 60$ (high penalty for errors) and $\sigma = 0.00002$ (extremely smooth kernel).

## 📊 Key Results & Diagnostics
The manually tuned **SVM with RBF Kernel** emerged as the superior model.

* **Accuracy:** Reached **95.99%** on the test set.
* **Overfitting Audit:** The generalization gap was highly optimal (Train: 98.64% vs Test: 95.99%), proving the model learned the signal, not the noise.
* **Confusion Matrix Analysis:** The model exhibited logical, phonetic-based misclassifications (e.g., confusing "M" and "N"), confirming it learned genuine audio properties rather than data leakage artifacts.
* **Final Production Model:** The winning SVM architecture was ultimately retrained on 100% of the available data to maximize stability for production deployment.

## 🛠️ Technology Stack
* **Language:** R
* **Libraries:** `tidymodels`, `tidyverse`, `kernlab` (SVM), `ranger` (Random Forest), `stacks`, `baguette`, `vip`, `doParallel`

## 🚀 Repository Contents & How to Run
1. Clone the repository and ensure `isolet_data.csv` is in your working directory.
2. Run `speech_recognition_isolet.R` to view the entire history of experiments, tuning, stacking attempts, and final model architecture.
3. Review `project_report.pdf` (or `project_report.Rmd`) for a detailed statistical breakdown and confusion matrix diagnostics.

*(Note: The final trained `model_production.rds` is excluded from this repository due to GitHub's 100MB file size limit. The original R scripts and the detailed PDF report are written in Polish as they were part of a university submission, but the code structure follows universal data science standards).*

## 👥 Contributors
* **Mateusz Panek**
* **Konrad Serocki**
