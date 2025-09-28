# 🕵️‍♂️ Fraud Detection Pipeline with Ensemble Learning

This repository contains a complete machine learning pipeline for detecting fraudulent transactions using advanced classification techniques. The project is designed for high precision, recall, and generalization on imbalanced datasets, and is optimized for real-world deployment and interview readiness.

---

## 🚀 Project Overview

Fraud detection is a high-stakes, imbalanced classification problem where false negatives can be costly and false positives disruptive. This pipeline tackles the challenge by:

- Engineering robust features from transactional data
- Benchmarking multiple classifiers with fold-wise diagnostics
- Tuning thresholds for optimal F1 score
- Stacking high-performing models into a resilient ensemble
- Visualizing performance across metrics and folds

---

## 📊 Dataset

- **Source**: Feature-engineered transactional dataset (`feature_engineered_merged.csv`)
- **Target Variable**: `IsFraud` (binary classification)
- **Size**: ~64,000 bytes
- **Preprocessing**: Includes scaling, encoding, and domain-specific feature transformations

---

## 🧠 Models Compared

| Model               | ROC AUC       | PR AUC        | Accuracy | Recall | Precision | F1 Score |
|--------------------|---------------|---------------|----------|--------|-----------|----------|
| Logistic Regression | 0.8750        | 0.8701        | 0.8153   | 0.7787 | 0.7838    | 0.7789   |
| Random Forest       | 0.9581        | 0.9542        | 0.9249   | 0.9115 | 0.9085    | 0.9100   |
| XGBoost             | 0.9649        | 0.9583        | 0.9434   | 0.9439 | 0.9240    | 0.9331   |
| LightGBM            | 0.9757        | 0.9750        | 0.9594   | 0.9616 | 0.9425    | 0.9518   |
| CatBoost            | **0.9799**    | **0.9805**    | 0.9618   | 0.9499 | **0.9585**| 0.9540   |
| Stacked Ensemble    | 0.9764        | 0.9760        | **0.9631**| **0.9557**| 0.9567 | **0.9557**|

---

## 🧪 Final Verdict

- 🧠 **Best Solo Model**: CatBoost — highest ROC AUC, PR AUC, and precision
- 🧬 **Best Overall System**: Stacked Ensemble — best F1 score, recall, and generalization
- 🏢 **Best for Industry Deployment**: Stacked Ensemble — modular, scalable, and resilient

---

## 🧰 Pipeline Features

- 📁 Feature-engineered dataset with domain-specific transformations
- 🔁 Stratified K-Fold cross-validation (5 folds)
- 📊 Threshold tuning using precision-recall curves
- 📈 ROC AUC and PR AUC tracking across folds
- 📉 Confusion matrices and classification reports
- 📦 Ensemble model using VotingClassifier (XGB + CatBoost + LightGBM)
- 📊 Visualizations for metric comparison across models


