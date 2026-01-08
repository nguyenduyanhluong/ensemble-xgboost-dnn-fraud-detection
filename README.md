# TMU Research Assistant
# ENSEMBLE MACHINE LEARNING WITH XGBOOST AND DEEP NEURAL NETWORKS FOR IMBALANCED FRAUD DETECTION
## Overview

This repository contains the implementation and experimental analysis of an **enhanced and extended version of my Major Research Project (MRP)** for the MSc in Data Science and Analytics program at Toronto Metropolitan University. The work has been developed into a **full-length journal manuscript** focusing on fraud detection in **large-scale, high-dimensional, and severely imbalanced financial transaction data.**

The project proposes a **hybrid ensemble framework** that integrates **Enhanced XGBoost** with a **Deep Neural Network (DNN)** to improve fraud detection accuracy, robustness, and interpretability. Multiple supervised learning models are systematically evaluated using imbalance-aware training strategies and metrics appropriate for rare-event detection. The study emphasizes both **predictive performance** and **operational readiness** for real-world financial applications.

## Objectives

- Develop a scalable and accurate fraud detection framework for **highly imbalanced transaction data.**
- Compare traditional machine learning, ensemble methods, and deep learning models under identical experimental conditions.
- Evaluate the benefits of combining **tree-based boosting** with **deep neural networks** in a hybrid ensemble.
- Improve minority-class (fraud) detection using **class weighting, focal loss, and threshold optimization.**
- Ensure model transparency through **SHAP-based interpretability analysis.**
- Provide practical insights for deploying fraud detection models in regulated financial environments.

## Methodology
- **Dataset:** IEEE-CIS Fraud Detection Dataset, containing anonymized transaction and identity information from real-world e-commerce activity.
- **Data Preprocessing:** Merged transaction and identity tables using TransactionID, Handled missing categorical values using explicit “missing” categories, Applied label encoding to high-cardinality categorical features, and Engineered temporal features (e.g., hour-of-day) to capture behavioral patterns.
- **Model Selection:** Logistic Regressionm Random Forest, XGBoost, Enhanced XGBoost (advanced tuning and regularization), Artificial Neural Network (ANN), Deep Neural Network (DNN), **Hybrid XGBoost + DNN Ensemble.**
- **Class Imbalance Handling:** Cost-sensitive learning via class weighting (scale_pos_weight), Focal loss for deep neural networks, Decision-threshold optimization to balance recall and precision.
- **Evaluation Metrics:** Recall, Precision, F1-score, ROC-AUC and Precision–Recall AUC (PR-AUC).
- **Model Interpretation:** SHAP (SHapley Additive Explanations) for global and local feature attribution, and Feature importance analysis for tree-based models.

## Key Results
- **Enhanced XGBoost** achieved strong single-model performance with **ROC-AUC ≈ 0.963** and improved recall–precision balance.
- The **XGBoost + DNN ensemble** delivered the **best overall performance,** achieving: **ROC-AUC ≈ 0.974**, **F1-score ≈ 0.74**, and **PR-AUC ≈ 0.79**
- The ensemble outperformed traditional baselines (Logistic Regression, Random Forest) and standalone deep learning models.
- SHAP analysis identified **transaction amount, card attributes, identity-linked variables, and temporal patterns** as the most influential fraud predictors.
- Results demonstrate that hybrid ensemble learning provides a **robust, interpretable, and deployment-ready** solution for fraud detection.

## Dataset

**Source:** [IEEE-CIS Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

The dataset contains over 590,000 anonymized transaction records, with fraud representing a very small fraction of observations. It reflects real-world challenges including **extreme class imbalance, high dimensionality,** missing identity information, and complex temporal patterns.

## Contact

**Author:** Nguyen Duy Anh Luong  
**Supervisor:** Dr. Shengkun Xie  
**Email:** [nguyenduyanh.luong@torontomu.ca]
