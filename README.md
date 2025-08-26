# Customer Churn Prediction
# Overview
This project focuses on predicting customer churn using the PIMA Indians Diabetes Dataset as a proxy, where the binary Outcome variable (0 = no diabetes, 1 = diabetes) represents churn. The goal is to identify at-risk customers using machine learning, enabling businesses to implement targeted retention strategies. The project covers data preprocessing, model training, evaluation, interpretability, and hyperparameter tuning, with a focus on actionable insights.

# Dataset
The PIMA Indians Diabetes Dataset contains 768 records with 8 features and 1 target:

# Features: 
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

#Target: 
Outcome (0 = no churn, 1 = churn)

# Methodology
# Data Preprocessing:
Replaced invalid zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI with median values.

# Applied StandardScaler for feature scaling.
Used SMOTE to handle class imbalance in the target variable.

# Exploratory Data Analysis (EDA):
Visualized class distribution and feature relationships (e.g., Glucose vs. Outcome) using Seaborn count plots and box plots.

# Model Training:
Trained three models: Logistic Regression, Random Forest, and XGBoost.
Used eval_metric='logloss' for XGBoost to optimize binary classification.

# Model Evaluation:
Evaluated models using classification_report (precision, recall, F1-score) and ROC-AUC.
Plotted ROC curves to compare model performance.

# Model Interpretability:
Used SHAP to analyze feature importance and explain predictions (e.g., Glucose and BMI as key drivers).

# Hyperparameter Tuning:
Performed GridSearchCV on XGBoost to optimize n_estimators, max_depth, and learning_rate.

# Final Model:
Trained a final XGBoost model with the best parameters and saved it using joblib.

# Requirements
To run the code, install the required libraries:
pip install pandas seaborn matplotlib scikit-learn imbalanced-learn xgboost shap joblib numpy

# How to Run
Download the PIMA Indians Diabetes Dataset and place it as Dataset.txt in the project directory.

# Run the script:
python churn_prediction.py
Check the output for model performance, visualizations, and the saved model (diabetes_model.pkl).

# Results
Logistic Regression: Good baseline but lower performance on imbalanced data.
Random Forest: Strong accuracy but moderate F1-score due to class imbalance.
XGBoost: Best performer with the highest F1-score and ROC-AUC after tuning.
Key Features: Glucose and BMI were identified as top predictors of churn via SHAP analysis.

# Next Steps
The next phase of this project will explore unsupervised learning techniques, such as K-Means Clustering, to segment customers based on behavior and uncover hidden patterns for targeted retention strategies. This will complement the supervised models by providing deeper insights into customer groups.

# Contributing
Feel free to fork this repository, submit pull requests, or open issues for suggestions and improvements.
