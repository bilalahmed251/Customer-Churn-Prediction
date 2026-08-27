# Advanced Customer Churn Prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Initializing Advanced Churn Pipeline...")
df = pd.DataFrame({
    'tenure': np.random.randint(1, 72, 1000),
    'MonthlyCharges': np.random.uniform(20, 120, 1000),
    'TotalCharges': np.random.uniform(20, 8000, 1000),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 1000),
    'Churn': np.random.choice(['Yes', 'No'], 1000)
})

X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['Contract', 'InternetService']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost within Pipeline...")
pipeline.fit(X_train, y_train)

print("Evaluating Pipeline...")
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))

print("Saving Pipeline to disk...")
joblib.dump(pipeline, 'churn_prediction_pipeline.pkl')
print("Advanced Churn Pipeline Complete!")
