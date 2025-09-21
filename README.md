# Spotify Churn Prediction

**This project predicts whether a Spotify user will churn (stop using the service) based on listening behavior, demographic information, and account details. The model uses XGBoost with hyperparameter tuning and SMOTE to handle class imbalance.**

---

## **Table of Contents**
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Modeling and Hyperparameter Tuning](#modeling-and-hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [License](#license)

---

## **Dataset**

The dataset contains sample Spotify user information with **15 features** including demographic, usage, and subscription details:

| Column                  | Description |
|-------------------------|-------------|
| user_id                 | Unique identifier for each user |
| gender                  | Gender of the user |
| age                     | Age of the user |
| country                 | Country of the user |
| subscription_type       | Type of subscription (Free, Premium, Family, Student) |
| listening_time          | Total listening time (minutes) |
| songs_played_per_day    | Average songs played per day |
| skip_rate               | Fraction of songs skipped |
| device_type             | Device used (Mobile, Desktop, Web) |
| ads_listened_per_week   | Number of ads listened per week |
| offline_listening       | 1 if offline listening is enabled, 0 otherwise |
| is_churned              | Target variable: 1 if user churned, 0 otherwise |

---

## **Requirements**

- Python 3.10+
- Pandas
- NumPy
- scikit-learn
- XGBoost
- imbalanced-learn

**Install dependencies:**

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn
```

## **Data Preprocessing**

**Before training the model, the following preprocessing steps were applied:**

| Step                     | Description |
|--------------------------|-------------|
| **Feature Selection**     | Selected features: `songs_played_per_day`, `listening_time`, `age`, `skip_rate`. |
| **One-Hot Encoding**      | Convert categorical features like `gender`, `country`, `subscription_type`, and `device_type` to numeric. |
| **Train/Test Split**      | Split dataset into **80% training** and **20% testing** using stratification on the target variable (`is_churned`). |
| **Feature Scaling**       | Applied `StandardScaler` to numerical features (optional for XGBoost). |
| **Handle Class Imbalance**| Used **SMOTE** to oversample the minority class in the training set. |

**Python Code Example:**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
```

# Features and target

```bash
features = ['songs_played_per_day', 'listening_time', 'age', 'skip_rate']
X = df[features]
y = df['is_churned']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

# Feature scaling

```bash
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

# Handle class imbalance with SMOTE

```bash
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
```

## **Modeling and Hyperparameter Tuning**

**The model uses an XGBoost classifier with GridSearchCV to find the best hyperparameters.**

| Parameter            | Description |
|---------------------|-------------|
| **n_estimators**     | Number of trees in the ensemble |
| **max_depth**        | Maximum depth of each tree |
| **learning_rate**    | Step size shrinkage to prevent overfitting |
| **subsample**        | Fraction of training samples used per tree |
| **colsample_bytree** | Fraction of features used per tree |

**Python Code Example:**

```bash
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
```

# Calculate scale_pos_weight to handle class imbalance

```bash
scale_pos_weight = (y_train_res == 0).sum() / (y_train_res == 1).sum()

xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='f1',  # Focus on minority class
    cv=3,
    verbose=1,
    n_jobs=-1
)
```

# Fit GridSearchCV

```bash
grid_search.fit(X_train_res, y_train_res)
```

# Best model and parameters

```bash
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
```

## **Evaluation**

**Model performance on the test set:**

| Metric                       | Value |
|-------------------------------|-------|
| **Accuracy**                  | 0.56875 |
| **Confusion Matrix**          | [[735, 451], [239, 175]] |
| **Precision / Recall / F1**  | See classification report below |

**Python Code Example:**

```bash
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

# Predictions on test set

```bash
y_pred = best_model.predict(X_test_scaled)
```

# Evaluation metrics

```bash
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```
