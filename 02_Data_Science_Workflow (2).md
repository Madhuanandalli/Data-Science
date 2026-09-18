# 🔄 Data Science Notes — Part 2: Data Science Workflow

> End-to-end pipeline of a real Data Science project — theory + practical Python syntax and code examples.

---

## Table of Contents
1. [Overview of the Workflow](#1-overview-of-the-workflow)
2. [Problem Definition](#2-problem-definition)
3. [Data Collection](#3-data-collection)
4. [Data Cleaning](#4-data-cleaning)
5. [Data Preprocessing](#5-data-preprocessing)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Building](#8-model-building)
9. [Model Evaluation](#9-model-evaluation)
10. [Deployment](#10-deployment)
11. [End-to-End Mini Project](#11-end-to-end-mini-project)

---

## 1. Overview of the Workflow

```
┌──────────────────────┐
│ 1. Problem Definition│  ← What are we solving? Business goal
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. Data Collection   │  ← CSV, SQL, API, Web Scraping
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. Data Cleaning     │  ← Missing values, duplicates, outliers
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. Data Preprocessing│  ← Encoding, scaling, splitting
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. EDA               │  ← Stats + visualization + correlation
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 6. Feature Engineering│ ← Create/select/transform features
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 7. Model Building    │  ← Train ML algorithms
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 8. Model Evaluation  │  ← Accuracy, RMSE, F1, ROC-AUC
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 9. Deployment        │  ← API / Web app / Dashboard
└──────────────────────┘
        ↺ Monitor & Retrain
```

> ⚠️ **Note:** The workflow is **iterative**, not strictly linear. You often go back from EDA → cleaning, or from evaluation → feature engineering.

### Time Distribution (Real-World Reality)
| Stage | Approx. Time Spent |
|---|---|
| Data Collection + Cleaning + Preprocessing | **~70–80%** |
| EDA + Feature Engineering | ~10–15% |
| Model Building + Evaluation | ~5–10% |
| Deployment | ~5% |

---

## 2. Problem Definition

The **most important** and most skipped step. A technically perfect model solving the wrong problem is worthless.

### What to define
| Question | Example |
|---|---|
| **Business objective?** | Reduce customer churn |
| **ML problem type?** | Binary classification |
| **Target variable (y)?** | `Churn` (Yes/No) |
| **Success metric?** | Recall ≥ 80% |
| **Data available?** | 2 years of customer transaction logs |
| **Constraints?** | Must predict within 200ms |

### Mapping Business Problem → ML Problem Type
| Business Question | ML Problem Type | Example Algorithm |
|---|---|---|
| Will this customer leave? | Classification | Logistic Regression |
| What will sales be next month? | Regression | Linear Regression |
| Which customers are similar? | Clustering | K-Means |
| What should we recommend? | Recommendation | Collaborative Filtering |
| Is this transaction unusual? | Anomaly Detection | Isolation Forest |

### Practical — Documenting the problem in code
```python
"""
PROJECT: Customer Churn Prediction
-----------------------------------
Business Goal  : Reduce monthly churn by identifying at-risk customers
ML Problem     : Binary Classification
Target (y)     : Churn  -> 1 = customer left, 0 = stayed
Features (X)   : tenure, monthly_charges, contract_type, support_calls
Success Metric : Recall >= 0.80 (catching churners matters more than precision)
Baseline       : Current rule-based system = 55% recall
"""
```

---

## 3. Data Collection

Gathering raw data from one or more sources.

### Common Data Sources
| Source | Tool / Method |
|---|---|
| CSV / Excel files | `pandas.read_csv()`, `read_excel()` |
| Databases | SQL, `sqlite3`, `SQLAlchemy` |
| APIs | `requests` library, REST/JSON |
| Web pages | Web scraping — `BeautifulSoup`, `Selenium` |
| Big Data | Hadoop, Spark, Hive |
| Public datasets | Kaggle, UCI ML Repository, data.gov |

### Practical — Reading from different sources

**a) CSV / Excel / JSON**
```python
import pandas as pd

df = pd.read_csv("data.csv")                     # CSV
df = pd.read_csv("data.csv", sep=";")            # custom delimiter
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")   # Excel
df = pd.read_json("data.json")                   # JSON
df = pd.read_csv("https://example.com/data.csv") # directly from URL
```

**b) SQL Database**
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("company.db")
query = """
    SELECT customer_id, tenure, monthly_charges, churn
    FROM customers
    WHERE signup_date >= '2023-01-01'
"""
df = pd.read_sql_query(query, conn)
conn.close()
```

**c) API**
```python
import requests
import pandas as pd

url = "https://api.example.com/v1/sales"
response = requests.get(url, params={"year": 2024})

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data["results"])
else:
    print("Error:", response.status_code)
```

**d) Web Scraping**
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

page = requests.get("https://example.com/products")
soup = BeautifulSoup(page.content, "html.parser")

names  = [t.text.strip() for t in soup.find_all("h2", class_="product-name")]
prices = [p.text.strip() for p in soup.find_all("span", class_="price")]

df = pd.DataFrame({"product": names, "price": prices})
```

**e) Saving collected data**
```python
df.to_csv("raw_data.csv", index=False)
df.to_excel("raw_data.xlsx", index=False)
```

---

## 4. Data Cleaning

Real-world data is **dirty**. Cleaning fixes errors so analysis isn't garbage-in-garbage-out.

### Common Data Quality Issues
1. Missing values (`NaN`, `null`, blanks)
2. Duplicate rows
3. Outliers / extreme values
4. Inconsistent formats (`"Male"`, `"male"`, `"M"`)
5. Wrong data types (`"25"` as string instead of int)
6. Irrelevant columns
7. Typos and whitespace

### 4.1 Inspecting the data
```python
df.shape            # (rows, columns)
df.info()           # dtypes + non-null counts
df.describe()       # numeric summary
df.head(10)         # first 10 rows
df.isnull().sum()   # missing values per column
df.duplicated().sum()  # number of duplicate rows
df['col'].unique()     # unique values
df['col'].value_counts()  # frequency count
```

### 4.2 Handling Missing Values

| Strategy | When to use |
|---|---|
| **Drop rows** | Very few missing (<5%) |
| **Drop column** | Column mostly empty (>50–60%) |
| **Mean imputation** | Numeric, roughly normal distribution |
| **Median imputation** | Numeric with outliers/skew |
| **Mode imputation** | Categorical data |
| **Forward/backward fill** | Time-series data |
| **Model-based (KNN)** | When accuracy matters |

```python
# Check missing
print(df.isnull().sum())
print(df.isnull().mean() * 100)   # % missing per column

# --- Dropping ---
df = df.dropna()                          # drop rows with any NaN
df = df.dropna(subset=['age'])            # drop only if 'age' is NaN
df = df.dropna(axis=1, thresh=len(df)*0.5)  # drop cols >50% missing

# --- Filling ---
df['age'] = df['age'].fillna(df['age'].mean())      # mean
df['salary'] = df['salary'].fillna(df['salary'].median())  # median
df['city'] = df['city'].fillna(df['city'].mode()[0])       # mode
df['temp'] = df['temp'].fillna(method='ffill')      # forward fill (time-series)
df['temp'] = df['temp'].fillna(method='bfill')      # backward fill

# --- Advanced: KNN Imputation ---
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[['age','salary']] = imputer.fit_transform(df[['age','salary']])
```

### 4.3 Handling Duplicates
```python
print(df.duplicated().sum())          # count duplicates
df = df.drop_duplicates()             # remove exact duplicates
df = df.drop_duplicates(subset=['customer_id'], keep='first')
```

### 4.4 Fixing Data Types
```python
df['age'] = df['age'].astype(int)
df['price'] = pd.to_numeric(df['price'], errors='coerce')  # bad values → NaN
df['join_date'] = pd.to_datetime(df['join_date'], errors='coerce')
df['category'] = df['category'].astype('category')
```

### 4.5 Fixing Inconsistent Text
```python
df['gender'] = df['gender'].str.strip().str.lower()
df['gender'] = df['gender'].replace({'m':'male', 'f':'female'})
df['city'] = df['city'].str.title()        # "new delhi" → "New Delhi"
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
```

### 4.6 Handling Outliers

**Method 1 — IQR (Interquartile Range)**
```python
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Detect
outliers = df[(df['salary'] < lower) | (df['salary'] > upper)]
print("Outliers found:", len(outliers))

# Remove
df = df[(df['salary'] >= lower) & (df['salary'] <= upper)]

# OR Cap (Winsorization) - keeps rows, limits extremes
df['salary'] = df['salary'].clip(lower, upper)
```

**Method 2 — Z-Score**
```python
from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df['salary']))
df = df[z < 3]      # keep rows within 3 standard deviations
```

---

## 5. Data Preprocessing

Converting clean data into a **machine-readable numeric format** ready for ML models.

### 5.1 Encoding Categorical Variables

| Technique | Use case |
|---|---|
| **Label Encoding** | Ordinal data (Low < Medium < High) |
| **One-Hot Encoding** | Nominal data with few categories |
| **Ordinal Encoding** | Explicit custom order |
| **Target/Frequency Encoding** | High-cardinality columns |

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# --- Label Encoding ---
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])   # male=1, female=0

# --- One-Hot Encoding (pandas - easiest) ---
df = pd.get_dummies(df, columns=['city'], drop_first=True)

# --- One-Hot Encoding (sklearn) ---
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded = ohe.fit_transform(df[['city']])

# --- Ordinal Encoding with explicit order ---
oe = OrdinalEncoder(categories=[['Low','Medium','High']])
df['rating_encoded'] = oe.fit_transform(df[['rating']])

# --- Manual mapping ---
df['size'] = df['size'].map({'S':1, 'M':2, 'L':3, 'XL':4})
```

### 5.2 Feature Scaling

| Scaler | Formula | When to use |
|---|---|---|
| **StandardScaler** | (x − μ) / σ | Normally distributed data; most ML models |
| **MinMaxScaler** | (x − min)/(max − min) → [0,1] | Neural networks, image data |
| **RobustScaler** | (x − median)/IQR | Data with outliers |

> Models that **need** scaling: KNN, SVM, Logistic Regression, Neural Networks, PCA, K-Means.
> Models that **don't**: Decision Trees, Random Forest, XGBoost.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # mean=0, std=1

mm = MinMaxScaler()
X_scaled = mm.fit_transform(X)              # range [0, 1]

rb = RobustScaler()
X_scaled = rb.fit_transform(X)              # outlier-resistant
```

> 🚨 **Critical rule:** `fit_transform()` on **training** data only, `transform()` on test data — otherwise you leak information.
```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)     # NOT fit_transform!
```

### 5.3 Train-Test Split
```python
from sklearn.model_selection import train_test_split

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 80% train, 20% test
    random_state=42,      # reproducibility
    stratify=y            # keep class ratio same (for classification)
)
print(X_train.shape, X_test.shape)
```

### 5.4 Handling Imbalanced Data
```python
# Check imbalance
print(y.value_counts(normalize=True))

# SMOTE - generate synthetic minority samples
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# OR use class weights inside the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
```

---

## 6. Exploratory Data Analysis (EDA)

Understanding the data through **statistics and visualization** before modeling.

### Goals of EDA
- Understand distributions and central tendency
- Find relationships & correlations
- Detect anomalies and data issues
- Generate hypotheses for feature engineering

### 6.1 Univariate Analysis (one variable)
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Numeric
df['age'].describe()
sns.histplot(df['age'], kde=True); plt.show()      # distribution
sns.boxplot(x=df['salary']); plt.show()            # outliers
print("Skewness:", df['age'].skew())

# Categorical
df['city'].value_counts()
sns.countplot(x='city', data=df); plt.show()
df['city'].value_counts().plot(kind='pie', autopct='%1.1f%%'); plt.show()
```

### 6.2 Bivariate Analysis (two variables)
```python
# Numeric vs Numeric
sns.scatterplot(x='age', y='salary', data=df); plt.show()
print(df['age'].corr(df['salary']))

# Categorical vs Numeric
sns.boxplot(x='department', y='salary', data=df); plt.show()
df.groupby('department')['salary'].mean()

# Categorical vs Categorical
pd.crosstab(df['gender'], df['churn'])
sns.countplot(x='gender', hue='churn', data=df); plt.show()
```

### 6.3 Multivariate Analysis
```python
# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix"); plt.show()

# Pairplot - all numeric relationships at once
sns.pairplot(df, hue='churn')
plt.show()

# Grouped aggregation
df.groupby(['department','gender'])['salary'].agg(['mean','median','count'])

# Pivot table
pd.pivot_table(df, values='salary', index='department',
               columns='gender', aggfunc='mean')
```

### 6.4 Quick Automated EDA
```python
# pip install ydata-profiling
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="EDA Report", explorative=True)
profile.to_file("eda_report.html")
```

---

## 7. Feature Engineering

Creating, transforming, and selecting features to **improve model performance**.

> "Better features beat better algorithms." — a core DS principle

### Types of Feature Engineering
1. **Feature Creation** — build new features from existing ones
2. **Feature Transformation** — log, sqrt, polynomial, binning
3. **Feature Selection** — keep only useful features
4. **Dimensionality Reduction** — PCA, t-SNE

### 7.1 Feature Creation
```python
# Arithmetic combinations
df['price_per_sqft'] = df['price'] / df['area']
df['total_charges'] = df['monthly_charges'] * df['tenure']

# Date features
df['join_date'] = pd.to_datetime(df['join_date'])
df['year']       = df['join_date'].dt.year
df['month']      = df['join_date'].dt.month
df['dayofweek']  = df['join_date'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
df['days_since_join'] = (pd.Timestamp.now() - df['join_date']).dt.days

# Text features
df['name_length'] = df['name'].str.len()
df['word_count']  = df['review'].str.split().str.len()

# Binning (continuous → categorical)
df['age_group'] = pd.cut(df['age'],
                         bins=[0, 18, 35, 60, 100],
                         labels=['Teen','Young','Middle','Senior'])
```

### 7.2 Feature Transformation
```python
import numpy as np

# Log transform - fixes right-skewed data
df['log_salary'] = np.log1p(df['salary'])   # log1p handles zeros safely

# Square root transform
df['sqrt_area'] = np.sqrt(df['area'])

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['age','salary']])
```

### 7.3 Feature Selection

**a) Filter Method — Correlation**
```python
# Drop highly correlated features (multicollinearity)
corr_matrix = df.corr(numeric_only=True).abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
df = df.drop(columns=to_drop)
```

**b) Statistical Test — SelectKBest**
```python
from sklearn.feature_selection import SelectKBest, chi2, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print(selected_features)
```

**c) Wrapper Method — RFE**
```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

rfe = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=5)
rfe.fit(X, y)
print(X.columns[rfe.support_])
```

**d) Embedded Method — Feature Importance**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title("Feature Importance"); plt.show()
```

### 7.4 Dimensionality Reduction (PCA)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)   # keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

print("Original features:", X_scaled.shape[1])
print("After PCA:", X_pca.shape[1])
print("Explained variance:", pca.explained_variance_ratio_.sum())
```

---

## 8. Model Building

Training algorithms to learn patterns from the data.

### Choosing an Algorithm
| Problem Type | Algorithms |
|---|---|
| **Regression** | Linear, Ridge, Lasso, Decision Tree, Random Forest, XGBoost |
| **Classification** | Logistic Regression, KNN, SVM, Naive Bayes, Random Forest, XGBoost |
| **Clustering** | K-Means, DBSCAN, Hierarchical |
| **Dimensionality Reduction** | PCA, t-SNE, LDA |

### 8.1 Regression Example
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

### 8.2 Classification Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]   # probability of class 1
```

### 8.3 Comparing Multiple Models
```python
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(random_state=42),
    "KNN":                 KNeighborsClassifier(),
    "SVM":                 SVC()
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name:22} : {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 8.4 Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [5, 10, 20, None],
    'min_samples_split':[2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best score :", grid.best_score_)
best_model = grid.best_estimator_
```

### 8.5 Pipelines (Best Practice)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  RandomForestClassifier(random_state=42))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

---

## 9. Model Evaluation

Measuring how well the model performs on **unseen** data.

### 9.1 Classification Metrics

**Confusion Matrix**
```
                  Predicted
                 0        1
Actual  0  [  TN  ] [  FP  ]
        1  [  FN  ] [  TP  ]
```

| Metric | Formula | Meaning / When to use |
|---|---|---|
| **Accuracy** | (TP+TN)/Total | Overall correctness — bad for imbalanced data |
| **Precision** | TP/(TP+FP) | Of predicted positives, how many were right (spam filter) |
| **Recall (Sensitivity)** | TP/(TP+FN) | Of actual positives, how many we caught (disease, fraud) |
| **F1-Score** | 2·(P·R)/(P+R) | Harmonic mean — balanced metric |
| **ROC-AUC** | Area under ROC curve | Overall separability (0.5 = random, 1.0 = perfect) |

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_proba))

print(classification_report(y_test, y_pred))

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(); plt.show()
```

### 9.2 Regression Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Mean Absolute Error — average error size, same unit as target |
| **MSE** | Mean Squared Error — penalizes large errors more |
| **RMSE** | √MSE — same unit as target, most interpretable |
| **R² Score** | Proportion of variance explained (0 to 1, higher better) |
| **Adjusted R²** | R² adjusted for number of features |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")
```

### 9.3 Cross-Validation
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='f1')

print("Fold scores:", scores)
print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 9.4 Overfitting vs Underfitting

| | Training Score | Test Score | Diagnosis | Fix |
|---|---|---|---|---|
| **Underfitting** | Low | Low | Model too simple (high bias) | More features, complex model, less regularization |
| **Good Fit** | High | High | ✅ Ideal | — |
| **Overfitting** | Very High | Low | Model memorized data (high variance) | More data, regularization, simpler model, dropout |

```python
train_score = model.score(X_train, y_train)
test_score  = model.score(X_test, y_test)

print(f"Train: {train_score:.4f} | Test: {test_score:.4f}")
if train_score - test_score > 0.1:
    print("⚠️ Likely OVERFITTING")
```

---

## 10. Deployment

Putting the model into production so real users/systems can use it.

### 10.1 Saving & Loading the Model
```python
import joblib

# Save
joblib.dump(model,  "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load
model  = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

prediction = model.predict(new_data)
```

```python
# Alternative: pickle
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
```

### 10.2 Deploy as REST API (Flask)
```python
# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model  = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features = scaler.transform(features)

    pred  = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    return jsonify({
        "prediction": int(pred),
        "probability": float(proba)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```
```bash
# Test it
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [12, 65.5, 1, 3]}'
```

### 10.3 Deploy as Web App (Streamlit)
```python
# streamlit_app.py
import streamlit as st
import joblib
import numpy as np

model = joblib.load("churn_model.pkl")

st.title("🔮 Customer Churn Predictor")

tenure   = st.slider("Tenure (months)", 0, 72, 12)
charges  = st.number_input("Monthly Charges", 0.0, 200.0, 65.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

if st.button("Predict"):
    contract_map = {"Month-to-month":0, "One year":1, "Two year":2}
    features = np.array([[tenure, charges, contract_map[contract]]])
    pred  = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    if pred == 1:
        st.error(f"⚠️ Likely to churn (probability: {proba:.1%})")
    else:
        st.success(f"✅ Likely to stay (churn probability: {proba:.1%})")
```
```bash
streamlit run streamlit_app.py
```

### 10.4 Deployment Options
| Platform | Use case |
|---|---|
| **Flask / FastAPI** | Custom REST APIs |
| **Streamlit / Gradio** | Quick interactive demos |
| **Docker** | Containerized, reproducible deployment |
| **AWS SageMaker / GCP Vertex AI / Azure ML** | Enterprise-scale ML |
| **Heroku / Render / Railway** | Simple cloud hosting |
| **Power BI / Tableau** | Dashboard integration |

### 10.5 Monitoring & Maintenance
After deployment, keep watching for:
- **Data drift** — input data distribution changes over time
- **Concept drift** — the relationship between X and y changes
- **Performance decay** — accuracy drops in production
- **Latency & uptime** — system health

```python
# Simple drift check: compare training vs production distributions
from scipy.stats import ks_2samp

stat, p_value = ks_2samp(X_train['monthly_charges'], X_production['monthly_charges'])
if p_value < 0.05:
    print("⚠️ Data drift detected — consider retraining")
```

---

## 11. End-to-End Mini Project

A complete workflow in one script.

```python
# ============================================================
# END-TO-END DATA SCIENCE WORKFLOW
# Project: Customer Churn Prediction
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ----- STEP 1: PROBLEM DEFINITION -----
# Binary classification: predict if a customer will churn. Metric: F1 / Recall.

# ----- STEP 2: DATA COLLECTION -----
df = pd.read_csv("customer_churn.csv")
print("Shape:", df.shape)

# ----- STEP 3: DATA CLEANING -----
df = df.drop_duplicates()
df = df.drop(columns=['customer_id'])                      # irrelevant
df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
df['total_charges'] = df['total_charges'].fillna(df['total_charges'].median())

# Outlier capping
for col in ['monthly_charges', 'total_charges']:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

# ----- STEP 4: EDA -----
print(df['churn'].value_counts(normalize=True))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

# ----- STEP 5: FEATURE ENGINEERING -----
df['avg_monthly_spend'] = df['total_charges'] / df['tenure'].replace(0, 1)
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,100],
                            labels=['0-1yr','1-2yr','2-4yr','4yr+'])

# ----- STEP 6: PREPROCESSING -----
df = pd.get_dummies(df, drop_first=True)

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- STEP 7: MODEL BUILDING -----
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

params = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth':    [10, 20, None]
}
grid = GridSearchCV(pipe, params, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

# ----- STEP 8: MODEL EVALUATION -----
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.show()

# ----- STEP 9: DEPLOYMENT -----
joblib.dump(best_model, "churn_model_pipeline.pkl")
print("✅ Model saved and ready for deployment")
```

---

## 📌 Summary (Quick Revision)

| Step | Key Question | Main Tools |
|---|---|---|
| **1. Problem Definition** | What are we solving? | Business discussion, KPIs |
| **2. Data Collection** | Where is the data? | `read_csv`, SQL, APIs, scraping |
| **3. Data Cleaning** | Is the data trustworthy? | `dropna`, `fillna`, `drop_duplicates`, IQR |
| **4. Preprocessing** | Is it model-ready? | Encoding, Scaling, `train_test_split` |
| **5. EDA** | What does the data say? | `describe`, `seaborn`, `corr` |
| **6. Feature Engineering** | Can we make it more informative? | New features, PCA, SelectKBest |
| **7. Model Building** | Which algorithm fits best? | `sklearn`, GridSearchCV, Pipeline |
| **8. Model Evaluation** | How good is it really? | Accuracy, F1, RMSE, Cross-validation |
| **9. Deployment** | How do users access it? | `joblib`, Flask, Streamlit, Docker |

### Golden Rules 🏆
1. Never fit scalers/encoders on test data — **data leakage**.
2. Always split **before** preprocessing when possible (use Pipelines).
3. Accuracy is misleading on imbalanced data — use F1/Recall/ROC-AUC.
4. Use `random_state` everywhere for reproducibility.
5. The workflow is **iterative** — expect to loop back.
6. Document assumptions; a model nobody understands won't be trusted.

---

*📁 Previous: Part 1 → Introduction to Data Science*
*📁 Next: Part 3 → Python Libraries for Data Science (NumPy, Pandas, Matplotlib, Seaborn)*

---
### 🔖 Tags
`#DataScience` `#Workflow` `#Python` `#MachineLearning` `#Pipeline` `#Notes`
