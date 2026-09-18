# 📊 Data Science Notes — Part 1: Introduction to Data Science

> A complete theory + practical (Python) guide, structured for GitHub notes/wiki.

---

## Table of Contents
1. [What is Data Science?](#1-what-is-data-science)
2. [What is a Data Scientist?](#2-what-is-a-data-scientist)
3. [Data Science Lifecycle](#3-data-science-lifecycle)
4. [Data → Information → Knowledge](#4-data--information--knowledge)
5. [Applications of Data Science](#5-applications-of-data-science)
6. [Data Science vs AI vs ML](#6-data-science-vs-ai-vs-ml)
7. [Types of Data](#7-types-of-data)
8. [Structured vs Unstructured Data](#8-structured-vs-unstructured-data)

---

## 1. What is Data Science?

**Data Science** is an interdisciplinary field that uses scientific methods, statistics, algorithms, programming, and domain knowledge to extract meaningful insights and knowledge from structured and unstructured data.

It combines three core areas:

```
Data Science = Mathematics/Statistics + Programming (Computer Science) + Domain Knowledge
```

### Key characteristics
- It is **data-driven decision making** — using evidence from data instead of pure intuition.
- It involves collecting, cleaning, analyzing, visualizing, and modeling data.
- The end goal is usually **prediction**, **classification**, **insight generation**, or **automation**.

### The Data Science Venn Diagram (Conceptual)
```
        Mathematics & Statistics
                 ▲
                / \
               /   \
              / DS  \
             /-------\
            /         \
    Programming ---- Domain
     Skills           Expertise
```

### Practical Example
Data Science in action — loading a dataset and getting a quick statistical summary:

```python
import pandas as pd

# Load a sample dataset
df = pd.read_csv("sales_data.csv")

# Basic exploration - the very first step of any DS project
print(df.head())          # first 5 rows
print(df.info())          # column types, nulls
print(df.describe())      # statistical summary (mean, std, min, max)
```

---

## 2. What is a Data Scientist?

A **Data Scientist** is a professional who uses statistical analysis, machine learning, and programming to collect, process, and extract insights from large amounts of data to help organizations make better decisions.

### Core Skills of a Data Scientist
| Category | Skills |
|---|---|
| Programming | Python, R, SQL |
| Statistics & Math | Probability, Linear Algebra, Hypothesis Testing |
| Machine Learning | Regression, Classification, Clustering |
| Data Handling | Data Cleaning, Wrangling, ETL |
| Visualization | Matplotlib, Seaborn, Power BI, Tableau |
| Big Data Tools | Hadoop, Spark |
| Soft Skills | Communication, Business Understanding, Storytelling |

### A Data Scientist's Typical Toolkit (Python Example)
```python
# Common imports used daily by a Data Scientist
import numpy as np                # numerical computing
import pandas as pd               # data manipulation
import matplotlib.pyplot as plt   # visualization
import seaborn as sns             # statistical visualization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

### Related Roles (often confused)
- **Data Analyst** – focuses on descriptive analysis & dashboards (past/present).
- **Data Engineer** – builds pipelines & infrastructure for data flow.
- **ML Engineer** – deploys and productionizes machine learning models.
- **Data Scientist** – does all of the above at a research/insight level (past, present, and *future* via prediction).

---

## 3. Data Science Lifecycle

The Data Science Lifecycle is the step-by-step process followed to go from a raw business problem to a deployed, monitored solution.

```
 1. Business Understanding
        │
        ▼
 2. Data Collection
        │
        ▼
 3. Data Cleaning / Preparation
        │
        ▼
 4. Exploratory Data Analysis (EDA)
        │
        ▼
 5. Feature Engineering
        │
        ▼
 6. Model Building
        │
        ▼
 7. Model Evaluation
        │
        ▼
 8. Deployment
        │
        ▼
 9. Monitoring & Maintenance
```

### Stage-wise Explanation

| Stage | Description |
|---|---|
| **1. Business Understanding** | Define the problem & objective (e.g., "reduce customer churn"). |
| **2. Data Collection** | Gather data from databases, APIs, web scraping, sensors, etc. |
| **3. Data Cleaning** | Handle missing values, duplicates, outliers, wrong formats. |
| **4. EDA** | Explore patterns, correlations, distributions using stats & plots. |
| **5. Feature Engineering** | Create/transform variables to improve model performance. |
| **6. Model Building** | Apply ML/statistical algorithms to the prepared data. |
| **7. Model Evaluation** | Test accuracy/performance using metrics (RMSE, F1-score, etc.). |
| **8. Deployment** | Push the model into production (API, app, dashboard). |
| **9. Monitoring** | Track model performance over time; retrain if needed. |

### Practical Mini-Example (Steps 2–7 in code)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 2. Data Collection
df = pd.read_csv("customer_churn.csv")

# 3. Data Cleaning
df = df.dropna()                     # remove missing values
df = df.drop_duplicates()            # remove duplicate rows

# 4. EDA (quick check)
print(df['Churn'].value_counts())

# 5. Feature Engineering
X = df.drop('Churn', axis=1)
y = df['Churn']
X = pd.get_dummies(X)                # convert categorical → numeric

# 6. Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Model Evaluation
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

---

## 4. Data → Information → Knowledge

This is the **DIKW Pyramid** (Data, Information, Knowledge, Wisdom) — the foundation of how raw facts become smart decisions.

```
          WISDOM        (Applying knowledge with judgment/experience)
             ▲
        KNOWLEDGE       (Understanding relationships/patterns - "why")
             ▲
       INFORMATION      (Organized, structured data - "who/what/when")
             ▲
          DATA          (Raw, unprocessed facts & figures)
```

### Definitions
- **Data**: Raw, unorganized facts (numbers, text, images) with no context.
  - *Example: `23, 45, 19, 30`*
- **Information**: Data that has been processed/organized to have meaning.
  - *Example: "Average customer age is 29 years."*
- **Knowledge**: Information combined with experience/context to understand patterns and causes.
  - *Example: "Customers aged 25–30 buy more on weekends because of payday cycles."*
- **Wisdom**: Applying knowledge to make the *right* decision.
  - *Example: "Launch weekend-only discounts targeted at the 25–30 age group."*

### Practical Example
```python
# DATA - raw numbers
ages = [23, 45, 19, 30, 29, 31, 22]

# INFORMATION - processed/organized
average_age = sum(ages) / len(ages)
print(f"Average age: {average_age:.1f}")   # Information: "Average age is 28.4"

# KNOWLEDGE - pattern/insight derived from information
import pandas as pd
df = pd.DataFrame({'age': ages, 'purchase': [1,0,1,1,1,0,1]})
insight = df.groupby('purchase')['age'].mean()
print(insight)   # Knowledge: which age group buys more

# WISDOM = business decision made by a human/system based on the above knowledge
# e.g., "Target ads to age group 25-31 since they purchase more often"
```

---

## 5. Applications of Data Science

Data Science is used across almost every industry today:

| Industry | Application |
|---|---|
| **Healthcare** | Disease prediction, medical imaging analysis, drug discovery |
| **Finance** | Fraud detection, credit scoring, algorithmic trading |
| **E-Commerce** | Recommendation systems (Amazon, Flipkart), price optimization |
| **Social Media** | Sentiment analysis, content recommendation, ad targeting |
| **Transportation** | Route optimization (Uber, Ola), self-driving cars |
| **Entertainment** | Content recommendation (Netflix, Spotify) |
| **Manufacturing** | Predictive maintenance, quality control |
| **Agriculture** | Crop yield prediction, precision farming |
| **Sports** | Player performance analytics, match strategy |
| **Government** | Policy making, crime prediction, census analysis |

### Practical Example — Simple Recommendation Logic
```python
import pandas as pd

# Sample purchase data
data = {
    'user': ['A','A','B','B','C'],
    'product': ['Phone','Charger','Laptop','Mouse','Phone']
}
df = pd.DataFrame(data)

# Users who bought "Phone" also bought "Charger" → basic recommendation logic
phone_buyers = df[df['product'] == 'Phone']['user'].unique()
other_products = df[(df['user'].isin(phone_buyers)) & (df['product'] != 'Phone')]
print("People who bought Phone also bought:\n", other_products['product'].unique())
```

---

## 6. Data Science vs AI vs ML

These three terms are related but **not the same**. Think of them as concentric circles.

```
 ┌─────────────────────────────────────────────┐
 │      Artificial Intelligence (AI)            │
 │   (Machines mimicking human intelligence)    │
 │  ┌─────────────────────────────────────┐    │
 │  │     Machine Learning (ML)            │    │
 │  │ (Algorithms that learn from data)    │    │
 │  │   ┌─────────────────────────┐        │    │
 │  │   │   Deep Learning (DL)    │        │    │
 │  │   └─────────────────────────┘        │    │
 │  └─────────────────────────────────────┘    │
 └─────────────────────────────────────────────┘

     Data Science overlaps with ALL of the above,
     but also includes stats, visualization & business insight.
```

### Comparison Table

| Aspect | Data Science | Artificial Intelligence | Machine Learning |
|---|---|---|---|
| **Definition** | Extracting insights from data | Making machines simulate human intelligence | Subset of AI; systems learn from data |
| **Goal** | Insight & decision-making | Autonomous intelligent behavior | Prediction/pattern recognition |
| **Scope** | Broadest (includes ML, stats, viz, business) | Broad (includes ML, robotics, NLP, vision) | Narrower (a technique/tool) |
| **Tools** | Pandas, SQL, Excel, Tableau, ML | Neural Networks, Expert Systems | Scikit-learn, TensorFlow |
| **Output** | Reports, dashboards, models | Smart agents/systems | Trained predictive models |
| **Example** | Analyzing sales trends | Self-driving car | Spam email classifier |

### Practical Example — Where ML fits inside a DS project
```python
# This single line is "Machine Learning"...
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ...but the FULL Data Science process includes everything around it:
# business problem -> data collection -> cleaning -> EDA -> [ML above] -> 
# evaluation -> visualization -> business recommendation
```

---

## 7. Types of Data

Data can be classified in multiple ways. The two most common classifications:

### A) By Nature
1. **Qualitative (Categorical) Data** – describes qualities/characteristics
   - *Nominal*: no order → `Gender: Male/Female`, `City: Delhi/Mumbai`
   - *Ordinal*: has order → `Rating: Low/Medium/High`
2. **Quantitative (Numerical) Data** – describes numbers/measurable quantities
   - *Discrete*: countable → `Number of students: 30`
   - *Continuous*: measurable → `Height: 5.6 ft`, `Temperature: 36.6°C`

### B) By Source/Format
- **Structured Data** – tables, rows, columns (databases, Excel)
- **Semi-structured Data** – has some organization (JSON, XML, CSV logs)
- **Unstructured Data** – no fixed format (images, videos, text, audio)

### Practical Example
```python
import pandas as pd

data = {
    'Name': ['Anita', 'Rahul', 'Kavya'],          # Qualitative - Nominal
    'Satisfaction': ['High', 'Medium', 'Low'],     # Qualitative - Ordinal
    'Age': [23, 35, 29],                           # Quantitative - Discrete
    'Height_cm': [162.5, 175.2, 158.9]             # Quantitative - Continuous
}
df = pd.DataFrame(data)
print(df.dtypes)   # shows data type of each column
```

---

## 8. Structured vs Unstructured Data

| Feature | Structured Data | Unstructured Data |
|---|---|---|
| **Format** | Fixed rows & columns (tabular) | No predefined format |
| **Storage** | Relational Databases (SQL) | NoSQL, Data Lakes, File systems |
| **Examples** | Excel sheets, SQL tables, CSV | Images, videos, emails, social media posts, PDFs |
| **Ease of Analysis** | Easy — direct querying | Hard — needs NLP/CV/preprocessing |
| **Search** | Fast (indexed) | Slower, needs specialized tools |
| **% of real-world data** | ~20% | ~80% |
| **Tools used** | SQL, Excel, Pandas | NLP libraries, OpenCV, Deep Learning |

> **Semi-structured data** sits in between — e.g., JSON, XML, log files — it has *tags/keys* but not a strict table format.

### Practical Example — Structured Data
```python
import sqlite3
import pandas as pd

# Structured data lives naturally in databases/tables
conn = sqlite3.connect(':memory:')
df = pd.DataFrame({'id':[1,2,3], 'name':['Ravi','Meera','Sam'], 'marks':[88,92,79]})
df.to_sql('students', conn, index=False)

result = pd.read_sql("SELECT name, marks FROM students WHERE marks > 80", conn)
print(result)
```

### Practical Example — Unstructured Data (Text)
```python
# Unstructured data (raw text) needs preprocessing before analysis
text = "Data Science is amazing! I LOVE working with data 😊"

# Basic text preprocessing (common first step for unstructured data)
cleaned = text.lower().replace('!', '').replace('.', '')
words = cleaned.split()
print(words)
# ['data', 'science', 'is', 'amazing', 'i', 'love', 'working', 'with', 'data', '😊']
```

### Practical Example — Semi-structured Data (JSON)
```python
import json

json_data = '''
{
  "name": "Priya",
  "skills": ["Python", "SQL", "Machine Learning"],
  "experience": {"years": 2, "domain": "Finance"}
}
'''
parsed = json.loads(json_data)
print(parsed['skills'])           # ['Python', 'SQL', 'Machine Learning']
print(parsed['experience']['years'])   # 2
```

---

## 📌 Summary (Quick Revision)

- **Data Science** = Stats + Programming + Domain Knowledge → turns data into decisions.
- **Data Scientist** = professional who builds this pipeline end-to-end.
- **Lifecycle** = Business Understanding → Collection → Cleaning → EDA → Feature Engg → Modeling → Evaluation → Deployment → Monitoring.
- **DIKW Pyramid**: Data → Information → Knowledge → Wisdom.
- **Applications**: Healthcare, Finance, E-commerce, Entertainment, etc.
- **AI ⊃ ML ⊃ DL**, and **Data Science overlaps all three** plus stats/business.
- **Types of Data**: Qualitative (Nominal/Ordinal) vs Quantitative (Discrete/Continuous).
- **Structured** (tabular, SQL) vs **Semi-structured** (JSON/XML) vs **Unstructured** (text/images/video).

---

*📁 Next: Part 2 → Statistics for Data Science (Descriptive & Inferential Statistics with Python)*

---
### 🔖 Tags
`#DataScience` `#Python` `#MachineLearning` `#Notes` `#Beginner`
