# 🌊 Data Science Notes — Part 7: Seaborn

> A statistical visualization library built on top of Matplotlib — theory + practical syntax and code examples for every core plot type.

---

## Table of Contents
1. [Introduction to Seaborn](#1-introduction-to-seaborn)
2. [Distribution Plots](#2-distribution-plots)
3. [Count Plots](#3-count-plots)
4. [Box Plots](#4-box-plots)
5. [Violin Plots](#5-violin-plots)
6. [Scatter Plots](#6-scatter-plots)
7. [Heatmaps](#7-heatmaps)
8. [Correlation Visualization](#8-correlation-visualization)
9. [Practical Mini Project](#9-practical-mini-project)
10. [Cheatsheet](#10-cheatsheet)

---

## 1. Introduction to Seaborn

**Seaborn** is a Python data visualization library built on top of **Matplotlib**. It provides a high-level interface for drawing attractive, informative **statistical graphics** with far less code than raw Matplotlib.

### Why Seaborn over plain Matplotlib?
| Feature | Matplotlib | Seaborn |
|---|---|---|
| **Default aesthetics** | Basic, needs manual styling | Beautiful themes out of the box |
| **DataFrame integration** | Manual column extraction | Pass a DataFrame + column names directly |
| **Statistical plots** | Must build manually | Built-in: box, violin, regression, KDE |
| **Categorical grouping** | Manual loops | Built-in `hue`, `col`, `row` parameters |
| **Code required** | More | Significantly less |

> 💡 Seaborn doesn't replace Matplotlib — it **sits on top of it**. You can still use `plt.title()`, `plt.xlabel()`, `plt.show()`, etc. alongside Seaborn functions.

### Installation & Import
```bash
pip install seaborn
```
```python
import seaborn as sns              # 'sns' is the universal convention
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print(sns.__version__)

# Set a theme (do this once at the top of your script/notebook)
sns.set_theme(style="whitegrid")   # 'darkgrid','whitegrid','dark','white','ticks'
sns.set_palette("Set2")            # color palette
```

### Seaborn's Built-in Sample Datasets (great for practice)
```python
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
titanic = sns.load_dataset("titanic")
print(sns.get_dataset_names())     # list all available datasets
print(tips.head())
```

### Figure-level vs Axes-level Functions
This is Seaborn's most important architectural concept:

| Type | Returns | Supports subplots via `col`/`row` | Examples |
|---|---|---|---|
| **Axes-level** | A single `Axes` object | ❌ (use `plt.subplots` manually) | `histplot`, `boxplot`, `violinplot`, `scatterplot`, `heatmap`, `countplot` |
| **Figure-level** | A `FacetGrid`/`JointGrid` | ✅ (built-in faceting) | `displot`, `catplot`, `relplot`, `lmplot`, `jointplot`, `pairplot` |

```python
# Axes-level - integrates with plt.subplots()
fig, ax = plt.subplots()
sns.boxplot(data=tips, x='day', y='total_bill', ax=ax)

# Figure-level - manages its own figure, supports faceting
sns.catplot(data=tips, x='day', y='total_bill', kind='box', col='time')
```

---

## 2. Distribution Plots

Distribution plots show how the values of a **single numeric variable** are spread out — the Seaborn equivalent (and upgrade) of a Matplotlib histogram.

### 2.1 histplot() — Histogram
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

sns.histplot(data=tips, x='total_bill')
plt.show()

# Customizing bins and appearance
sns.histplot(data=tips, x='total_bill', bins=20, color='skyblue', edgecolor='black')
plt.show()

# Add a KDE curve on top
sns.histplot(data=tips, x='total_bill', kde=True)
plt.show()

# Split by category with hue
sns.histplot(data=tips, x='total_bill', hue='sex', kde=True, alpha=0.5)
plt.show()

# Normalize to show density/probability instead of raw counts
sns.histplot(data=tips, x='total_bill', stat='density')
sns.histplot(data=tips, x='total_bill', stat='probability')
plt.show()
```

### 2.2 kdeplot() — Kernel Density Estimate (smoothed curve)
```python
sns.kdeplot(data=tips, x='total_bill')
plt.show()

# Fill under the curve
sns.kdeplot(data=tips, x='total_bill', fill=True, color='purple')
plt.show()

# Compare distributions across a category
sns.kdeplot(data=tips, x='total_bill', hue='time', fill=True, alpha=0.4)
plt.show()

# 2-D KDE (bivariate) - like a smoothed scatter/heatmap
sns.kdeplot(data=tips, x='total_bill', y='tip', fill=True, cmap='Blues')
plt.show()
```

### 2.3 displot() — Figure-level Distribution Plot (supports faceting)
```python
sns.displot(data=tips, x='total_bill', kind='hist', kde=True)
plt.show()

sns.displot(data=tips, x='total_bill', kind='kde')
plt.show()

sns.displot(data=tips, x='total_bill', kind='ecdf')   # cumulative distribution
plt.show()

# Faceted by column - separate subplot per category
sns.displot(data=tips, x='total_bill', col='time', kind='hist', kde=True)
plt.show()
```

### 2.4 rugplot() — Individual Data Points as Ticks
```python
sns.histplot(data=tips, x='total_bill', kde=True)
sns.rugplot(data=tips, x='total_bill')       # adds tick marks below the plot
plt.show()
```

### 2.5 jointplot() — Distribution + Relationship Combined
```python
sns.jointplot(data=tips, x='total_bill', y='tip', kind='scatter')
plt.show()

sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')     # for dense data
sns.jointplot(data=tips, x='total_bill', y='tip', kind='kde')     # smoothed density
sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')     # with regression line
plt.show()
```

### When to Use Distribution Plots
✅ Checking if a variable is normally distributed / skewed
✅ Comparing distributions across categories (`hue`)
✅ Spotting multi-modal patterns (multiple peaks)
❌ Not for categorical count comparisons (use countplot instead)

---

## 3. Count Plots

A **countplot** shows the **frequency of each category** in a categorical column — Seaborn's version of a bar chart, but it counts the rows for you automatically (no need to pre-aggregate with `.value_counts()`).

### 3.1 Basic Count Plot
```python
titanic = sns.load_dataset("titanic")

sns.countplot(data=titanic, x='class')
plt.show()

sns.countplot(data=titanic, y='class')        # horizontal
plt.show()
```

### 3.2 Grouping with hue
```python
sns.countplot(data=titanic, x='class', hue='sex')
plt.show()

sns.countplot(data=titanic, x='class', hue='survived', palette='Set2')
plt.title('Survival Count by Class')
plt.show()
```

### 3.3 Ordering Bars
```python
# By explicit category order
sns.countplot(data=titanic, x='class', order=['Third','Second','First'])

# By frequency (most common first)
order = titanic['class'].value_counts().index
sns.countplot(data=titanic, x='class', order=order)
plt.show()
```

### 3.4 Styling
```python
sns.countplot(data=titanic, x='embark_town', palette='pastel', edgecolor='black')
plt.xticks(rotation=30)
plt.xlabel('Port of Embarkation')
plt.ylabel('Number of Passengers')
plt.title('Passengers by Embarkation Port')
plt.show()
```

### 3.5 Adding Count Labels on Bars
```python
ax = sns.countplot(data=titanic, x='class')
for container in ax.containers:
    ax.bar_label(container)
plt.show()
```

### 3.6 catplot() — Figure-level Version (supports faceting)
```python
sns.catplot(data=titanic, x='class', hue='sex', kind='count', col='survived')
plt.show()
```

### countplot vs barplot — Key Difference
| | `countplot()` | `barplot()` |
|---|---|---|
| What it plots | **Frequency** of categories (auto-counted) | Aggregated **statistic** (default: mean) of a numeric column per category |
| Needs a `y` value | ❌ No — it counts rows | ✅ Yes — needs a numeric column |
| Example | "How many passengers per class?" | "What's the average fare per class?" |

```python
# barplot example - mean of a numeric column, per category, with confidence interval
sns.barplot(data=titanic, x='class', y='fare', hue='sex')
plt.show()
```

### When to Use Count Plots
✅ Visualizing frequency/imbalance of categorical variables (e.g., class distribution before modeling)
✅ Comparing category counts across a second categorical variable (`hue`)
❌ Not for numeric summaries (use `barplot` or `boxplot` instead)

---

## 4. Box Plots

Box plots ("box-and-whisker" plots) summarize a numeric variable's **distribution, spread, and outliers** using quartiles.

### 4.1 Anatomy of a Box Plot
```
         Outliers
             •
             │
         ┌───┴───┐  ← Whisker (max within 1.5×IQR)
         │       │
    Q3 ──┤───────├──  75th percentile
         │       │
median ──┤═══════├──  50th percentile (the line inside the box)
         │       │
    Q1 ──┤───────├──  25th percentile
         │       │
         └───┬───┘  ← Whisker (min within 1.5×IQR)
             │
             •
         Outliers

    IQR = Q3 - Q1   (height of the box)
    Whisker limit = Q1 - 1.5×IQR  to  Q3 + 1.5×IQR
    Points beyond whiskers = outliers (plotted as individual dots)
```

### 4.2 Basic Box Plot
```python
tips = sns.load_dataset("tips")

sns.boxplot(data=tips, y='total_bill')          # single variable
plt.show()

sns.boxplot(data=tips, x='day', y='total_bill')  # numeric across categories
plt.show()
```

### 4.3 Grouping with hue
```python
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex')
plt.show()

sns.boxplot(data=tips, x='day', y='total_bill', hue='smoker', palette='Set3')
plt.legend(title='Smoker', bbox_to_anchor=(1.02,1), loc='upper left')
plt.show()
```

### 4.4 Horizontal Box Plot
```python
sns.boxplot(data=tips, x='total_bill', y='day')     # swap x and y
plt.show()
```

### 4.5 Customizing Appearance
```python
sns.boxplot(data=tips, x='day', y='total_bill',
           palette='coolwarm',
           width=0.5,
           linewidth=1.5,
           showmeans=True,                # show a marker for the mean too
           meanprops={'marker':'D', 'markerfacecolor':'white'})
plt.show()

# Removing/customizing outlier markers
sns.boxplot(data=tips, x='day', y='total_bill', showfliers=False)   # hide outliers
sns.boxplot(data=tips, x='day', y='total_bill',
           flierprops={'marker':'x', 'markersize':8, 'color':'red'})
plt.show()
```

### 4.6 Multiple Numeric Columns (Wide-form Data)
```python
iris = sns.load_dataset("iris")
sns.boxplot(data=iris.drop(columns='species'))     # box per numeric column
plt.xticks(rotation=15)
plt.show()
```

### 4.7 Using Box Plots for Outlier Detection (DS Workflow)
```python
numeric_cols = ['total_bill', 'tip', 'size']

fig, axes = plt.subplots(1, len(numeric_cols), figsize=(12, 4))
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=tips, y=col, ax=axes[i], color='lightblue')
    axes[i].set_title(f'{col} — outlier check')
plt.tight_layout()
plt.show()

# Programmatically identifying outliers behind the plot
Q1 = tips['total_bill'].quantile(0.25)
Q3 = tips['total_bill'].quantile(0.75)
IQR = Q3 - Q1
outliers = tips[(tips['total_bill'] < Q1-1.5*IQR) | (tips['total_bill'] > Q3+1.5*IQR)]
print(f"Outliers found: {len(outliers)}")
```

### When to Use Box Plots
✅ Comparing distributions/spread across categories at a glance
✅ Spotting outliers quickly
✅ Comparing median and IQR (robust to outliers, unlike mean/std)
❌ Doesn't show the full shape of the distribution (can't see multi-modal patterns — use violin plot for that)

---

## 5. Violin Plots

A violin plot combines a **box plot** with a **KDE (density curve)**, showing both summary statistics AND the full shape of the distribution — including multiple peaks that a box plot would hide.

### 5.1 Basic Violin Plot
```python
sns.violinplot(data=tips, x='day', y='total_bill')
plt.show()
```

### 5.2 Comparing Box Plot vs Violin Plot Side-by-Side
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=tips, x='day', y='total_bill', ax=axes[0])
axes[0].set_title('Box Plot — summary stats only')
sns.violinplot(data=tips, x='day', y='total_bill', ax=axes[1])
axes[1].set_title('Violin Plot — full distribution shape')
plt.tight_layout()
plt.show()
```

### 5.3 Grouping with hue
```python
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex')
plt.show()
```

### 5.4 Split Violin (compact comparison for binary hue)
```python
# split=True merges both hue halves into ONE violin per category - great for A/B comparison
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True)
plt.show()
```

### 5.5 Customizing Inner Display
```python
sns.violinplot(data=tips, x='day', y='total_bill', inner='box')       # default: mini box plot inside
sns.violinplot(data=tips, x='day', y='total_bill', inner='quartile')  # quartile lines
sns.violinplot(data=tips, x='day', y='total_bill', inner='point')     # individual points
sns.violinplot(data=tips, x='day', y='total_bill', inner='stick')     # each observation as a tick
plt.show()
```

### 5.6 Styling
```python
sns.violinplot(data=tips, x='day', y='total_bill',
              palette='muted',
              linewidth=1.2,
              width=0.8)
plt.title('Bill Distribution by Day')
plt.show()
```

### 5.7 When Violin Reveals What Box Plot Hides (Bimodal Example)
```python
import numpy as np
# Simulated bimodal data (two peaks) - e.g. customers who tip generously vs poorly
bimodal = np.concatenate([np.random.normal(10, 2, 150), np.random.normal(25, 2, 150)])
df_bi = pd.DataFrame({'value': bimodal, 'group': 'A'})

fig, axes = plt.subplots(1, 2, figsize=(10,4))
sns.boxplot(data=df_bi, y='value', ax=axes[0])
axes[0].set_title('Box Plot — hides the two peaks!')
sns.violinplot(data=df_bi, y='value', ax=axes[1])
axes[1].set_title('Violin Plot — reveals bimodal shape')
plt.tight_layout()
plt.show()
```

### Box Plot vs Violin Plot — When to Use Which
| Situation | Use |
|---|---|
| Quick summary / many categories to compare | **Box plot** (cleaner, less visual clutter) |
| Need to see full distribution shape, detect multi-modality | **Violin plot** |
| Presenting to non-technical audience | **Box plot** (simpler to explain) |
| Comparing exactly two groups compactly | **Split violin plot** |

---

## 6. Scatter Plots

Seaborn's `scatterplot()` shows the relationship between two numeric variables, with built-in support for encoding extra dimensions via `hue`, `size`, and `style`.

### 6.1 Basic Scatter Plot
```python
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.show()
```

### 6.2 Encoding a Third Variable with hue (Color)
```python
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time')
plt.show()

sns.scatterplot(data=tips, x='total_bill', y='tip', hue='day', palette='deep')
plt.show()

# hue with a NUMERIC variable (continuous color scale)
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='size', palette='viridis')
plt.show()
```

### 6.3 Encoding a Fourth Variable with size and style
```python
sns.scatterplot(data=tips, x='total_bill', y='tip',
                hue='time', size='size', style='smoker',
                sizes=(20, 200), alpha=0.7)
plt.show()
```

### 6.4 scatterplot with Regression Line — regplot() / lmplot()
```python
sns.regplot(data=tips, x='total_bill', y='tip')          # scatter + regression line + CI band
plt.show()

# lmplot - figure-level, supports faceting by category
sns.lmplot(data=tips, x='total_bill', y='tip', hue='smoker')
plt.show()

sns.lmplot(data=tips, x='total_bill', y='tip', col='time', hue='sex')
plt.show()
```

### 6.5 relplot() — Figure-level Scatter/Line (supports faceting)
```python
sns.relplot(data=tips, x='total_bill', y='tip', hue='day', kind='scatter')
plt.show()

# Faceted grid - separate subplot per combination of row/col
sns.relplot(data=tips, x='total_bill', y='tip', hue='sex', col='time', row='smoker')
plt.show()
```

### 6.6 pairplot() — All Numeric Pairs at Once
```python
iris = sns.load_dataset("iris")

sns.pairplot(iris)                       # scatter for every numeric pair + histogram on diagonal
plt.show()

sns.pairplot(iris, hue='species')        # color by category - excellent for classification EDA
plt.show()

sns.pairplot(iris, hue='species', diag_kind='kde', corner=True)   # only lower triangle
plt.show()
```

### When to Use Scatter Plots
✅ Relationship/correlation between two numeric variables
✅ Encoding 3-4 dimensions at once (`hue`, `size`, `style`)
✅ `pairplot()` for a fast first-look at a whole dataset before modeling
❌ Not ideal for very large datasets (overplotting) — consider `sns.kdeplot()` (2-D) or lowering `alpha`

---

## 7. Heatmaps

Heatmaps display a **matrix of values as color**, making patterns in large grids of numbers immediately visible — most commonly used for correlation matrices, confusion matrices, and pivot tables.

### 7.1 Basic Heatmap
```python
import numpy as np

data = np.random.rand(6, 6)
sns.heatmap(data)
plt.show()
```

### 7.2 Annotating Values
```python
sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
```

### 7.3 Common Parameters
```python
sns.heatmap(data,
           annot=True,          # show numbers on each cell
           fmt='.2f',           # number format (2 decimal places)
           cmap='viridis',      # color scheme
           linewidths=0.5,      # gridlines between cells
           linecolor='white',
           cbar=True,           # show the color scale bar
           cbar_kws={'label':'Value'},
           vmin=0, vmax=1,      # fix the color scale range
           square=True)         # make cells square
plt.show()
```

### 7.4 Heatmap from a Pivot Table (very common EDA pattern)
```python
tips = sns.load_dataset("tips")

pivot = tips.pivot_table(values='total_bill', index='day', columns='time', aggfunc='mean')
print(pivot)

sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu')
plt.title('Average Bill by Day and Time')
plt.show()
```

### 7.5 Heatmap for Missing Value Patterns
```python
df = pd.DataFrame({
    'a': [1, np.nan, 3, 4],
    'b': [np.nan, 2, 3, np.nan],
    'c': [1, 2, 3, 4]
})

sns.heatmap(df.isnull(), cbar=False, cmap='Reds', yticklabels=False)
plt.title('Missing Value Pattern')
plt.show()
```

### 7.6 Masking Part of a Heatmap (useful for correlation triangles)
```python
corr = tips.select_dtypes(include='number').corr()

mask = np.triu(np.ones_like(corr, dtype=bool))    # mask the upper triangle
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0)
plt.show()
```

### When to Use Heatmaps
✅ Correlation matrices
✅ Confusion matrices (classification model evaluation)
✅ Any matrix/pivot table of numeric values
✅ Visualizing missing data patterns
❌ Not suited to non-matrix data

---

## 8. Correlation Visualization

Visualizing how numeric variables relate to each other — a critical EDA step before feature selection or modeling.

### 8.1 Computing Correlation
```python
tips = sns.load_dataset("tips")

corr = tips.select_dtypes(include='number').corr()      # Pearson by default
print(corr)

corr_spearman = tips.select_dtypes(include='number').corr(method='spearman')
corr_kendall  = tips.select_dtypes(include='number').corr(method='kendall')
```

| Method | Measures | When to use |
|---|---|---|
| **Pearson** (default) | Linear relationship | Numeric, roughly normal, linear trend |
| **Spearman** | Monotonic relationship (rank-based) | Non-linear but consistent direction, has outliers |
| **Kendall** | Ordinal association | Small samples, ordinal data |

### 8.2 Correlation Heatmap (the standard EDA visual)
```python
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
           square=True, linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
```

### 8.3 Interpreting Correlation Values
| Value range | Strength | Interpretation |
|---|---|---|
| `0.8 to 1.0` | Very strong positive | As X increases, Y strongly increases |
| `0.5 to 0.8` | Strong positive | Clear positive relationship |
| `0.3 to 0.5` | Moderate positive | Some positive relationship |
| `-0.3 to 0.3` | Weak / none | Little to no linear relationship |
| `-0.5 to -0.3` | Moderate negative | Some negative relationship |
| `-1.0 to -0.8` | Very strong negative | As X increases, Y strongly decreases |

> ⚠️ **Correlation ≠ Causation.** A strong correlation doesn't prove one variable causes the other.

### 8.4 Pairwise Scatter with Correlation — pairplot() + corr()
```python
iris = sns.load_dataset("iris")

sns.pairplot(iris, hue='species', corner=True)
plt.show()

print(iris.select_dtypes(include='number').corr())
```

### 8.5 regplot for a Single Pair (with correlation printed)
```python
from scipy.stats import pearsonr

r, p_value = pearsonr(tips['total_bill'], tips['tip'])

sns.regplot(data=tips, x='total_bill', y='tip', line_kws={'color':'red'})
plt.title(f'Correlation: r = {r:.2f}, p-value = {p_value:.4f}')
plt.show()
```

### 8.6 Finding Highly Correlated Feature Pairs (Multicollinearity Check)
```python
corr = tips.select_dtypes(include='number').corr().abs()

# Get only the upper triangle (avoid duplicate pairs and self-correlation)
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

high_corr_pairs = [(col, row, upper.loc[row, col])
                   for col in upper.columns for row in upper.index
                   if pd.notnull(upper.loc[row, col]) and upper.loc[row, col] > 0.7]

print("Highly correlated pairs (r > 0.7):", high_corr_pairs)
```

### 8.7 Correlation with the Target Variable (Feature Selection Use Case)
```python
titanic = sns.load_dataset("titanic")
numeric_titanic = titanic.select_dtypes(include='number')

target_corr = numeric_titanic.corr()['survived'].sort_values(ascending=False)
print(target_corr)

plt.figure(figsize=(6,5))
sns.barplot(x=target_corr.values, y=target_corr.index, palette='coolwarm')
plt.title('Feature Correlation with Survival')
plt.xlabel('Correlation Coefficient')
plt.show()
```

### 8.8 clustermap() — Heatmap with Hierarchical Clustering
```python
# Automatically groups similar rows/columns together - reveals hidden structure
sns.clustermap(corr, annot=True, cmap='coolwarm', center=0, figsize=(7,7))
plt.show()
```

---

## 9. Practical Mini Project

```python
# ============================================================
# SEABORN MINI PROJECT: Exploratory Data Analysis Dashboard
# Dataset: Titanic
# ============================================================
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")
titanic = sns.load_dataset("titanic").dropna(subset=['age', 'embarked'])

print(titanic.shape)
print(titanic.head())

# --- Build a comprehensive EDA dashboard ---
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Titanic Dataset — Exploratory Data Analysis', fontsize=18, fontweight='bold')

# 1. Distribution plot - Age
sns.histplot(data=titanic, x='age', kde=True, ax=axes[0,0], color='steelblue')
axes[0,0].set_title('Age Distribution')

# 2. Distribution split by survival
sns.kdeplot(data=titanic, x='age', hue='survived', fill=True, alpha=0.4, ax=axes[0,1])
axes[0,1].set_title('Age Distribution by Survival')

# 3. Count plot - Passenger class
sns.countplot(data=titanic, x='class', hue='survived', ax=axes[0,2], palette='Set2')
axes[0,2].set_title('Survival Count by Class')

# 4. Box plot - Fare by class
sns.boxplot(data=titanic, x='class', y='fare', ax=axes[1,0], palette='pastel')
axes[1,0].set_title('Fare Distribution by Class')
axes[1,0].set_ylim(0, 300)

# 5. Violin plot - Age by survival and sex
sns.violinplot(data=titanic, x='survived', y='age', hue='sex', split=True, ax=axes[1,1])
axes[1,1].set_title('Age Distribution: Survived x Gender')

# 6. Scatter plot - Age vs Fare, colored by survival
sns.scatterplot(data=titanic, x='age', y='fare', hue='survived',
                style='sex', alpha=0.6, ax=axes[1,2])
axes[1,2].set_title('Age vs Fare (Survival Highlighted)')
axes[1,2].set_ylim(0, 300)

# 7. Count plot - Embarkation port
sns.countplot(data=titanic, x='embarked', ax=axes[2,0], palette='muted')
axes[2,0].set_title('Passengers by Embarkation Port')

# 8. Heatmap - Correlation matrix
numeric_cols = titanic.select_dtypes(include='number')
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[2,1])
axes[2,1].set_title('Correlation Matrix')

# 9. Bar plot - Survival rate by class and sex
survival_rate = titanic.groupby(['class','sex'])['survived'].mean().reset_index()
sns.barplot(data=survival_rate, x='class', y='survived', hue='sex', ax=axes[2,2], palette='coolwarm')
axes[2,2].set_title('Survival Rate by Class & Gender')
axes[2,2].set_ylabel('Survival Rate')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('titanic_eda_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Key statistical insights ---
print("\n--- Key Insights ---")
print("Overall survival rate:", f"{titanic['survived'].mean():.1%}")
print("\nSurvival rate by class:\n", titanic.groupby('class')['survived'].mean())
print("\nSurvival rate by sex:\n", titanic.groupby('sex')['survived'].mean())
print("\nStrongest correlations with survival:\n",
      corr['survived'].sort_values(ascending=False))
```

---

## 10. Cheatsheet

### Setup
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
df = sns.load_dataset("tips")
```

### Distribution
```python
sns.histplot(data=df, x='col', kde=True, hue='cat')
sns.kdeplot(data=df, x='col', fill=True)
sns.displot(data=df, x='col', kind='hist', col='cat')
```

### Categorical
```python
sns.countplot(data=df, x='cat', hue='cat2')
sns.barplot(data=df, x='cat', y='num')
sns.catplot(data=df, x='cat', y='num', kind='box')
```

### Spread / Outliers
```python
sns.boxplot(data=df, x='cat', y='num', hue='cat2')
sns.violinplot(data=df, x='cat', y='num', split=True)
```

### Relationship
```python
sns.scatterplot(data=df, x='num1', y='num2', hue='cat', size='num3')
sns.regplot(data=df, x='num1', y='num2')
sns.pairplot(df, hue='cat')
```

### Matrix / Correlation
```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
sns.clustermap(df.corr(), annot=True)
```

### Chart Selection Guide
| Goal | Function |
|---|---|
| Shape of one numeric variable | `histplot()`, `kdeplot()` |
| Frequency of categories | `countplot()` |
| Spread + outliers across categories | `boxplot()` |
| Full distribution shape across categories | `violinplot()` |
| Relationship between two numeric variables | `scatterplot()`, `regplot()` |
| Matrix of values / correlation | `heatmap()` |
| All numeric variable relationships at once | `pairplot()` |

### Key Rules to Remember 🏆
1. Seaborn works **directly with DataFrames** — pass `data=df, x='col', y='col'` rather than extracting arrays manually.
2. **Figure-level** functions (`displot`, `catplot`, `relplot`, `lmplot`) support `col=`/`row=` faceting; **axes-level** functions (`histplot`, `boxplot`, etc.) plug into `plt.subplots()` via `ax=`.
3. Use `hue=` to add a third categorical dimension to almost any plot.
4. `countplot()` counts rows automatically; `barplot()` needs a numeric `y` to aggregate.
5. Violin plots reveal multi-modal distributions that box plots hide.
6. Always set `sns.set_theme()` once at the top — cleaner default styling.
7. Correlation ≠ causation — always state this caveat when presenting a correlation heatmap.
8. `pairplot()` is an excellent one-line first look at a new dataset before deeper EDA.

---

## 📌 Summary

| Topic | Key Point |
|---|---|
| **Distribution plots** | `histplot()`, `kdeplot()`, `displot()` — shape of one numeric variable |
| **Count plots** | `countplot()` — auto-counted frequency of categories |
| **Box plots** | `boxplot()` — quartiles, median, IQR-based outliers |
| **Violin plots** | `violinplot()` — box plot + full density shape, reveals multi-modality |
| **Scatter plots** | `scatterplot()`, `regplot()` — relationship between two numeric variables |
| **Heatmaps** | `heatmap()` — color-coded matrix, ideal for correlation/pivot tables |
| **Correlation visualization** | `.corr()` + `heatmap()` — find linear relationships, check multicollinearity |

---

*📁 Previous: Part 6 → Matplotlib*
*📁 Next: Part 8 → Statistics for Data Science*

---
### 🔖 Tags
`#Seaborn` `#Python` `#DataVisualization` `#DataScience` `#EDA` `#Notes`
