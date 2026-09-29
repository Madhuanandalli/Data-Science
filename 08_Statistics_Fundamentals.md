# 📐 Data Science Notes — Part 8: Statistics Fundamentals

> The mathematical foundation behind every EDA step and ML model — theory + practical Python (NumPy, Pandas, SciPy) syntax and code examples.

---

## Table of Contents
1. [Why Statistics Matters in Data Science](#1-why-statistics-matters-in-data-science)
2. [Population vs Sample](#2-population-vs-sample)
3. [Mean](#3-mean)
4. [Median](#4-median)
5. [Mode](#5-mode)
6. [Range](#6-range)
7. [Variance](#7-variance)
8. [Standard Deviation](#8-standard-deviation)
9. [Percentiles](#9-percentiles)
10. [Quartiles](#10-quartiles)
11. [IQR](#11-iqr)
12. [Bringing It All Together](#12-bringing-it-all-together)
13. [Practical Mini Project](#13-practical-mini-project)
14. [Cheatsheet](#14-cheatsheet)

---

## 1. Why Statistics Matters in Data Science

Statistics is the **language of data**. Every EDA step, every model evaluation metric, and every business insight ultimately reduces to statistical concepts.

```
Statistics
│
├── Descriptive Statistics  → Summarizing data (this note)
│     "What does the data look like?"
│     Mean, Median, Mode, Variance, Std, Percentiles...
│
└── Inferential Statistics  → Drawing conclusions beyond the data
      "What can we conclude about the population from a sample?"
      Hypothesis testing, confidence intervals, p-values...
```

This part covers **Descriptive Statistics** — the measures used to summarize and understand a dataset. Inferential statistics (hypothesis testing, confidence intervals) is covered separately.

### Two Categories of Descriptive Statistics
| Category | Purpose | Measures |
|---|---|---|
| **Measures of Central Tendency** | Where is the "center" of the data? | Mean, Median, Mode |
| **Measures of Dispersion (Spread)** | How spread out is the data? | Range, Variance, Std Dev, IQR |
| **Measures of Position** | Where does a value sit relative to others? | Percentiles, Quartiles |

```python
import numpy as np
import pandas as pd
from scipy import stats
```

---

## 2. Population vs Sample

This is the **most fundamental distinction** in statistics — and it changes which formula you use.

### Definitions
| | Population | Sample |
|---|---|---|
| **Definition** | The **entire** group you want to study | A **subset** drawn from the population |
| **Symbol (mean)** | μ (mu) | x̄ (x-bar) |
| **Symbol (std dev)** | σ (sigma) | s |
| **Symbol (size)** | N | n |
| **Symbol (variance)** | σ² | s² |
| **Cost to measure** | Often expensive/impossible | Practical, affordable |
| **Example** | All 1.4 billion people in India | A survey of 5,000 people in India |

```
┌─────────────────────────────────────┐
│           POPULATION (N)             │
│   ┌───────────────────────────┐      │
│   │        SAMPLE (n)          │      │
│   │   drawn to represent the   │      │
│   │        population          │      │
│   └───────────────────────────┘      │
│                                       │
└─────────────────────────────────────┘
```

### Why the Distinction Matters
In almost all real-world Data Science work, we have a **sample**, not the full population — and we use that sample to **estimate** population parameters. This affects certain formulas (variance and standard deviation use `n-1` for a sample instead of `n`).

```python
import numpy as np

data = [23, 45, 12, 67, 34, 89, 21, 55]

# Population statistics (ddof=0, the NumPy default)
pop_var = np.var(data, ddof=0)
pop_std = np.std(data, ddof=0)

# Sample statistics (ddof=1)
sample_var = np.var(data, ddof=1)
sample_std = np.std(data, ddof=1)

print(f"Population variance: {pop_var:.2f}, std: {pop_std:.2f}")
print(f"Sample variance    : {sample_var:.2f}, std: {sample_std:.2f}")
```

### Why Sample Variance Divides by (n-1), Not n
Using `n` in the sample variance formula **underestimates** the true population variance, because a sample's own mean is calculated FROM the sample — pulling the data slightly closer to its own mean than to the true population mean. Dividing by `n-1` (called **Bessel's correction**) corrects this bias, making it an unbiased estimator.

```python
# Demonstration: sample variance with n-1 is a better (unbiased) estimator
np.random.seed(42)
population = np.random.normal(50, 10, 100000)   # "true" population
true_var = np.var(population)                    # ≈ 100

sample_vars_n   = []
sample_vars_n1  = []
for _ in range(1000):
    sample = np.random.choice(population, size=30, replace=False)
    sample_vars_n.append(np.var(sample, ddof=0))     # divide by n
    sample_vars_n1.append(np.var(sample, ddof=1))    # divide by n-1

print(f"True population variance : {true_var:.2f}")
print(f"Avg sample var (ddof=0)  : {np.mean(sample_vars_n):.2f}  ← biased, too low")
print(f"Avg sample var (ddof=1)  : {np.mean(sample_vars_n1):.2f}  ← unbiased, closer to true")
```

### Sampling in Practice
```python
import pandas as pd

df = pd.DataFrame({'customer_id': range(1, 100001),
                   'spend': np.random.gamma(2, 500, 100000)})

# Simple random sample
sample = df.sample(n=1000, random_state=42)

# Stratified sample (preserving group proportions)
df['spend_tier'] = pd.qcut(df['spend'], 3, labels=['Low','Medium','High'])
stratified = df.groupby('spend_tier', group_keys=False) \
               .apply(lambda x: x.sample(frac=0.01, random_state=42))

print("Population mean:", df['spend'].mean())
print("Sample mean    :", sample['spend'].mean())      # close, but not identical
```

> 💡 **Rule of thumb for Data Science:** Pandas' `.std()`, `.var()` default to **sample** statistics (`ddof=1`). NumPy's `np.std()`, `np.var()` default to **population** statistics (`ddof=0`). This mismatch is one of the most common sources of "why are my numbers different" confusion between the two libraries.

```python
data = [23, 45, 12, 67, 34, 89, 21, 55]

print(pd.Series(data).std())      # sample std (ddof=1) — Pandas default
print(np.std(data))               # population std (ddof=0) — NumPy default
print(np.std(data, ddof=1))       # match Pandas by setting ddof=1
```

---

## 3. Mean

The **arithmetic average** — sum of all values divided by the count.

**Formula:**
```
        Σxᵢ
mean = ─────
          n
```

### 3.1 Calculating the Mean
```python
data = [23, 45, 12, 67, 34, 89, 21, 55]

# Manual
manual_mean = sum(data) / len(data)

# NumPy
np_mean = np.mean(data)

# Pandas
s = pd.Series(data)
pd_mean = s.mean()

print(manual_mean, np_mean, pd_mean)     # all give 43.25

# On a DataFrame
df = pd.DataFrame({'math': [85,90,78,92], 'science': [88,76,95,80]})
print(df.mean())                # mean of each numeric column
print(df.mean(axis=1))          # mean across each row
print(df['math'].mean())        # single column
```

### 3.2 Types of Mean
```python
# Arithmetic mean (most common - shown above)

# Weighted mean - when observations don't count equally
scores  = np.array([80, 90, 70])
weights = np.array([0.5, 0.3, 0.2])       # e.g. exam weight percentages
weighted_mean = np.average(scores, weights=weights)
print(weighted_mean)              # 81.0

# Geometric mean - for rates of change, growth rates (e.g. investment returns)
from scipy.stats import gmean
returns = [1.05, 1.10, 0.95, 1.08]        # growth multipliers
print(gmean(returns))

# Harmonic mean - for rates/ratios (e.g. average speed over equal distances)
from scipy.stats import hmean
speeds = [60, 40, 80]      # km/h over equal-distance legs
print(hmean(speeds))
```

### 3.3 Strength & Weakness
```python
normal    = [10, 20, 30, 40, 50]
with_out  = [10, 20, 30, 40, 1000]     # one extreme outlier

print(np.mean(normal))       # 30.0
print(np.mean(with_out))     # 220.0   ← heavily distorted by the outlier
```
✅ Uses every data point (no information lost)
✅ Mathematically convenient (basis for variance, regression, etc.)
❌ **Highly sensitive to outliers** and skewed distributions

---

## 4. Median

The **middle value** when data is sorted. Splits the dataset exactly in half.

### 4.1 Calculating the Median
```python
# Odd count → the exact middle value
odd_data = [12, 45, 7, 23, 56]
print(np.median(odd_data))          # 23.0 (sorted: 7,12,23,45,56 → middle)

# Even count → average of the two middle values
even_data = [12, 45, 7, 23]
print(np.median(even_data))         # 17.5 (sorted: 7,12,23,45 → (12+23)/2)

# Pandas
s = pd.Series([12, 45, 7, 23, 56])
print(s.median())

# DataFrame
df = pd.DataFrame({'math':[85,90,78,92], 'science':[88,76,95,80]})
print(df.median())
print(df.median(axis=1))
```

### 4.2 Manual Calculation (Understanding the Formula)
```python
def manual_median(data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0:                                    # even count
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:                                              # odd count
        return sorted_data[mid]

print(manual_median([12, 45, 7, 23, 56]))     # 23
print(manual_median([12, 45, 7, 23]))         # 17.5
```

### 4.3 Strength — Robust to Outliers
```python
normal    = [10, 20, 30, 40, 50]
with_out  = [10, 20, 30, 40, 1000]

print(np.median(normal))       # 30.0
print(np.median(with_out))     # 30.0  ← UNCHANGED, unaffected by the outlier
```
✅ **Robust to outliers and skewed data** — best default for income, house prices, response times
❌ Ignores the actual magnitude of extreme values (doesn't use all data equally)
❌ Less mathematically convenient for further calculations (e.g., can't easily be used in regression like the mean)

### 4.4 Median vs Mean — Quick Decision Guide
| Data characteristic | Use |
|---|---|
| Symmetric, no outliers (e.g., heights, test scores) | **Mean** |
| Skewed or has outliers (e.g., income, house prices, hospital wait times) | **Median** |
| Comparing mean vs median tells you about skew | mean > median → right-skewed; mean < median → left-skewed |

---

## 5. Mode

The value that occurs **most frequently** in the dataset. The only central tendency measure that works for **categorical (non-numeric) data**.

### 5.1 Calculating the Mode
```python
data = [1, 2, 2, 3, 3, 3, 4, 5]

# SciPy
from scipy import stats
result = stats.mode(data, keepdims=True)
print(result.mode[0], result.count[0])       # 3   3  (value, frequency)

# Pandas (handles multiple modes automatically)
s = pd.Series(data)
print(s.mode())                # returns ALL modes as a Series, e.g. [3]

# Python's statistics module
import statistics
print(statistics.mode(data))              # 3
# print(statistics.multimode(data))       # [3]  - all modes, no error if tied
```

### 5.2 Mode for Categorical Data (Its Primary Use Case)
```python
cities = pd.Series(['Mumbai','Delhi','Mumbai','Pune','Mumbai','Delhi'])
print(cities.mode())                        # Mumbai (most frequent)
print(cities.value_counts())                # full frequency breakdown

# On a DataFrame column
df = pd.DataFrame({'city': cities})
print(df['city'].mode()[0])
```

### 5.3 Types of Modal Distributions
```python
unimodal   = [1, 2, 2, 2, 3, 4]              # one mode: 2
bimodal    = [1, 2, 2, 5, 8, 8, 9]           # two modes: 2 and 8
no_mode    = [1, 2, 3, 4, 5]                 # all unique, no mode (or every value is a mode)

print(pd.Series(bimodal).mode())             # returns [2, 8]
```

```
Unimodal          Bimodal              Multimodal
   ▁▃█▃▁          ▁▃█▁▁█▃▁              ▁▃█▁▃█▁▃█▁
    ↑               ↑  ↑                 ↑ ↑ ↑
  1 peak         2 peaks               3+ peaks
```

### 5.4 Common Use Case — Most Common Category (Mode Imputation)
```python
df = pd.DataFrame({'city': ['Mumbai', 'Delhi', np.nan, 'Mumbai', np.nan, 'Pune']})

# Filling missing categorical values with the mode - standard DS practice
mode_value = df['city'].mode()[0]
df['city'] = df['city'].fillna(mode_value)
print(df)
```

### Mean vs Median vs Mode — Full Comparison
| | Mean | Median | Mode |
|---|---|---|---|
| Works on numeric data | ✅ | ✅ | ✅ |
| Works on categorical data | ❌ | ❌ (ordinal only) | ✅ |
| Affected by outliers | ✅ Highly | ❌ No | ❌ No |
| Uses all data points | ✅ | ❌ (only position) | ❌ |
| Can have multiple values | ❌ (always one) | ❌ (always one) | ✅ (can be multi-modal) |

---

## 6. Range

The simplest measure of spread — the difference between the **maximum** and **minimum** values.

**Formula:**
```
Range = Max − Min
```

### 6.1 Calculating the Range
```python
data = [23, 45, 12, 67, 34, 89, 21, 55]

# Manual
manual_range = max(data) - min(data)
print(manual_range)              # 77

# NumPy
print(np.ptp(data))              # "peak to peak" = 77
print(np.max(data) - np.min(data))

# Pandas
s = pd.Series(data)
print(s.max() - s.min())
```

### 6.2 Strength & Weakness
```python
tight = [48, 49, 50, 51, 52]           # range = 4
loose = [10, 30, 50, 70, 90]           # range = 80 (same mean, very different spread)
outlier_case = [45, 47, 48, 50, 500]   # range = 455 ← one outlier dominates

print(max(tight)-min(tight), max(loose)-min(loose), max(outlier_case)-min(outlier_case))
```
✅ Extremely simple and fast to compute
✅ Useful for a quick first check of data spread
❌ **Uses only 2 values** (max, min) — ignores everything in between
❌ **Extremely sensitive to outliers** — a single extreme value distorts it completely

---

## 7. Variance

Measures the **average squared deviation** from the mean — quantifies how spread out the data is, in **squared units**.

**Population Variance Formula:**
```
        Σ(xᵢ − μ)²
σ²  =   ───────────
             N
```

**Sample Variance Formula (uses Bessel's correction):**
```
        Σ(xᵢ − x̄)²
s²  =   ───────────
            n − 1
```

### 7.1 Calculating Variance
```python
data = [23, 45, 12, 67, 34, 89, 21, 55]

# NumPy - default is POPULATION variance (ddof=0)
pop_var = np.var(data)
print(pop_var)                    # 555.9375

# Sample variance (ddof=1)
sample_var = np.var(data, ddof=1)
print(sample_var)                 # 635.357...

# Pandas - default is SAMPLE variance (ddof=1)
s = pd.Series(data)
print(s.var())                    # matches np.var(data, ddof=1)
print(s.var(ddof=0))              # matches population variance
```

### 7.2 Manual Calculation (Step by Step)
```python
def manual_variance(data, sample=True):
    n = len(data)
    mean = sum(data) / n
    squared_diffs = [(x - mean) ** 2 for x in data]
    divisor = (n - 1) if sample else n
    return sum(squared_diffs) / divisor

data = [23, 45, 12, 67, 34, 89, 21, 55]
print(manual_variance(data, sample=True))    # matches np.var(data, ddof=1)
print(manual_variance(data, sample=False))   # matches np.var(data)
```

### 7.3 Why Squaring? (Understanding the Formula)
```python
data = [10, 20, 30]
mean = np.mean(data)          # 20

deviations = [x - mean for x in data]
print(deviations)             # [-10, 0, 10]
print(sum(deviations))        # 0  ← deviations always sum to zero!

# Squaring makes them positive so they don't cancel out, and it also
# penalizes larger deviations more heavily than smaller ones
squared = [(x - mean)**2 for x in data]
print(squared)                # [100, 0, 100]
print(sum(squared) / len(data))   # variance = 66.67
```

### 7.4 Interpreting Variance
```python
tight = [48, 49, 50, 51, 52]
loose = [10, 30, 50, 70, 90]

print(f"Tight variance: {np.var(tight):.2f}")     # 2.0    (small spread)
print(f"Loose variance: {np.var(loose):.2f}")     # 800.0  (large spread)
```
✅ Uses **every** data point (unlike range)
✅ Foundation for many statistical/ML methods (e.g. PCA, ANOVA, regression)
❌ **Units are squared** — hard to interpret directly (e.g., "625 squared rupees" is meaningless intuitively) → this is exactly why we take the square root to get **standard deviation**

---

## 8. Standard Deviation

The **square root of variance** — brings the measure of spread back into the **original units** of the data, making it directly interpretable.

**Formula:**
```
σ = √σ²      (population)          s = √s²      (sample)
```

### 8.1 Calculating Standard Deviation
```python
data = [23, 45, 12, 67, 34, 89, 21, 55]

# NumPy
pop_std = np.std(data)                # default: ddof=0 (population)
sample_std = np.std(data, ddof=1)     # sample

print(pop_std, sample_std)            # 23.58   25.21

# Pandas - default is SAMPLE std (ddof=1)
s = pd.Series(data)
print(s.std())                        # matches np.std(data, ddof=1)

# Verify: std = sqrt(variance)
print(np.sqrt(np.var(data, ddof=1)) == s.std())    # True
```

### 8.2 On DataFrames
```python
df = pd.DataFrame({'math':[85,90,78,92,88], 'science':[88,76,95,80,84]})

print(df.std())                # std per column (sample, ddof=1 default)
print(df.std(axis=1))          # std across each row
print(df.describe())           # includes std automatically
```

### 8.3 Why Standard Deviation Is Preferred Over Variance
```python
salaries = [45000, 48000, 52000, 47000, 95000]

variance = np.var(salaries)
std_dev  = np.std(salaries)

print(f"Variance: {variance:,.0f}")        # 313,760,000  ← "squared rupees", meaningless
print(f"Std Dev : {std_dev:,.0f}")         # 17,713       ← "rupees", directly interpretable
```

### 8.4 The Empirical Rule (68-95-99.7 Rule)
For **normally distributed** data:
```
        68% of data
      ┌───────────────┐
      │       95%       │
    ┌─┴─────────────────┴─┐
    │         99.7%         │
┌───┴───────────────────────┴───┐
-3σ  -2σ  -1σ   μ   +1σ  +2σ  +3σ
```

```python
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=10000)

mean, std = data.mean(), data.std()

within_1sd = np.mean(np.abs(data - mean) <= 1*std)
within_2sd = np.mean(np.abs(data - mean) <= 2*std)
within_3sd = np.mean(np.abs(data - mean) <= 3*std)

print(f"Within ±1σ: {within_1sd:.1%}")     # ~68%
print(f"Within ±2σ: {within_2sd:.1%}")     # ~95%
print(f"Within ±3σ: {within_3sd:.1%}")     # ~99.7%
```

### 8.5 Z-Score — Standardizing Using Std Dev
The **z-score** tells you how many standard deviations a value is from the mean — the basis of outlier detection and feature scaling.

**Formula:**
```
      x − μ
z  =  ──────
        σ
```

```python
data = np.array([23, 45, 12, 67, 34, 89, 21, 55, 200])   # 200 is an outlier

mean, std = data.mean(), data.std()
z_scores = (data - mean) / std
print(z_scores.round(2))

# Standard outlier rule: |z| > 3 is considered an outlier
outliers = data[np.abs(z_scores) > 3]
print("Outliers:", outliers)

# SciPy shortcut
from scipy import stats
z_scipy = stats.zscore(data)
print(z_scipy.round(2))
```

### 8.6 Coefficient of Variation (CV) — Comparing Spread Across Different Scales
```python
# CV = (std / mean) * 100 — lets you compare variability between datasets
# with very different units or magnitudes
heights_cm = np.array([160, 165, 170, 175, 180])
weights_kg = np.array([55, 60, 70, 75, 90])

cv_height = (heights_cm.std() / heights_cm.mean()) * 100
cv_weight = (weights_kg.std() / weights_kg.mean()) * 100

print(f"CV Height: {cv_height:.1f}%")
print(f"CV Weight: {cv_weight:.1f}%")
# Higher CV = more relative variability, even though units differ
```

---

## 9. Percentiles

A percentile indicates the value **below which a given percentage of observations fall**. The Pth percentile means P% of the data lies at or below that value.

### 9.1 Calculating Percentiles
```python
data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 100]

print(np.percentile(data, 25))      # 25th percentile → 20.5
print(np.percentile(data, 50))      # 50th percentile = median → 26.5
print(np.percentile(data, 90))      # 90th percentile → higher value
print(np.percentile(data, [10, 50, 90]))   # multiple at once

# Pandas
s = pd.Series(data)
print(s.quantile(0.25))             # same as 25th percentile, but 0-1 scale
print(s.quantile([0.1, 0.5, 0.9]))
```

### 9.2 Percentile vs Quantile — Same Concept, Different Scale
```python
# percentile uses 0-100 scale, quantile uses 0-1 scale - otherwise identical
print(np.percentile(data, 75))      # 75
print(np.quantile(data, 0.75))      # same result as above
```

### 9.3 Interpolation Methods
When the percentile falls between two data points, NumPy/Pandas interpolate:
```python
data = [10, 20, 30, 40]

print(np.percentile(data, 50, interpolation='linear'))   # default: interpolates
print(np.percentile(data, 50, interpolation='lower'))    # takes the lower value
print(np.percentile(data, 50, interpolation='higher'))   # takes the higher value
print(np.percentile(data, 50, interpolation='nearest'))  # nearest data point
```
> Note: In recent NumPy versions, the parameter is named `method` instead of `interpolation`.

### 9.4 Real-World Use Cases
```python
# 1. Understanding relative standing (e.g. "what percentile is my score in?")
scores = np.array([45, 60, 72, 55, 80, 90, 65, 70, 85, 95])
my_score = 72
percentile_rank = (scores < my_score).mean() * 100
print(f"Your score is in the {percentile_rank:.0f}th percentile")

# 2. Setting SLA thresholds (e.g. "95% of API responses under X ms")
response_times = np.random.exponential(200, 1000)     # simulated latency data
p95 = np.percentile(response_times, 95)
p99 = np.percentile(response_times, 99)
print(f"P95 latency: {p95:.0f}ms, P99 latency: {p99:.0f}ms")

# 3. Capping extreme values (Winsorization) using percentiles
data = np.array([10, 12, 13, 15, 18, 20, 500])
lower_cap = np.percentile(data, 5)
upper_cap = np.percentile(data, 95)
capped = np.clip(data, lower_cap, upper_cap)
print(capped)
```

### 9.5 Percentile-Based Binning
```python
df = pd.DataFrame({'income': np.random.gamma(2, 30000, 1000)})

# Split into equal-sized groups by percentile (not equal-width!)
df['income_tier'] = pd.qcut(df['income'], q=4,
                            labels=['Q1(Low)','Q2','Q3','Q4(High)'])
print(df['income_tier'].value_counts())
```

---

## 10. Quartiles

Quartiles are **specific percentiles** that divide the data into **4 equal parts**. They're the most commonly used percentiles in EDA.

### 10.1 The Four Quartiles
| Quartile | Percentile | Meaning |
|---|---|---|
| **Q0** (Minimum) | 0th | Smallest value |
| **Q1** (Lower Quartile) | 25th | 25% of data falls below this |
| **Q2** (Median) | 50th | 50% of data falls below this — the median |
| **Q3** (Upper Quartile) | 75th | 75% of data falls below this |
| **Q4** (Maximum) | 100th | Largest value |

```
Min ──── Q1 ──── Q2(Median) ──── Q3 ──── Max
 │        │           │           │       │
 0%      25%         50%         75%    100%

|←── 25% ──|←── 25% ──|←── 25% ──|←── 25% ──|
        (each segment holds ~25% of the data)
```

### 10.2 Calculating Quartiles
```python
data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 100]

Q1 = np.percentile(data, 25)
Q2 = np.percentile(data, 50)    # = median
Q3 = np.percentile(data, 75)

print(f"Q1: {Q1}, Q2 (median): {Q2}, Q3: {Q3}")

# Pandas — quantile() and describe() both give quartiles
s = pd.Series(data)
print(s.quantile([0.25, 0.5, 0.75]))
print(s.describe())              # includes 25%, 50%, 75% automatically
```

### 10.3 Quartiles on a DataFrame
```python
df = pd.DataFrame({'salary': [30000, 45000, 50000, 60000, 75000, 90000, 120000]})

print(df['salary'].describe())
print(df.quantile([0.25, 0.5, 0.75]))
```

### 10.4 Using Quartiles to Create Categories (Binning)
```python
df = pd.DataFrame({'score': np.random.randint(0, 100, 20)})

df['quartile'] = pd.qcut(df['score'], q=4, labels=['Q1','Q2','Q3','Q4'])
print(df.sort_values('score'))
```

---

## 11. IQR

The **Interquartile Range** measures the **spread of the middle 50%** of the data — the most important measure for detecting outliers.

**Formula:**
```
IQR = Q3 − Q1
```

### 11.1 Calculating IQR
```python
data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 100]

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")

# SciPy shortcut
from scipy.stats import iqr
print(iqr(data))

# Pandas
s = pd.Series(data)
q1, q3 = s.quantile(0.25), s.quantile(0.75)
print(q3 - q1)
```

### 11.2 Outlier Detection Using IQR (The 1.5×IQR Rule)
The standard rule: any value **below `Q1 - 1.5×IQR`** or **above `Q3 + 1.5×IQR`** is considered an outlier.

```python
data = np.array([12, 15, 18, 22, 25, 28, 30, 35, 40, 100])

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

outliers = data[(data < lower_bound) | (data > upper_bound)]
print("Outliers:", outliers)          # [100]

clean_data = data[(data >= lower_bound) & (data <= upper_bound)]
print("Clean data:", clean_data)
```

### 11.3 Applying IQR Outlier Removal to a DataFrame
```python
df = pd.DataFrame({'salary': [30000, 45000, 50000, 48000, 52000, 250000, 47000]})

Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Method 1: Remove outliers
df_clean = df[(df['salary'] >= lower) & (df['salary'] <= upper)]

# Method 2: Cap outliers (Winsorization) instead of removing
df['salary_capped'] = df['salary'].clip(lower, upper)

print(df)
print(df_clean)
```

### 11.4 IQR and the Box Plot Connection
```python
import matplotlib.pyplot as plt

data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 100]
plt.boxplot(data)
plt.title("Box Plot — Q1, Median, Q3, and IQR-based outliers")
plt.show()
# The box spans Q1 to Q3 (height = IQR), whiskers extend to
# the last point within 1.5×IQR, and points beyond are outliers
```

### 11.5 Why 1.5×IQR? (Understanding the Convention)
```python
# For a NORMAL distribution, the 1.5x IQR rule flags roughly the outer 0.7%
# of data as outliers - it's a widely-used statistical convention (John Tukey, 1977)
np.random.seed(42)
normal_data = np.random.normal(50, 10, 10000)

Q1, Q3 = np.percentile(normal_data, [25, 75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR

outlier_pct = np.mean((normal_data < lower) | (normal_data > upper)) * 100
print(f"Outlier percentage on normal data: {outlier_pct:.2f}%")   # ~0.7%

# For SKEWED data, be cautious - the 1.5xIQR rule may flag too many "outliers"
# that are actually just legitimate extreme (but valid) values
skewed_data = np.random.exponential(scale=20, size=10000)
Q1s, Q3s = np.percentile(skewed_data, [25, 75])
IQRs = Q3s - Q1s
lower_s, upper_s = Q1s - 1.5*IQRs, Q3s + 1.5*IQRs
outlier_pct_skewed = np.mean((skewed_data < lower_s) | (skewed_data > upper_s)) * 100
print(f"Outlier percentage on skewed data: {outlier_pct_skewed:.2f}%")  # much higher!
```

### IQR vs Standard Deviation for Outlier Detection
| | IQR Method (1.5×IQR) | Z-Score Method (\|z\|>3) |
|---|---|---|
| Based on | Quartiles (position) | Mean & standard deviation |
| Robust to outliers itself | ✅ Yes (uses median-based quartiles) | ❌ No (mean/std are themselves outlier-sensitive) |
| Best for | Skewed data | Normally distributed data |
| Common threshold | 1.5× (or 3× for "extreme" outliers) | \|z\| > 3 |

---

## 12. Bringing It All Together

### 12.1 The Complete Descriptive Statistics Toolkit
```python
import numpy as np
import pandas as pd
from scipy import stats

def full_summary(data, name="Dataset"):
    """A complete descriptive statistics report."""
    arr = np.array(data)

    print(f"\n{'='*40}\n{name}\n{'='*40}")

    # Central Tendency
    print("\n--- Central Tendency ---")
    print(f"Mean   : {np.mean(arr):.2f}")
    print(f"Median : {np.median(arr):.2f}")
    mode_result = stats.mode(arr, keepdims=True)
    print(f"Mode   : {mode_result.mode[0]} (count: {mode_result.count[0]})")

    # Dispersion
    print("\n--- Dispersion ---")
    print(f"Range     : {np.ptp(arr):.2f}")
    print(f"Variance  : {np.var(arr, ddof=1):.2f}")
    print(f"Std Dev   : {np.std(arr, ddof=1):.2f}")

    # Position
    Q1, Q2, Q3 = np.percentile(arr, [25, 50, 75])
    IQR = Q3 - Q1
    print("\n--- Position / Quartiles ---")
    print(f"Q1 (25%)  : {Q1:.2f}")
    print(f"Q2 (50%)  : {Q2:.2f}")
    print(f"Q3 (75%)  : {Q3:.2f}")
    print(f"IQR       : {IQR:.2f}")

    # Outliers
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = arr[(arr < lower) | (arr > upper)]
    print(f"\nOutlier bounds: [{lower:.2f}, {upper:.2f}]")
    print(f"Outliers found: {list(outliers)}")

    # Shape
    skew_direction = ("Right-skewed" if np.mean(arr) > np.median(arr)
                      else "Left-skewed" if np.mean(arr) < np.median(arr)
                      else "Symmetric")
    print(f"\nDistribution shape: {skew_direction}")
    print(f"Skewness (scipy)  : {stats.skew(arr):.2f}")
    print(f"Kurtosis (scipy)  : {stats.kurtosis(arr):.2f}")

data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 100]
full_summary(data, "Sample Data")
```

### 12.2 Pandas' Built-in describe() Does Most of This Automatically
```python
df = pd.DataFrame({'values': [12, 15, 18, 22, 25, 28, 30, 35, 40, 100]})
print(df.describe())
# count, mean, std, min, 25%, 50%, 75%, max — all in one call!

# Add custom percentiles
print(df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
```

---

## 13. Practical Mini Project

```python
# ============================================================
# STATISTICS MINI PROJECT: Employee Salary Analysis
# ============================================================
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# --- 1. Create a realistic (right-skewed) salary dataset ---
salaries = np.random.gamma(shape=3, scale=15000, size=500)
salaries = np.append(salaries, [250000, 280000, 300000])   # add a few "executive" outliers

df = pd.DataFrame({'employee_id': range(1, len(salaries)+1), 'salary': salaries.round(0)})

# --- 2. Population vs Sample framing ---
population_mean = df['salary'].mean()          # treating full company as "population" here
sample = df.sample(n=100, random_state=1)
sample_mean = sample['salary'].mean()
print(f"Population mean: ₹{population_mean:,.0f}")
print(f"Sample mean (n=100): ₹{sample_mean:,.0f}")

# --- 3. Central tendency ---
mean_sal   = df['salary'].mean()
median_sal = df['salary'].median()
mode_sal   = stats.mode(df['salary'].round(-3), keepdims=True).mode[0]   # rounded for a meaningful mode

print(f"\nMean  : ₹{mean_sal:,.0f}")
print(f"Median: ₹{median_sal:,.0f}")
print(f"Mode  : ₹{mode_sal:,.0f} (rounded to nearest 1000)")

if mean_sal > median_sal:
    print("→ Mean > Median: distribution is RIGHT-SKEWED (a few very high earners)")

# --- 4. Dispersion ---
data_range = df['salary'].max() - df['salary'].min()
variance   = df['salary'].var()          # sample variance, ddof=1 default in pandas
std_dev    = df['salary'].std()

print(f"\nRange   : ₹{data_range:,.0f}")
print(f"Variance: {variance:,.0f}")
print(f"Std Dev : ₹{std_dev:,.0f}")

# --- 5. Percentiles & Quartiles ---
percentiles = df['salary'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print("\n--- Percentiles ---")
print(percentiles)

Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
print(f"\nQ1: ₹{Q1:,.0f}  Q3: ₹{Q3:,.0f}  IQR: ₹{IQR:,.0f}")

# --- 6. Outlier detection with IQR ---
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['salary'] < lower_bound) | (df['salary'] > upper_bound)]

print(f"\nOutlier bounds: ₹{lower_bound:,.0f} to ₹{upper_bound:,.0f}")
print(f"Number of outliers: {len(outliers)}")
print(outliers[['employee_id','salary']])

# --- 7. Visualizing everything together ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['salary'], bins=30, color='steelblue', edgecolor='black')
axes[0].axvline(mean_sal, color='red', linestyle='--', label=f'Mean: ₹{mean_sal:,.0f}')
axes[0].axvline(median_sal, color='green', linestyle='--', label=f'Median: ₹{median_sal:,.0f}')
axes[0].set_title('Salary Distribution')
axes[0].set_xlabel('Salary')
axes[0].legend()

axes[1].boxplot(df['salary'], vert=True)
axes[1].set_title('Salary Box Plot (IQR & Outliers)')
axes[1].set_ylabel('Salary')

plt.tight_layout()
plt.savefig('salary_analysis.png', dpi=150)
plt.show()

# --- 8. Final Report ---
print("\n" + "="*50)
print("SUMMARY REPORT")
print("="*50)
print(f"Total employees analyzed : {len(df)}")
print(f"Typical salary (median)  : ₹{median_sal:,.0f}")
print(f"Salary spread (IQR)      : ₹{IQR:,.0f}")
print(f"High earners flagged     : {len(outliers)} ({len(outliers)/len(df):.1%})")
print("Recommendation: Use MEDIAN, not mean, to report 'typical' salary")
print("due to right-skew caused by high-earner outliers.")
```

---

## 14. Cheatsheet

### Population vs Sample
```python
np.var(data, ddof=0)     # population variance (NumPy default)
np.var(data, ddof=1)     # sample variance
pd.Series(data).var()    # sample variance (Pandas default, ddof=1)
```

### Central Tendency
```python
np.mean(data)                              # mean
np.median(data)                            # median
stats.mode(data, keepdims=True).mode[0]    # mode
pd.Series(data).mode()                     # mode (handles multi-modal)
```

### Dispersion
```python
np.ptp(data)                # range
np.var(data, ddof=1)        # sample variance
np.std(data, ddof=1)        # sample standard deviation
```

### Position
```python
np.percentile(data, 25)             # 25th percentile
np.percentile(data, [25,50,75])     # Q1, Q2, Q3 at once
pd.Series(data).quantile(0.25)      # same, 0-1 scale
```

### IQR & Outliers
```python
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers = data[(data < lower) | (data > upper)]
```

### All-in-One
```python
pd.Series(data).describe()       # count, mean, std, min, 25%, 50%, 75%, max
```

### Key Rules to Remember 🏆
1. **Population** = entire group (μ, σ, N); **Sample** = subset (x̄, s, n) — most DS work uses samples.
2. Sample variance/std uses **n−1** (Bessel's correction) to avoid underestimating the true population value.
3. NumPy defaults to **population** stats (`ddof=0`); Pandas defaults to **sample** stats (`ddof=1`) — always check which one you're getting.
4. **Mean** is sensitive to outliers; **Median** and **Mode** are robust.
5. **Mean > Median** → right-skewed; **Mean < Median** → left-skewed.
6. **Variance** is in squared units (hard to interpret); **Standard deviation** is in original units.
7. **Range** uses only 2 points and is very outlier-sensitive; **IQR** uses the middle 50% and is robust.
8. The **1.5×IQR rule** is the standard convention for outlier detection — works best on roughly symmetric data.
9. `Q1`, `Q2` (median), `Q3` are the 25th, 50th, and 75th **percentiles** — quartiles are a special case of percentiles.

---

## 📌 Summary

| Concept | Formula / Key Idea | Robust to Outliers? |
|---|---|---|
| **Population vs Sample** | μ,σ,N (population) vs x̄,s,n (sample) | — |
| **Mean** | Σx / n | ❌ No |
| **Median** | Middle value when sorted | ✅ Yes |
| **Mode** | Most frequent value | ✅ Yes |
| **Range** | Max − Min | ❌ No |
| **Variance** | Σ(x−mean)² / (n or n−1) | ❌ No |
| **Standard Deviation** | √Variance | ❌ No |
| **Percentile** | Value below which P% of data falls | ✅ Yes (position-based) |
| **Quartiles** | 25th, 50th, 75th percentiles | ✅ Yes |
| **IQR** | Q3 − Q1 (middle 50% spread) | ✅ Yes |

---

*📁 Previous: Part 7 → Seaborn*
*📁 Next: Part 9 → Inferential Statistics (Probability, Hypothesis Testing, Confidence Intervals)*

---
### 🔖 Tags
`#Statistics` `#DataScience` `#Python` `#NumPy` `#Pandas` `#EDA` `#Notes`
