# 🐼 Data Science Notes — Part 5: Pandas

> The most-used Python library for data manipulation and analysis, built on top of NumPy. Theory + practical syntax and code examples.

---

## Table of Contents
1. [Introduction to Pandas](#1-introduction-to-pandas)
2. [Series](#2-series)
3. [DataFrame](#3-dataframe)
4. [Reading CSV](#4-reading-csv)
5. [head()](#5-head)
6. [tail()](#6-tail)
7. [info()](#7-info)
8. [describe()](#8-describe)
9. [Selecting Columns](#9-selecting-columns)
10. [Filtering Rows](#10-filtering-rows)
11. [Sorting](#11-sorting)
12. [Adding/Removing Columns](#12-addingremoving-columns)
13. [Handling Missing Values](#13-handling-missing-values)
14. [groupby()](#14-groupby)
15. [Merging/Joining](#15-mergingjoining)
16. [Practical Mini Project](#16-practical-mini-project)
17. [Cheatsheet](#17-cheatsheet)

---

## 1. Introduction to Pandas

**Pandas** = "**Pan**el **Da**ta". It's the primary library for loading, cleaning, transforming, and analyzing tabular data in Python — the tool you'll use in almost every DS project.

### Why Pandas?
| Feature | Benefit |
|---|---|
| **DataFrame** | Excel-like table structure, but programmable |
| **Built on NumPy** | Fast, vectorized operations |
| **File I/O** | Reads/writes CSV, Excel, SQL, JSON, Parquet |
| **Missing data handling** | Built-in `NaN` support and tools |
| **Powerful grouping** | `groupby()` for aggregation, like SQL |
| **Time series support** | Native datetime indexing |

### Installation & Import
```bash
pip install pandas
```
```python
import pandas as pd        # 'pd' is the universal convention
import numpy as np
print(pd.__version__)
```

### Two Core Data Structures
```
Series      → 1-D labeled array (like a single column)
DataFrame   → 2-D labeled table (like a spreadsheet / SQL table)
```

---

## 2. Series

A **Series** is a **one-dimensional labeled array** that can hold any data type (int, string, float, objects). It's essentially a NumPy array with an **index** attached.

### 2.1 Creating a Series
```python
import pandas as pd

# From a list (default integer index: 0,1,2...)
s = pd.Series([10, 20, 30, 40])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64

# With a custom index
s2 = pd.Series([85, 90, 78], index=['Ravi', 'Meera', 'Sam'])
print(s2)
# Ravi     85
# Meera    90
# Sam      78

# From a dictionary (keys become the index)
s3 = pd.Series({'a': 1, 'b': 2, 'c': 3})

# From a scalar (broadcast to given index)
s4 = pd.Series(5, index=['x', 'y', 'z'])    # 5 5 5

# With a name
s5 = pd.Series([1,2,3], name="marks")
```

### 2.2 Accessing Series Data
```python
s = pd.Series([85, 90, 78, 92], index=['Ravi','Meera','Sam','Priya'])

print(s['Ravi'])          # 85          label-based
print(s.iloc[0])          # 85          position-based
print(s.loc['Meera'])     # 90          label-based (explicit)

print(s[['Ravi','Sam']])  # multiple labels
print(s.iloc[0:2])        # positional slicing

print(s.values)           # array([85, 90, 78, 92])
print(s.index)            # Index(['Ravi','Meera','Sam','Priya'])
print(s.name)              # None unless set
```

### 2.3 Series Operations (Vectorized, like NumPy)
```python
s = pd.Series([10, 20, 30, 40])

print(s + 5)               # element-wise
print(s * 2)
print(s[s > 15])           # boolean filtering
print(s.mean(), s.sum(), s.max(), s.min(), s.std())

# Apply a custom function
print(s.apply(lambda x: x**2))

# Check membership
print(20 in s.values)      # True

# Sorting
print(s.sort_values(ascending=False))
print(s.sort_index())
```

### 2.4 Series vs NumPy Array vs Python List
| Feature | List | NumPy Array | Pandas Series |
|---|---|---|---|
| Labeled index | ❌ | ❌ (positional only) | ✅ Custom labels |
| Mixed types | ✅ | ❌ (homogeneous) | ✅ (via `object` dtype) |
| Vectorized ops | ❌ | ✅ | ✅ |
| Missing value support | ❌ | Limited | ✅ Built-in `NaN` handling |

---

## 3. DataFrame

A **DataFrame** is a **two-dimensional labeled table** — rows and columns, like an Excel sheet or SQL table. It's a collection of Series sharing the same index.

### 3.1 Creating a DataFrame
```python
import pandas as pd

# From a dictionary of lists (most common)
data = {
    'Name':   ['Ravi', 'Meera', 'Sam', 'Priya'],
    'Age':    [23, 25, 22, 24],
    'Marks':  [85, 92, 78, 88]
}
df = pd.DataFrame(data)
print(df)
#     Name  Age  Marks
# 0   Ravi   23     85
# 1  Meera   25     92
# 2    Sam   22     78
# 3  Priya   24     88

# From a list of dictionaries (row-wise)
records = [
    {'Name': 'Ravi', 'Age': 23},
    {'Name': 'Meera', 'Age': 25}
]
df2 = pd.DataFrame(records)

# From a list of lists (need column names)
df3 = pd.DataFrame([[1,'Ravi',85], [2,'Meera',92]],
                    columns=['ID','Name','Marks'])

# From a NumPy array
import numpy as np
df4 = pd.DataFrame(np.random.randint(0,100,(3,3)),
                    columns=['A','B','C'])

# Custom index
df5 = pd.DataFrame(data, index=['s1','s2','s3','s4'])

# Empty DataFrame
df6 = pd.DataFrame(columns=['Name','Age'])
```

### 3.2 DataFrame Attributes
```python
print(df.shape)         # (4, 3)   rows, columns
print(df.columns)       # Index(['Name','Age','Marks'])
print(df.index)         # RangeIndex(start=0, stop=4, step=1)
print(df.dtypes)        # data type of each column
print(df.values)        # underlying NumPy array
print(df.size)          # total number of cells (12)
print(df.ndim)          # 2
print(len(df))          # number of rows (4)
print(df.axes)          # [row index, column index]
```

### 3.3 DataFrame from CSV/dict/Series (quick reference)
```python
df = pd.DataFrame(data)                    # dict
df = pd.DataFrame({'col': pd.Series([1,2,3])})   # from Series
df = pd.read_csv("file.csv")               # from file (see Section 4)
```

---

## 4. Reading CSV

`pd.read_csv()` is the most-used Pandas function — loads tabular data from a `.csv` file into a DataFrame.

### 4.1 Basic Usage
```python
df = pd.read_csv("data.csv")
print(df.head())
```

### 4.2 Common Parameters
```python
df = pd.read_csv(
    "data.csv",
    sep=',',                    # delimiter (';', '\t' for tab-separated etc.)
    header=0,                   # row to use as column names (0 = first row)
    names=['a','b','c'],        # custom column names (use with header=None)
    index_col=0,                # use column 0 as the row index
    usecols=['Name','Age'],     # load only specific columns
    dtype={'Age': int},         # force specific dtypes
    parse_dates=['join_date'],  # parse column(s) as datetime
    na_values=['NA', '--', ''], # treat these strings as NaN
    skiprows=1,                 # skip first N rows
    nrows=1000,                 # read only first N rows (sampling)
    encoding='utf-8',           # handle special characters
    on_bad_lines='skip'         # skip malformed rows instead of erroring
)
```

### 4.3 Reading Other Formats
```python
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_json("data.json")
df = pd.read_sql("SELECT * FROM customers", conn)     # needs a DB connection
df = pd.read_parquet("data.parquet")
df = pd.read_html("https://example.com/table.html")[0]   # scrapes HTML tables
df = pd.read_csv("https://example.com/data.csv")       # directly from URL
df = pd.read_clipboard()                                # from copied data
```

### 4.4 Writing Data
```python
df.to_csv("output.csv", index=False)               # index=False avoids extra column
df.to_csv("output.csv", index=False, columns=['Name','Age'])
df.to_excel("output.xlsx", index=False, sheet_name="Sheet1")
df.to_json("output.json", orient="records")
df.to_sql("table_name", conn, if_exists="replace")
df.to_parquet("output.parquet")
```

### 4.5 Handling Large Files
```python
# Read in chunks to save memory
chunks = pd.read_csv("huge_file.csv", chunksize=100000)
for chunk in chunks:
    process(chunk)          # process each chunk separately

# Specify dtypes upfront to reduce memory
dtypes = {'id':'int32', 'category':'category', 'value':'float32'}
df = pd.read_csv("data.csv", dtype=dtypes)
```

---

## 5. head()

Returns the **first N rows** of a DataFrame (default 5). Used to quickly preview data.

```python
df = pd.read_csv("data.csv")

print(df.head())        # first 5 rows (default)
print(df.head(10))      # first 10 rows
print(df.head(1))       # just the first row
print(df.head(-2))      # all rows EXCEPT the last 2
```

### Practical Use
```python
# Quick sanity check after loading data
df = pd.read_csv("sales.csv")
print("Shape:", df.shape)
print(df.head())            # verify columns look correct, no parsing errors
```

---

## 6. tail()

Returns the **last N rows** of a DataFrame (default 5). Useful for checking the end of a dataset (e.g., most recent records in time-series data).

```python
print(df.tail())        # last 5 rows (default)
print(df.tail(3))       # last 3 rows
print(df.tail(-2))      # all rows EXCEPT the first 2
```

### Practical Use
```python
# Checking if data is sorted chronologically / most recent entries
sales_df = pd.read_csv("daily_sales.csv", parse_dates=['date'])
print(sales_df.tail())      # see the most recent dates

# Combine head + tail to see both ends of a large dataset
print(pd.concat([df.head(3), df.tail(3)]))
```

---

## 7. info()

Prints a **concise summary** of the DataFrame: column names, non-null counts, data types, and memory usage. Your first diagnostic step after loading any dataset.

```python
df.info()
```

**Sample output:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   customer_id  1000 non-null   int64
 1   name         1000 non-null   object
 2   age          980 non-null    float64
 3   city         995 non-null    object
 4   signup_date  1000 non-null   datetime64[ns]
dtypes: datetime64[ns](1), float64(1), int64(1), object(2)
memory usage: 39.2+ KB
```

### What to Look For
| Signal | Meaning |
|---|---|
| `Non-Null Count` < total rows | Column has missing values |
| `Dtype` = `object` for a numeric column | Data may need cleaning/conversion |
| Unexpectedly high memory usage | Consider downcasting dtypes / using `category` |

### Useful Parameters
```python
df.info(verbose=True)          # show all columns even if many
df.info(memory_usage='deep')   # accurate memory usage for object columns
df.info(show_counts=True)      # force showing non-null counts

# Related diagnostic methods
print(df.dtypes)               # just the data types
print(df.memory_usage(deep=True))  # memory per column
print(df.isnull().sum())       # missing value counts per column
```

---

## 8. describe()

Generates **descriptive statistics** for numeric columns by default: count, mean, std, min, quartiles, max.

```python
df.describe()
```

**Sample output:**
```
              age        salary
count  980.000000    1000.00000
mean    34.521429   55234.50000
std     10.233145   15678.23000
min     18.000000   25000.00000
25%     27.000000   43500.00000
50%     34.000000   54000.00000
75%     41.000000   65500.00000
max     78.000000  150000.00000
```

### 8.1 Interpreting the Output
| Row | Meaning |
|---|---|
| `count` | Number of non-null values |
| `mean` | Average |
| `std` | Standard deviation (spread) |
| `min` / `max` | Range boundaries |
| `25%` / `50%` / `75%` | Quartiles (50% = median) |

### 8.2 Useful Parameters
```python
df.describe()                           # numeric columns only (default)
df.describe(include='object')           # categorical/text columns only
df.describe(include='all')              # all columns combined
df.describe(percentiles=[0.1, 0.5, 0.9])   # custom percentiles
df.describe(exclude=[np.number])        # exclude numeric columns

# For categorical columns, output includes: count, unique, top, freq
print(df['city'].describe())
# count     995
# unique     12
# top      Mumbai
# freq      145
```

### 8.3 Practical Use — Quick EDA
```python
df = pd.read_csv("data.csv")

# The standard first-look combo
print(df.shape)
df.info()
df.describe()
df.describe(include='object')
df.head()

# Spotting issues from describe()
stats = df.describe()
if stats.loc['min', 'age'] < 0:
    print("⚠️ Negative age detected - data error")
if stats.loc['max', 'age'] > 120:
    print("⚠️ Unrealistic age detected - possible outlier")
```

---

## 9. Selecting Columns

### 9.1 Single Column
```python
df['Name']            # returns a Series
df.Name                # dot notation (avoid if column name has spaces/matches a method)
df[['Name']]           # returns a DataFrame (double brackets!)
```

### 9.2 Multiple Columns
```python
df[['Name', 'Age']]              # select multiple columns → DataFrame
df.loc[:, ['Name', 'Age']]       # equivalent, using loc
df.loc[:, 'Name':'Marks']        # column range (label-based, INCLUSIVE)
df.iloc[:, 0:2]                  # column range (position-based, EXCLUSIVE)
df.iloc[:, [0, 2]]               # specific column positions
```

### 9.3 loc vs iloc (Critical Distinction)
| | `loc` | `iloc` |
|---|---|---|
| Based on | **Labels** (names) | **Integer positions** |
| Range end | **Inclusive** | **Exclusive** |
| Example | `df.loc[0:3, 'Name']` | `df.iloc[0:3, 0]` |

```python
df = pd.DataFrame({'Name':['A','B','C','D'], 'Age':[20,21,22,23]},
                   index=['w','x','y','z'])

print(df.loc['w'])              # row with label 'w'
print(df.loc['w':'y'])          # rows w to y INCLUSIVE
print(df.loc['w', 'Name'])      # single value
print(df.loc[:, 'Name'])        # entire column

print(df.iloc[0])               # row at position 0
print(df.iloc[0:2])             # rows 0-1 (2 EXCLUSIVE)
print(df.iloc[0, 1])            # value at row0, col1
print(df.iloc[[0, 2], [0, 1]])  # specific rows & columns
```

### 9.4 Selecting by Data Type
```python
df.select_dtypes(include='number')       # only numeric columns
df.select_dtypes(include='object')       # only text/categorical columns
df.select_dtypes(exclude='number')       # everything except numeric
df.select_dtypes(include=['int64','float64'])
```

### 9.5 Dropping / Renaming Columns
```python
# Selecting the "opposite" of some columns
df.drop(columns=['Age'])                 # returns new df; original unchanged
df.drop(columns=['Age'], inplace=True)   # modifies original directly

# Renaming
df.rename(columns={'Name':'FullName', 'Age':'Years'}, inplace=True)
df.columns = ['col1', 'col2', 'col3']    # rename ALL columns at once
```

---

## 10. Filtering Rows

### 10.1 Boolean Conditions (Most Common)
```python
df[df['Age'] > 23]                          # single condition
df[df['Name'] == 'Ravi']

# Multiple conditions - use & | ~ with parentheses (NOT and/or/not)
df[(df['Age'] > 20) & (df['Marks'] > 80)]           # AND
df[(df['Age'] < 20) | (df['Marks'] > 90)]           # OR
df[~(df['Age'] > 25)]                                # NOT

# isin() - membership check
df[df['City'].isin(['Mumbai', 'Delhi'])]
df[~df['City'].isin(['Mumbai', 'Delhi'])]           # NOT in list

# between()
df[df['Age'].between(20, 25)]                        # inclusive range

# String conditions
df[df['Name'].str.startswith('R')]
df[df['Name'].str.contains('a', case=False)]
df[df['Email'].str.endswith('.com')]
```

### 10.2 query() Method (Readable Alternative)
```python
df.query('Age > 23')
df.query('Age > 20 and Marks > 80')
df.query('City in ["Mumbai", "Delhi"]')

# Using variables inside query with @
min_age = 21
df.query('Age > @min_age')
```

### 10.3 loc for Conditional Filtering
```python
df.loc[df['Age'] > 23]                      # rows only
df.loc[df['Age'] > 23, 'Name']              # rows + specific column
df.loc[df['Age'] > 23, ['Name', 'Marks']]   # rows + multiple columns

# Conditional value assignment
df.loc[df['Marks'] >= 90, 'Grade'] = 'A'
df.loc[df['Marks'] < 40, 'Status'] = 'Fail'
```

### 10.4 Filtering with isnull/notnull
```python
df[df['Age'].isnull()]         # rows with missing Age
df[df['Age'].notnull()]        # rows WITHOUT missing Age
df.dropna(subset=['Age'])      # same effect via dropping
```

### 10.5 Sampling Rows
```python
df.sample(5)                    # 5 random rows
df.sample(frac=0.1)             # random 10% of rows
df.sample(5, random_state=42)   # reproducible sample
df.nlargest(5, 'Marks')         # top 5 by column value
df.nsmallest(5, 'Age')          # bottom 5 by column value
```

---

## 11. Sorting

### 11.1 sort_values() — Sort by Column(s)
```python
df.sort_values('Age')                        # ascending (default)
df.sort_values('Age', ascending=False)       # descending
df.sort_values(['Age', 'Marks'])             # multi-column sort
df.sort_values(['Age', 'Marks'],
               ascending=[True, False])      # mixed order per column

df.sort_values('Age', inplace=True)          # modify original
df.sort_values('Age', na_position='first')   # NaNs at top instead of bottom
```

### 11.2 sort_index() — Sort by Row/Column Labels
```python
df.sort_index()                    # sort by row index ascending
df.sort_index(ascending=False)     # descending
df.sort_index(axis=1)              # sort COLUMNS alphabetically
```

### 11.3 rank()
```python
df['Rank'] = df['Marks'].rank(ascending=False)   # 1 = highest marks
df['Rank'] = df['Marks'].rank(method='dense', ascending=False)
```

### 11.4 Practical Examples
```python
# Top 5 highest earners
top_earners = df.sort_values('Salary', ascending=False).head(5)

# Sort within groups
df.sort_values(['Department', 'Salary'], ascending=[True, False])

# Custom sort order (categorical)
order = ['Low', 'Medium', 'High']
df['Priority'] = pd.Categorical(df['Priority'], categories=order, ordered=True)
df.sort_values('Priority')
```

---

## 12. Adding/Removing Columns

### 12.1 Adding Columns
```python
df['Country'] = 'India'                      # constant value for all rows
df['Age_Next_Year'] = df['Age'] + 1          # derived from existing column
df['Total'] = df['Marks1'] + df['Marks2']    # combining columns
df['Pass'] = df['Marks'] >= 40               # boolean column

# Using apply() with a custom function
df['Grade'] = df['Marks'].apply(
    lambda x: 'A' if x>=90 else 'B' if x>=75 else 'C'
)

# Using np.where (vectorized, faster than apply)
df['Result'] = np.where(df['Marks'] >= 40, 'Pass', 'Fail')

# Multiple conditions with np.select
conditions = [df['Marks']>=90, df['Marks']>=75, df['Marks']>=40]
choices    = ['A', 'B', 'C']
df['Grade'] = np.select(conditions, choices, default='F')

# insert() - add at a specific position
df.insert(1, 'ID', range(1, len(df)+1))     # position, column name, values

# assign() - returns a NEW dataframe (doesn't modify original)
df2 = df.assign(Double_Marks = df['Marks'] * 2)
```

### 12.2 Removing Columns
```python
df.drop('Country', axis=1)                    # returns new df
df.drop(columns=['Country', 'Age_Next_Year'])  # drop multiple
df.drop('Country', axis=1, inplace=True)       # modify original
del df['Country']                              # alternative syntax
popped = df.pop('Country')                     # removes AND returns the column
```

### 12.3 Reordering Columns
```python
df = df[['Name', 'Age', 'Marks', 'Grade']]     # explicit order
cols = ['Name'] + [c for c in df.columns if c != 'Name']
df = df[cols]                                   # move 'Name' to front
```

### 12.4 Working with Row-wise Column Creation
```python
# apply() across a row (axis=1)
df['Full_Info'] = df.apply(lambda row: f"{row['Name']} ({row['Age']})", axis=1)

# Combine multiple columns conditionally
df['Category'] = df.apply(
    lambda row: 'Senior' if row['Age'] > 30 and row['Marks'] > 80 else 'Junior',
    axis=1
)
```

---

## 13. Handling Missing Values

Missing data appears as `NaN` (Not a Number) or `None`. Handling it correctly is critical before modeling.

### 13.1 Detecting Missing Values
```python
df.isnull()                 # DataFrame of True/False
df.isna()                   # same as isnull() (alias)
df.notnull()                # opposite

df.isnull().sum()           # count of NaN per column
df.isnull().sum().sum()     # total NaN in the whole DataFrame
df.isnull().mean() * 100    # % missing per column

df.isnull().any()           # which columns have ANY missing value
df.isnull().all()           # which columns are ENTIRELY missing

# Visualize missing data pattern
import seaborn as sns
sns.heatmap(df.isnull(), cbar=False)
```

### 13.2 Dropping Missing Values
```python
df.dropna()                              # drop rows with ANY NaN
df.dropna(axis=1)                        # drop COLUMNS with any NaN
df.dropna(how='all')                     # drop rows where ALL values are NaN
df.dropna(subset=['Age', 'Salary'])      # drop rows with NaN in specific columns
df.dropna(thresh=3)                      # keep rows with at least 3 non-NaN values
df.dropna(inplace=True)                  # modify original
```

### 13.3 Filling Missing Values
```python
df.fillna(0)                                  # fill all NaN with 0
df['Age'].fillna(df['Age'].mean(), inplace=True)      # mean imputation
df['Age'].fillna(df['Age'].median(), inplace=True)    # median (robust to outliers)
df['City'].fillna(df['City'].mode()[0], inplace=True) # mode (for categorical)

df.fillna(method='ffill')     # forward fill - propagate last valid value
df.fillna(method='bfill')     # backward fill - use next valid value

df.fillna({'Age': 0, 'City': 'Unknown'})      # different fill value per column

# Fill using group-wise statistics (smarter imputation)
df['Salary'] = df.groupby('Department')['Salary'] \
                  .transform(lambda x: x.fillna(x.mean()))

# Interpolation - for time-series/sequential numeric data
df['Value'] = df['Value'].interpolate(method='linear')
```

### 13.4 Replacing Specific Values
```python
df.replace('?', np.nan, inplace=True)          # treat '?' as missing
df.replace({'City': {'Mumbai':'MUM', 'Delhi':'DEL'}})
df['Gender'].replace(['m','M'], 'Male', inplace=True)
```

### 13.5 Complete Missing-Value Workflow Example
```python
df = pd.read_csv("data.csv")

# Step 1: Inspect
print(df.isnull().sum())
print(df.isnull().mean() * 100)

# Step 2: Drop columns that are mostly empty (>60% missing)
threshold = 0.6
df = df.loc[:, df.isnull().mean() < threshold]

# Step 3: Impute numeric columns with median
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Step 4: Impute categorical columns with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 5: Verify
print("Remaining missing values:", df.isnull().sum().sum())
```

---

## 14. groupby()

Splits data into groups based on column values, applies a function to each group, and combines the results — the **"Split-Apply-Combine"** pattern (like SQL's `GROUP BY`).

```
   Split                Apply               Combine
┌──────────┐        ┌──────────┐        ┌──────────┐
│ Group A  │  →     │ mean()   │  →     │          │
├──────────┤        ├──────────┤        │ Result   │
│ Group B  │  →     │ mean()   │  →     │ Table    │
├──────────┤        ├──────────┤        │          │
│ Group C  │  →     │ mean()   │  →     │          │
└──────────┘        └──────────┘        └──────────┘
```

### 14.1 Basic Grouping
```python
df.groupby('Department')['Salary'].mean()          # avg salary per department
df.groupby('Department')['Salary'].sum()
df.groupby('Department')['Salary'].count()          # count of non-null values
df.groupby('Department').size()                     # count including NaN (row count)

df.groupby('Department')['Salary'].agg(['mean','sum','count','min','max'])
```

### 14.2 Multiple Columns / Multiple Aggregations
```python
# Group by multiple columns
df.groupby(['Department', 'Gender'])['Salary'].mean()

# Different aggregations for different columns
df.groupby('Department').agg({
    'Salary': ['mean', 'max'],
    'Age': 'mean',
    'Name': 'count'
})

# Named aggregation (clean column names, Pandas 0.25+)
df.groupby('Department').agg(
    avg_salary = ('Salary', 'mean'),
    max_salary = ('Salary', 'max'),
    headcount  = ('Name', 'count')
).reset_index()
```

### 14.3 Iterating Groups & Getting Specific Group
```python
for name, group in df.groupby('Department'):
    print(f"\n--- {name} ---")
    print(group.head(2))

sales_dept = df.groupby('Department').get_group('Sales')
```

### 14.4 transform() vs apply() vs agg()
| Method | Returns | Use case |
|---|---|---|
| `agg()` | One value per group (summary) | Aggregated statistics |
| `transform()` | Same shape as original | Add group-stat back to each row |
| `apply()` | Flexible — depends on function | Custom, complex group logic |

```python
# agg() - collapses to one row per group
avg_by_dept = df.groupby('Department')['Salary'].agg('mean')

# transform() - broadcasts back to original shape (great for feature engineering)
df['Dept_Avg_Salary'] = df.groupby('Department')['Salary'].transform('mean')
df['Salary_vs_Dept_Avg'] = df['Salary'] - df['Dept_Avg_Salary']

# apply() - custom function per group
def top_2(group):
    return group.nlargest(2, 'Salary')
top_earners_per_dept = df.groupby('Department').apply(top_2)
```

### 14.5 Filtering Groups
```python
# Keep only departments with more than 5 employees
df.groupby('Department').filter(lambda x: len(x) > 5)
```

### 14.6 groupby with sort, as_index, and reset_index
```python
df.groupby('Department', as_index=False)['Salary'].mean()   # keeps 'Department' as a column
df.groupby('Department')['Salary'].mean().reset_index()     # same effect
df.groupby('Department', sort=False)['Salary'].mean()       # preserve original order
```

### 14.7 Pivot Table (groupby's cousin)
```python
pd.pivot_table(df, values='Salary', index='Department',
               columns='Gender', aggfunc='mean', fill_value=0)

pd.pivot_table(df, values='Salary', index='Department',
               aggfunc=['mean', 'count'], margins=True)   # margins = grand totals

# crosstab - frequency counts between two categorical columns
pd.crosstab(df['Department'], df['Gender'])
pd.crosstab(df['Department'], df['Gender'], normalize='index')   # row percentages
```

---

## 15. Merging/Joining

Combining multiple DataFrames — similar to SQL joins.

### 15.1 merge() — SQL-style Joins
```python
customers = pd.DataFrame({
    'cust_id': [1, 2, 3, 4],
    'name': ['Ravi', 'Meera', 'Sam', 'Priya']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103],
    'cust_id': [1, 2, 5],          # note: 5 doesn't exist in customers
    'amount': [500, 750, 300]
})

# INNER JOIN (default) - only matching rows in BOTH
pd.merge(customers, orders, on='cust_id', how='inner')
#    cust_id   name  order_id  amount
# 0        1   Ravi       101     500
# 1        2  Meera       102     750

# LEFT JOIN - all rows from left, matched where possible
pd.merge(customers, orders, on='cust_id', how='left')
#    cust_id   name  order_id  amount
# 0        1   Ravi     101.0   500.0
# 1        2  Meera     102.0   750.0
# 2        3    Sam       NaN     NaN
# 3        4  Priya       NaN     NaN

# RIGHT JOIN - all rows from right
pd.merge(customers, orders, on='cust_id', how='right')

# OUTER JOIN - all rows from both, NaN where no match
pd.merge(customers, orders, on='cust_id', how='outer')

# Cross join - every combination (Pandas 1.2+)
pd.merge(customers, orders, how='cross')
```

### Join Types Visual
```
INNER  : only matching rows          LEFT   : all of left + matches
  A ∩ B                                A  ⟕  B

RIGHT  : all of right + matches      OUTER  : everything, matched where possible
  A ⟖ B                                A  ⟗  B
```

### 15.2 Merging on Different Column Names
```python
pd.merge(customers, orders, left_on='cust_id', right_on='cust_id')

# When key columns have different names
df1 = pd.DataFrame({'id': [1,2,3], 'name': ['A','B','C']})
df2 = pd.DataFrame({'customer_id': [1,2,4], 'value': [10,20,30]})
pd.merge(df1, df2, left_on='id', right_on='customer_id', how='left')
```

### 15.3 Merging on Multiple Columns
```python
pd.merge(df1, df2, on=['col1', 'col2'], how='inner')
```

### 15.4 Handling Duplicate Column Names
```python
pd.merge(df1, df2, on='id', suffixes=('_left', '_right'))
```

### 15.5 Merging on Index
```python
pd.merge(df1, df2, left_index=True, right_index=True)
df1.join(df2, how='left')          # join() = merge() but on INDEX by default
df1.join(df2, on='id')             # join df1's column to df2's index
```

### 15.6 concat() — Stacking DataFrames
```python
# Vertical stacking (same columns, more rows) - like UNION
df1 = pd.DataFrame({'A':[1,2], 'B':[3,4]})
df2 = pd.DataFrame({'A':[5,6], 'B':[7,8]})

pd.concat([df1, df2])                       # keeps original index (0,1,0,1)
pd.concat([df1, df2], ignore_index=True)    # resets index (0,1,2,3)

# Horizontal stacking (same rows, more columns)
pd.concat([df1, df2], axis=1)

# Concatenate many files
import glob
files = glob.glob("data/*.csv")
combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# With keys - creates a multi-index to track source
pd.concat([df1, df2], keys=['2023', '2024'])
```

### 15.7 merge vs concat vs join — Quick Guide
| Method | Use when |
|---|---|
| `merge()` | Combining on a **common key column**, like SQL JOIN |
| `concat()` | **Stacking** DataFrames vertically or horizontally |
| `join()` | Combining based on the **index** |

### 15.8 Validating Merges (Best Practice)
```python
# Check for unexpected row multiplication (duplicate keys cause fan-out)
merged = pd.merge(customers, orders, on='cust_id', how='left', validate='one_to_many')

# indicator=True shows which side each row came from
merged = pd.merge(customers, orders, on='cust_id', how='outer', indicator=True)
print(merged['_merge'].value_counts())
# both          2
# left_only     2
# right_only    1
```

---

## 16. Practical Mini Project

```python
# ============================================================
# PANDAS MINI PROJECT: Retail Sales Analysis
# ============================================================
import pandas as pd
import numpy as np

# --- 1. Create sample datasets ---
np.random.seed(42)

sales = pd.DataFrame({
    'order_id':  range(1, 21),
    'cust_id':   np.random.randint(1, 8, 20),
    'product':   np.random.choice(['Laptop','Mouse','Keyboard','Monitor'], 20),
    'quantity':  np.random.randint(1, 5, 20),
    'price':     np.random.choice([500, 25, 45, 200], 20),
    'order_date':pd.date_range('2024-01-01', periods=20, freq='3D')
})
sales.loc[[2, 7, 15], 'price'] = np.nan       # inject some missing values

customers = pd.DataFrame({
    'cust_id': range(1, 8),
    'name':    ['Ravi','Meera','Sam','Priya','Arjun','Kavya','Rohit'],
    'city':    ['Mumbai','Delhi','Mumbai','Pune','Delhi','Pune','Mumbai']
})

# --- 2. First look ---
print(sales.head())
print(sales.tail(3))
sales.info()
print(sales.describe())

# --- 3. Handle missing values ---
print("\nMissing values:\n", sales.isnull().sum())
sales['price'] = sales.groupby('product')['price'].transform(
    lambda x: x.fillna(x.median())
)
print("After fill:", sales.isnull().sum().sum())

# --- 4. Feature engineering ---
sales['total'] = sales['quantity'] * sales['price']
sales['month'] = sales['order_date'].dt.month_name()

# --- 5. Filtering ---
high_value = sales[sales['total'] > 500]
laptop_orders = sales[sales['product'] == 'Laptop']
mumbai_period = sales[(sales['order_date'] >= '2024-01-01') &
                      (sales['order_date'] <= '2024-01-15')]

print(f"\nHigh value orders: {len(high_value)}")

# --- 6. Sorting ---
top_orders = sales.sort_values('total', ascending=False).head(5)
print("\nTop 5 orders by value:\n", top_orders[['order_id','product','total']])

# --- 7. Merging with customer data ---
full_data = pd.merge(sales, customers, on='cust_id', how='left')
print("\nMerged shape:", full_data.shape)

# --- 8. GroupBy analysis ---
print("\n--- Revenue by Product ---")
product_summary = full_data.groupby('product').agg(
    total_revenue = ('total', 'sum'),
    avg_order_val = ('total', 'mean'),
    order_count   = ('order_id', 'count')
).sort_values('total_revenue', ascending=False)
print(product_summary)

print("\n--- Revenue by City ---")
print(full_data.groupby('city')['total'].sum().sort_values(ascending=False))

print("\n--- Top Customer ---")
customer_spend = full_data.groupby('name')['total'].sum().sort_values(ascending=False)
print(f"Top customer: {customer_spend.index[0]} (₹{customer_spend.iloc[0]:.0f})")

# --- 9. Pivot table ---
pivot = pd.pivot_table(full_data, values='total', index='city',
                       columns='product', aggfunc='sum', fill_value=0)
print("\n--- City x Product Pivot ---\n", pivot)

# --- 10. Export final result ---
full_data.to_csv("processed_sales.csv", index=False)
print("\n✅ Analysis complete. File saved.")
```

---

## 17. Cheatsheet

### Creating
```python
pd.Series([1,2,3])              pd.DataFrame({'a':[1,2],'b':[3,4]})
pd.read_csv('f.csv')            pd.read_excel('f.xlsx')
```

### Inspecting
```python
df.head()   df.tail()   df.info()   df.describe()
df.shape    df.columns  df.dtypes   df.isnull().sum()
```

### Selecting
```python
df['col']          df[['c1','c2']]         df.loc[row, col]
df.iloc[row, col]  df.select_dtypes('number')
```

### Filtering
```python
df[df['col'] > 5]                      df[(cond1) & (cond2)]
df[df['col'].isin([...])]              df.query('col > 5')
```

### Sorting
```python
df.sort_values('col')                  df.sort_values('col', ascending=False)
df.sort_index()
```

### Modifying columns
```python
df['new'] = df['a'] + df['b']          df.drop(columns=['col'])
df.rename(columns={'old':'new'})       np.where(cond, x, y)
```

### Missing values
```python
df.isnull().sum()      df.dropna()             df.fillna(value)
df.fillna(df['c'].mean())                       df.interpolate()
```

### Grouping
```python
df.groupby('col')['x'].mean()
df.groupby('col').agg(avg=('x','mean'), cnt=('x','count'))
df.groupby('col')['x'].transform('mean')
```

### Merging
```python
pd.merge(df1, df2, on='key', how='inner')      # inner/left/right/outer
pd.concat([df1, df2], ignore_index=True)        # stacking
df1.join(df2)                                    # index-based
```

### Key Rules to Remember 🏆
1. `df[['col']]` returns a DataFrame; `df['col']` returns a Series.
2. `loc` is **label-based & inclusive**; `iloc` is **position-based & exclusive**.
3. Use `&`, `|`, `~` with parentheses around each condition when filtering.
4. Most operations return a **new** object unless `inplace=True` is passed.
5. `merge()` for joining on keys; `concat()` for stacking; `join()` for index-based merges.
6. `groupby().transform()` keeps original row count — ideal for feature engineering.
7. Always check `.isnull().sum()` and `.info()` right after loading data.
8. `on=`, `how=` in merges — always verify row counts before/after to catch fan-out duplication.

---

## 📌 Summary

| Topic | Key Point |
|---|---|
| **Series** | 1-D labeled array — a single column |
| **DataFrame** | 2-D labeled table — the core Pandas object |
| **read_csv** | Loads tabular data; tune with `sep`, `dtype`, `parse_dates`, `na_values` |
| **head/tail** | Preview first/last N rows |
| **info()** | Structure, dtypes, non-null counts, memory |
| **describe()** | Statistical summary — count, mean, std, quartiles |
| **Selecting columns** | `df['col']`, `df[['c1','c2']]`, `loc`/`iloc` |
| **Filtering rows** | Boolean masks, `.isin()`, `.query()` |
| **Sorting** | `sort_values()`, `sort_index()` |
| **Add/remove columns** | Direct assignment, `apply()`, `np.where()`, `drop()` |
| **Missing values** | `isnull()`, `dropna()`, `fillna()`, group-wise imputation |
| **groupby()** | Split-apply-combine; `agg()`, `transform()`, pivot tables |
| **Merging/joining** | `merge()` (SQL joins), `concat()` (stacking), `join()` (index-based) |

---

*📁 Previous: Part 4 → NumPy*
*📁 Next: Part 6 → Data Visualization (Matplotlib & Seaborn)*

---
### 🔖 Tags
`#Pandas` `#Python` `#DataScience` `#DataAnalysis` `#DataFrame` `#Notes`
