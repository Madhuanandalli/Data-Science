# 📈 Data Science Notes — Part 6: Matplotlib

> Python's foundational plotting library — theory + practical syntax and code examples for every core chart type.

---

## Table of Contents
1. [Introduction to Matplotlib](#1-introduction-to-matplotlib)
2. [Anatomy of a Matplotlib Plot](#2-anatomy-of-a-matplotlib-plot)
3. [Line Plots](#3-line-plots)
4. [Bar Charts](#4-bar-charts)
5. [Histograms](#5-histograms)
6. [Scatter Plots](#6-scatter-plots)
7. [Labels](#7-labels)
8. [Titles](#8-titles)
9. [Legends](#9-legends)
10. [Subplots (Bonus)](#10-subplots-bonus)
11. [Styling & Saving (Bonus)](#11-styling--saving-bonus)
12. [Practical Mini Project](#12-practical-mini-project)
13. [Cheatsheet](#13-cheatsheet)

---

## 1. Introduction to Matplotlib

**Matplotlib** is Python's core 2-D plotting library — the foundation almost every other visualization library (Seaborn, Pandas `.plot()`) is built on top of.

### Why Matplotlib?
| Feature | Benefit |
|---|---|
| **Full control** | Every pixel — colors, sizes, ticks, spacing — is customizable |
| **Wide chart support** | Line, bar, histogram, scatter, pie, box, heatmap, 3-D |
| **Integrates everywhere** | Works with NumPy, Pandas, Jupyter, web apps |
| **Publication-ready** | Used in academic papers, dashboards, reports |

### Installation & Import
```bash
pip install matplotlib
```
```python
import matplotlib.pyplot as plt      # 'plt' is the universal convention
import numpy as np
print(plt.matplotlib.__version__)

# In Jupyter notebooks, this makes plots appear inline
%matplotlib inline
```

### The Two Interfaces
```python
# 1. Pyplot interface (MATLAB-style) - quick & simple, good for single plots
plt.plot([1,2,3], [4,5,6])
plt.show()

# 2. Object-Oriented interface - more control, RECOMMENDED for multi-plot/complex figures
fig, ax = plt.subplots()
ax.plot([1,2,3], [4,5,6])
plt.show()
```
> 💡 This guide primarily uses the simple `plt.` interface (best for learning), with the OO `fig, ax` style shown for subplots.

---

## 2. Anatomy of a Matplotlib Plot

```
                    Title
        ┌─────────────────────────────┐
        │                        ▲     │
  y     │          Legend  ┌───┐ │     │
 label  │                  │ • │ │     │
        │           ╱╲     └───┘ │     │
        │          ╱  ╲          │  ← Figure (the whole canvas)
        │    •    ╱    ╲    •    │  ← Axes (the actual plot area)
        │     ╲  ╱      ╲  ╱     │
        │      ╲╱        ╲╱      │
        └─────────────────────────────┘
                x label
```

| Term | Meaning |
|---|---|
| **Figure** | The entire canvas/window that holds everything |
| **Axes** | The actual plot area (a Figure can contain multiple Axes) |
| **Axis** | The x or y number-line with ticks and labels |
| **Title** | Text describing the whole plot |
| **Label** | Text describing an axis |
| **Legend** | Key explaining what colors/markers represent |
| **Ticks** | The marked values along an axis |

```python
fig = plt.figure()          # creates a Figure
ax = fig.add_subplot()      # adds an Axes to it
# OR shorthand:
fig, ax = plt.subplots()    # creates both at once
```

---

## 3. Line Plots

Best for showing **trends over a continuous variable** — most commonly time.

### 3.1 Basic Syntax
```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]

plt.plot(x, y)
plt.show()
```

### 3.2 Customizing Line Appearance
```python
plt.plot(x, y,
         color='blue',          # or 'b', or hex '#1f77b4'
         linestyle='--',        # '-' solid, '--' dashed, ':' dotted, '-.' dash-dot
         linewidth=2,
         marker='o',            # 'o' circle, 's' square, '^' triangle, 'x', '*'
         markersize=8,
         markerfacecolor='red',
         alpha=0.8)              # transparency 0-1
plt.show()

# Shorthand format string: [color][marker][linestyle]
plt.plot(x, y, 'go--')          # green circles, dashed line
plt.plot(x, y, 'r^-')           # red triangles, solid line
```

### 3.3 Multiple Lines on One Plot
```python
x = [1, 2, 3, 4, 5]
y1 = [10, 20, 15, 25, 30]
y2 = [5, 15, 10, 20, 25]

plt.plot(x, y1, label='Product A', color='blue', marker='o')
plt.plot(x, y2, label='Product B', color='orange', marker='s')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Sales Comparison')
plt.legend()
plt.show()
```

### 3.4 Line Plot with NumPy / Pandas Data
```python
import numpy as np
import pandas as pd

# NumPy - mathematical function
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# Pandas - time series
dates = pd.date_range('2024-01-01', periods=30)
values = np.cumsum(np.random.randn(30)) + 100
df = pd.DataFrame({'date': dates, 'value': values})

plt.plot(df['date'], df['value'])
plt.xticks(rotation=45)          # rotate date labels for readability
plt.show()

# Directly from a DataFrame (Pandas' own plotting, uses Matplotlib underneath)
df.plot(x='date', y='value', kind='line')
plt.show()
```

### 3.5 Filling Area Under a Line
```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.fill_between(x, y, alpha=0.3)              # shade area under the curve
plt.fill_between(x, y, where=(y>0), color='green', alpha=0.3)   # conditional fill
plt.show()
```

### When to Use Line Plots
✅ Time-series data (stock prices, sensor readings, sales over months)
✅ Showing trends and rate of change
❌ Not for categorical/unordered data (use bar chart instead)

---

## 4. Bar Charts

Best for **comparing quantities across categories**.

### 4.1 Vertical Bar Chart
```python
categories = ['Math', 'Physics', 'Chemistry', 'English']
scores = [85, 78, 92, 88]

plt.bar(categories, scores)
plt.show()

# Customized
plt.bar(categories, scores,
        color=['#4C72B0','#DD8452','#55A868','#C44E52'],
        width=0.6,
        edgecolor='black')
plt.show()
```

### 4.2 Horizontal Bar Chart
```python
plt.barh(categories, scores, color='teal')
plt.show()
# Useful when category names are long (more readable than vertical)
```

### 4.3 Grouped Bar Chart (Multiple Series)
```python
import numpy as np

subjects = ['Math', 'Physics', 'Chemistry']
student_A = [85, 78, 92]
student_B = [75, 88, 80]

x = np.arange(len(subjects))       # [0, 1, 2]
width = 0.35                       # bar width

plt.bar(x - width/2, student_A, width, label='Ravi')
plt.bar(x + width/2, student_B, width, label='Meera')

plt.xticks(x, subjects)            # replace 0,1,2 with subject names
plt.xlabel('Subject')
plt.ylabel('Marks')
plt.legend()
plt.show()
```

### 4.4 Stacked Bar Chart
```python
months = ['Jan', 'Feb', 'Mar']
online  = [200, 250, 300]
offline = [150, 180, 160]

plt.bar(months, online, label='Online')
plt.bar(months, offline, bottom=online, label='Offline')   # stacks on top
plt.legend()
plt.ylabel('Sales')
plt.show()
```

### 4.5 Bar Chart from Pandas (Common EDA pattern)
```python
import pandas as pd

df = pd.DataFrame({'City':['Mumbai','Delhi','Pune','Mumbai','Delhi'],
                   'Sales':[100, 150, 80, 120, 130]})

city_totals = df.groupby('City')['Sales'].sum().sort_values(ascending=False)
city_totals.plot(kind='bar', color='skyblue')
plt.ylabel('Total Sales')
plt.title('Sales by City')
plt.xticks(rotation=0)
plt.show()
```

### 4.6 Adding Value Labels on Bars
```python
bars = plt.bar(categories, scores, color='cornflowerblue')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f'{height}', ha='center', va='bottom')
plt.show()
```

### When to Use Bar Charts
✅ Comparing discrete categories (sales by region, marks by subject)
✅ Showing counts/frequencies of categorical data
❌ Not for continuous trends over time (use line plot)

---

## 5. Histograms

Show the **distribution** of a single numeric variable by grouping values into bins. Different from a bar chart — histograms represent continuous ranges, not discrete categories, and have **no gaps** between bars.

### 5.1 Basic Histogram
```python
import numpy as np

data = np.random.normal(loc=50, scale=15, size=1000)   # simulated exam scores

plt.hist(data)
plt.show()

# Controlling bins
plt.hist(data, bins=20)                # more bins = finer detail
plt.hist(data, bins=[0,20,40,60,80,100])   # custom bin edges
plt.show()
```

### 5.2 Customizing Histograms
```python
plt.hist(data,
         bins=30,
         color='skyblue',
         edgecolor='black',
         alpha=0.7)
plt.show()

# Density (normalized so the area sums to 1, for comparing with a PDF curve)
plt.hist(data, bins=30, density=True)
plt.show()

# Cumulative histogram
plt.hist(data, bins=30, cumulative=True)
plt.show()
```

### 5.3 Multiple Histograms (Comparing Distributions)
```python
group_a = np.random.normal(50, 10, 500)
group_b = np.random.normal(65, 12, 500)

plt.hist(group_a, bins=20, alpha=0.5, label='Group A', color='blue')
plt.hist(group_b, bins=20, alpha=0.5, label='Group B', color='orange')
plt.legend()
plt.show()

# Side-by-side bars instead of overlapping
plt.hist([group_a, group_b], bins=15, label=['Group A','Group B'])
plt.legend()
plt.show()
```

### 5.4 Histogram from Pandas
```python
import pandas as pd

df = pd.DataFrame({'age': np.random.randint(18, 70, 500)})

df['age'].hist(bins=15, color='salmon', edgecolor='black')
plt.title('Age Distribution')
plt.show()

# Multiple columns at once
df2 = pd.DataFrame({'math': np.random.randint(40,100,200),
                    'science': np.random.randint(40,100,200)})
df2.hist(bins=15, figsize=(10,4))
plt.show()
```

### 5.5 Inspecting Bin Data
```python
counts, bin_edges, patches = plt.hist(data, bins=10)
print("Counts per bin:", counts)
print("Bin edges:", bin_edges)
plt.show()
```

### When to Use Histograms
✅ Understanding the shape/spread of a single numeric variable (age, salary, test scores)
✅ Detecting skewness, checking for normal distribution
✅ Spotting outliers (isolated bars far from the main cluster)
❌ Not for comparing categories (use bar chart) or two numeric variables (use scatter plot)

---

## 6. Scatter Plots

Show the **relationship between two numeric variables** — each point represents one observation.

### 6.1 Basic Scatter Plot
```python
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11]
y = [99, 86, 87, 88, 100, 86, 103, 87, 94, 78]

plt.scatter(x, y)
plt.show()
```

### 6.2 Customizing Points
```python
plt.scatter(x, y,
            color='purple',
            s=100,             # marker size
            alpha=0.6,         # transparency (helps with overlapping points)
            marker='o',        # 'o','s','^','D','*', etc.
            edgecolors='black')
plt.show()
```

### 6.3 Color-Coding by a Third Variable
```python
import numpy as np

np.random.seed(1)
x = np.random.rand(100) * 100
y = np.random.rand(100) * 100
colors = np.random.rand(100)          # a value per point (e.g. temperature)

scatter = plt.scatter(x, y, c=colors, cmap='viridis', s=80, alpha=0.7)
plt.colorbar(scatter, label='Intensity')     # shows the color scale
plt.show()
```

### 6.4 Size-Coding by a Fourth Variable (Bubble Chart)
```python
sizes = np.random.rand(100) * 500      # e.g. population, revenue

plt.scatter(x, y, s=sizes, c=colors, cmap='plasma', alpha=0.5, edgecolors='w')
plt.show()
```

### 6.5 Scatter Plot with Categories
```python
import pandas as pd

df = pd.DataFrame({
    'height': np.random.normal(165, 10, 150),
    'weight': np.random.normal(65, 12, 150),
    'gender': np.random.choice(['Male','Female'], 150)
})

for gender in df['gender'].unique():
    subset = df[df['gender'] == gender]
    plt.scatter(subset['height'], subset['weight'], label=gender, alpha=0.6)

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend()
plt.show()
```

### 6.6 Adding a Trend Line
```python
x = np.array([5, 7, 8, 7, 2, 17, 2, 9, 4, 11])
y = np.array([99, 86, 87, 88, 100, 86, 103, 87, 94, 78])

plt.scatter(x, y, label='Data points')

# Fit a linear trend line
z = np.polyfit(x, y, 1)             # degree-1 polynomial (straight line)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", label='Trend line')
plt.legend()
plt.show()
```

### 6.7 Scatter Matrix (All Pairs at Once)
```python
from pandas.plotting import scatter_matrix

df_num = pd.DataFrame({
    'age': np.random.randint(20,60,100),
    'salary': np.random.randint(30000,100000,100),
    'experience': np.random.randint(0,30,100)
})
scatter_matrix(df_num, figsize=(8,8), diagonal='hist')
plt.show()
```

### When to Use Scatter Plots
✅ Checking correlation/relationship between two numeric variables
✅ Detecting clusters, patterns, or outliers
✅ Visualizing regression fit
❌ Not for a single variable's distribution (use histogram) or categorical comparisons (use bar chart)

---

## 7. Labels

Labels describe what the **x-axis** and **y-axis** represent. Every plot should have them.

### 7.1 Basic Axis Labels
```python
plt.plot([1,2,3], [10,20,30])
plt.xlabel('Time (months)')
plt.ylabel('Revenue (₹ thousands)')
plt.show()
```

### 7.2 Styling Labels
```python
plt.xlabel('Time (months)', fontsize=12, fontweight='bold', color='darkblue')
plt.ylabel('Revenue (₹ thousands)', fontsize=12, fontweight='bold', color='darkblue')
plt.show()

# Positioning
plt.xlabel('Time', labelpad=15)     # extra spacing from axis
plt.ylabel('Value', rotation=0, labelpad=30)   # horizontal y-label
```

### 7.3 Customizing Tick Labels
```python
x = [1,2,3,4,5]
y = [10,20,15,25,30]
plt.plot(x, y)

plt.xticks([1,2,3,4,5], ['Jan','Feb','Mar','Apr','May'])   # replace tick labels
plt.yticks(fontsize=10)
plt.xticks(rotation=45)                                     # rotate for readability
plt.show()
```

### 7.4 Annotating Specific Points
```python
plt.plot(x, y, marker='o')

plt.annotate('Peak', xy=(5, 30), xytext=(4, 35),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.text(3, 16, 'Dip here', fontsize=9, color='red')
plt.show()
```

### 7.5 Adding Gridlines (aids readability of labeled values)
```python
plt.plot(x, y)
plt.grid(True)
plt.grid(True, linestyle='--', alpha=0.5, axis='y')   # only horizontal, dashed
plt.show()
```

---

## 8. Titles

The title summarizes what the plot shows — always place it prominently at the top.

### 8.1 Basic Title
```python
plt.plot([1,2,3], [10,20,15])
plt.title('Monthly Revenue Trend')
plt.show()
```

### 8.2 Styling the Title
```python
plt.title('Monthly Revenue Trend',
          fontsize=16,
          fontweight='bold',
          color='navy',
          loc='left')              # 'left', 'center' (default), 'right'
plt.show()
```

### 8.3 Title with Subtitle (using suptitle for figures)
```python
fig, ax = plt.subplots()
ax.plot([1,2,3],[4,5,6])
fig.suptitle('Sales Report 2024', fontsize=16, fontweight='bold')   # main title
ax.set_title('Q1 Performance', fontsize=11, color='gray')            # subtitle
plt.show()
```

### 8.4 Dynamic Titles (using f-strings)
```python
product = "Laptop"
total_sales = 15000
plt.plot([1,2,3],[10,20,15])
plt.title(f'{product} Sales — Total: ₹{total_sales:,}')
plt.show()
```

### 8.5 Multi-line Titles
```python
plt.title('Monthly Revenue Trend\n(January - June 2024)', fontsize=13)
plt.show()
```

---

## 9. Legends

Legends identify what different colors, markers, or line styles represent — essential whenever a plot has multiple series.

### 9.1 Basic Legend
```python
x = [1,2,3,4,5]
plt.plot(x, [10,20,15,25,30], label='Product A')
plt.plot(x, [5,15,10,20,25], label='Product B')
plt.legend()
plt.show()
```

### 9.2 Legend Positioning
```python
plt.legend(loc='upper left')      # 'upper right' (default), 'lower left',
                                   # 'center', 'best', etc.
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)  # below the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')    # outside the plot area
plt.show()
```

### 9.3 Customizing Legend Appearance
```python
plt.legend(fontsize=10,
          title='Products',
          title_fontsize=11,
          frameon=True,          # box around legend
          shadow=True,
          facecolor='lightyellow',
          edgecolor='gray',
          ncol=2)                # number of columns
plt.show()
```

### 9.4 Setting Labels Without `label=` in plot()
```python
line1, = plt.plot(x, [10,20,15,25,30])
line2, = plt.plot(x, [5,15,10,20,25])
plt.legend([line1, line2], ['Product A', 'Product B'])
plt.show()
```

### 9.5 Legend for Bar Charts and Scatter Plots
```python
plt.bar(['Math','Science'], [85, 90], label='Class A')
plt.bar(['Math','Science'], [78, 82], bottom=[85, 90], label='Class B')
plt.legend()
plt.show()

plt.scatter([1,2,3],[4,5,6], color='blue', label='Group 1')
plt.scatter([2,3,4],[5,6,7], color='red', label='Group 2')
plt.legend()
plt.show()
```

### 9.6 Removing a Legend
```python
plt.legend().remove()
# or simply don't call plt.legend()
```

---

## 10. Subplots (Bonus)

Displaying multiple plots in a single figure — essential for comparing charts side-by-side in EDA.

```python
# Grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))    # 2 rows x 2 columns

axes[0,0].plot([1,2,3],[4,5,6])
axes[0,0].set_title('Line Plot')

axes[0,1].bar(['A','B','C'],[3,7,5])
axes[0,1].set_title('Bar Chart')

axes[1,0].hist(np.random.randn(1000), bins=20)
axes[1,0].set_title('Histogram')

axes[1,1].scatter(np.random.rand(50), np.random.rand(50))
axes[1,1].set_title('Scatter Plot')

plt.tight_layout()          # prevents overlapping titles/labels
plt.show()

# 1 row, multiple columns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, ax in enumerate(axes):
    ax.plot(np.random.randn(50).cumsum())
    ax.set_title(f'Series {i+1}')
plt.show()

# Sharing axes
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10,4))
```

---

## 11. Styling & Saving (Bonus)

### 11.1 Figure Size & Style
```python
plt.figure(figsize=(10, 6))         # width, height in inches
plt.plot([1,2,3],[4,5,6])
plt.show()

# Built-in styles
print(plt.style.available)          # list all styles
plt.style.use('seaborn-v0_8')       # apply a style globally
plt.style.use('ggplot')
plt.style.use('default')            # reset
```

### 11.2 Colors, Colormaps
```python
# Named colors, hex codes, or RGB tuples all work
plt.plot(x, y, color='#FF5733')
plt.plot(x, y, color=(0.2, 0.4, 0.6))

# Common colormaps for heatmaps/scatter: 'viridis', 'plasma', 'coolwarm', 'Blues'
```

### 11.3 Saving Plots
```python
plt.plot([1,2,3],[4,5,6])
plt.savefig('chart.png')                        # save as PNG
plt.savefig('chart.png', dpi=300)                # high resolution
plt.savefig('chart.pdf')                         # vector format for print
plt.savefig('chart.png', bbox_inches='tight')    # avoid cropped labels
plt.show()          # savefig BEFORE show(), or the saved file may be blank
```

### 11.4 Clearing / Closing Figures
```python
plt.clf()       # clear current figure (keep window open)
plt.close()     # close current figure window
plt.close('all')   # close all figures (useful in loops to avoid memory buildup)
```

---

## 12. Practical Mini Project

```python
# ============================================================
# MATPLOTLIB MINI PROJECT: Sales Dashboard
# ============================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

# --- Sample data ---
months = ['Jan','Feb','Mar','Apr','May','Jun']
revenue_2023 = [120, 135, 150, 145, 160, 175]
revenue_2024 = [140, 150, 165, 170, 180, 200]

products = ['Laptop','Mouse','Keyboard','Monitor','Headset']
product_sales = [450, 200, 180, 300, 150]

ages = np.random.normal(35, 12, 500).clip(18, 70)

height = np.random.normal(165, 10, 200)
weight = height * 0.5 + np.random.normal(0, 8, 200)

# --- Build a 2x2 dashboard ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Retail Business Dashboard — 2024', fontsize=18, fontweight='bold')

# 1. Line plot - YoY revenue trend
ax1 = axes[0, 0]
ax1.plot(months, revenue_2023, marker='o', label='2023', color='gray', linestyle='--')
ax1.plot(months, revenue_2024, marker='o', label='2024', color='#1f77b4', linewidth=2.5)
ax1.set_title('Monthly Revenue Trend', fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Revenue (₹ thousands)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Bar chart - product sales
ax2 = axes[0, 1]
bars = ax2.bar(products, product_sales, color='#55A868', edgecolor='black')
ax2.set_title('Product-wise Units Sold', fontweight='bold')
ax2.set_xlabel('Product')
ax2.set_ylabel('Units Sold')
ax2.tick_params(axis='x', rotation=30)
for bar in bars:
    h = bar.get_height()
    ax2.text(bar.get_x()+bar.get_width()/2, h+5, str(h), ha='center', fontsize=9)

# 3. Histogram - customer age distribution
ax3 = axes[1, 0]
ax3.hist(ages, bins=20, color='#DD8452', edgecolor='black', alpha=0.8)
ax3.set_title('Customer Age Distribution', fontweight='bold')
ax3.set_xlabel('Age')
ax3.set_ylabel('Number of Customers')
ax3.axvline(ages.mean(), color='red', linestyle='--', label=f'Mean: {ages.mean():.1f}')
ax3.legend()

# 4. Scatter plot - height vs weight with trend line
ax4 = axes[1, 1]
ax4.scatter(height, weight, alpha=0.5, color='#C44E52', edgecolors='w', s=40)
z = np.polyfit(height, weight, 1)
p = np.poly1d(z)
x_line = np.linspace(height.min(), height.max(), 100)
ax4.plot(x_line, p(x_line), 'k--', linewidth=2, label='Trend')
ax4.set_title('Height vs Weight', fontweight='bold')
ax4.set_xlabel('Height (cm)')
ax4.set_ylabel('Weight (kg)')
ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])       # leave room for suptitle
plt.savefig('sales_dashboard.png', dpi=200, bbox_inches='tight')
plt.show()

print("✅ Dashboard created and saved as sales_dashboard.png")
```

---

## 13. Cheatsheet

### Setup
```python
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10,6))
```

### Plot Types
```python
plt.plot(x, y)                # line
plt.bar(x, y)   plt.barh(x,y) # bar (vertical/horizontal)
plt.hist(data, bins=20)       # histogram
plt.scatter(x, y)             # scatter
plt.pie(values, labels=labels)  # pie
plt.boxplot(data)             # box plot
```

### Styling
```python
color='blue'  linestyle='--'  linewidth=2  marker='o'  alpha=0.7
```

### Labels, Title, Legend
```python
plt.xlabel('X'); plt.ylabel('Y')
plt.title('My Chart')
plt.legend(loc='best')
plt.grid(True)
plt.xticks(rotation=45)
```

### Multi-plot & Output
```python
fig, axes = plt.subplots(rows, cols, figsize=(w,h))
plt.tight_layout()
plt.savefig('name.png', dpi=300)
plt.show()
```

### Chart Selection Guide
| Data / Goal | Chart |
|---|---|
| Trend over time | **Line plot** |
| Compare categories | **Bar chart** |
| Distribution of one numeric variable | **Histogram** |
| Relationship between two numeric variables | **Scatter plot** |
| Parts of a whole | Pie chart |
| Spread/outliers of a numeric variable | Box plot |

### Key Rules to Remember 🏆
1. Always call `plt.show()` at the end (or your plot won't render outside Jupyter).
2. Call `plt.savefig()` **before** `plt.show()` — after `show()`, the figure is cleared.
3. Use `label=` in each plot call + `plt.legend()` to identify multiple series.
4. `plt.tight_layout()` fixes overlapping titles/labels in subplots.
5. Histograms show **one** numeric variable's distribution; scatter plots show the relationship between **two**.
6. Set `alpha < 1` when points/bars overlap, to see density.
7. Set `np.random.seed()` when generating example/demo data for reproducibility.

---

## 📌 Summary

| Topic | Key Point |
|---|---|
| **Line plots** | `plt.plot()` — trends over continuous/time data |
| **Bar charts** | `plt.bar()` / `plt.barh()` — comparing categories |
| **Histograms** | `plt.hist()` — distribution of one numeric variable |
| **Scatter plots** | `plt.scatter()` — relationship between two numeric variables |
| **Labels** | `plt.xlabel()`, `plt.ylabel()` — always describe your axes |
| **Titles** | `plt.title()` — summarize what the chart shows |
| **Legends** | `plt.legend()` — required whenever there's more than one series |

---

*📁 Previous: Part 5 → Pandas*
*📁 Next: Part 7 → Seaborn for Statistical Visualization*

---
### 🔖 Tags
`#Matplotlib` `#Python` `#DataVisualization` `#DataScience` `#Notes`
