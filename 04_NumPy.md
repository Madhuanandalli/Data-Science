# 🔢 Data Science Notes — Part 4: NumPy

> **Num**erical **Py**thon — the foundation library for numerical computing and the base on which Pandas, Scikit-learn, TensorFlow, and SciPy are built.

---

## Table of Contents
1. [Introduction to NumPy](#1-introduction-to-numpy)
2. [Arrays](#2-arrays)
3. [Dimensions](#3-dimensions)
4. [shape](#4-shape)
5. [ndim](#5-ndim)
6. [Array Attributes Summary](#6-array-attributes-summary)
7. [Indexing](#7-indexing)
8. [Slicing](#8-slicing)
9. [Reshaping](#9-reshaping)
10. [Mathematical Operations](#10-mathematical-operations)
11. [Statistical Functions](#11-statistical-functions)
12. [Mean, Median, Standard Deviation](#12-mean-median-standard-deviation)
13. [Practical Mini Project](#13-practical-mini-project)
14. [Cheatsheet](#14-cheatsheet)

---

## 1. Introduction to NumPy

**NumPy** is a Python library that provides a fast, memory-efficient multi-dimensional array object called **ndarray**, along with mathematical functions to operate on these arrays.

### Why NumPy over Python Lists?
| Feature | Python List | NumPy Array |
|---|---|---|
| **Speed** | Slow (interpreted loops) | Fast (C-implemented, vectorized) |
| **Memory** | High (stores pointers + objects) | Low (contiguous block) |
| **Data type** | Mixed types allowed | Homogeneous (single dtype) |
| **Operations** | Element-wise needs a loop | Vectorized — no loop needed |
| **Functionality** | Basic | Linear algebra, FFT, random, stats |

```python
# Speed comparison
import numpy as np
import time

lst = list(range(1_000_000))
arr = np.arange(1_000_000)

start = time.time()
lst_result = [x * 2 for x in lst]
print(f"List: {time.time()-start:.4f}s")

start = time.time()
arr_result = arr * 2                    # vectorized - no loop!
print(f"NumPy: {time.time()-start:.4f}s")
# NumPy is typically 10-50x faster
```

### Installation & Import
```bash
pip install numpy
```
```python
import numpy as np        # 'np' is the universal convention
print(np.__version__)
```

---

## 2. Arrays

An **ndarray** (N-dimensional array) is a grid of values, **all of the same data type**, indexed by a tuple of non-negative integers.

### 2.1 Creating Arrays from Python Objects
```python
import numpy as np

# 1-D array (Vector)
a = np.array([1, 2, 3, 4, 5])
print(a)            # [1 2 3 4 5]
print(type(a))      # <class 'numpy.ndarray'>

# 2-D array (Matrix)
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b)
# [[1 2 3]
#  [4 5 6]]

# 3-D array (Tensor)
c = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])

# From a tuple
d = np.array((1, 2, 3))

# Specify data type
e = np.array([1, 2, 3], dtype=float)     # [1. 2. 3.]
f = np.array([1.7, 2.9], dtype=int)      # [1 2]  (truncated)
```

### 2.2 Creating Arrays with Built-in Functions
```python
# Zeros and Ones
np.zeros(5)                 # [0. 0. 0. 0. 0.]
np.zeros((2, 3))            # 2x3 matrix of zeros
np.ones((3, 3))             # 3x3 matrix of ones
np.ones((2,2), dtype=int)   # integer ones

# Full - filled with a constant
np.full((2, 3), 7)          # 2x3 filled with 7

# Empty - uninitialized (fast, contains garbage values)
np.empty((2, 2))

# arange - like range() but returns an array
np.arange(5)                # [0 1 2 3 4]
np.arange(2, 10, 2)         # [2 4 6 8]  (start, stop, step)
np.arange(0, 1, 0.25)       # [0. 0.25 0.5 0.75]  works with floats

# linspace - N evenly spaced values between start and stop (INCLUSIVE)
np.linspace(0, 10, 5)       # [ 0.   2.5  5.   7.5 10. ]
np.linspace(0, 1, 11)       # 11 values from 0 to 1

# Identity matrix
np.eye(3)                   # 3x3 identity matrix
np.identity(3)              # same thing

# Diagonal matrix
np.diag([1, 2, 3])          # diagonal matrix from a list

# Like-arrays (same shape as another array)
np.zeros_like(b)            # zeros with b's shape
np.ones_like(b)
np.full_like(b, 9)
```

### 2.3 Random Arrays
```python
np.random.seed(42)          # for reproducibility - ALWAYS set in DS

np.random.rand(3)           # 3 random floats in [0,1)
np.random.rand(2, 3)        # 2x3 uniform random

np.random.randn(3)          # standard normal (mean=0, std=1)
np.random.randn(2, 3)

np.random.randint(1, 100, 5)        # 5 random ints from 1 to 99
np.random.randint(0, 10, (3, 3))    # 3x3 random ints

np.random.normal(loc=50, scale=10, size=1000)   # normal dist: mean=50, std=10
np.random.uniform(1, 10, 5)                      # uniform distribution

np.random.choice([1,2,3,4,5], size=3)            # random sample
np.random.choice([1,2,3], size=5, p=[0.5,0.3,0.2])  # weighted

arr = np.array([1,2,3,4,5])
np.random.shuffle(arr)      # shuffles IN PLACE
print(np.random.permutation([1,2,3,4,5]))   # returns a shuffled COPY
```

### 2.4 Data Types (dtype)
| dtype | Description |
|---|---|
| `int8/16/32/64` | Integers of varying size |
| `float16/32/64` | Floating point |
| `bool` | True / False |
| `complex64/128` | Complex numbers |
| `str_` / `object` | Strings / Python objects |

```python
a = np.array([1, 2, 3])
print(a.dtype)              # int64

b = np.array([1.0, 2.0])
print(b.dtype)              # float64

# Casting
c = a.astype(float)         # [1. 2. 3.]
d = b.astype(int)           # [1 2]
e = a.astype(bool)          # [True True True]

# Memory optimization (important for big datasets)
big = np.arange(1000000, dtype=np.int64)
small = big.astype(np.int32)
print(big.nbytes, "→", small.nbytes)     # halves memory usage
```

---

## 3. Dimensions

A **dimension (axis)** is a direction along which data is arranged. NumPy arrays can have any number of dimensions.

```
0-D (Scalar)          5

1-D (Vector)          [1, 2, 3]
                       ← axis 0 →

2-D (Matrix)          [[1, 2, 3],        ↑
                       [4, 5, 6]]      axis 0 (rows)
                       ← axis 1 →        ↓
                        (columns)

3-D (Tensor)          [[[1,2],[3,4]],
                       [[5,6],[7,8]]]
                      axis 0 = blocks/depth
                      axis 1 = rows
                      axis 2 = columns
```

### Code Examples
```python
import numpy as np

# 0-D : Scalar
d0 = np.array(42)
print(d0, d0.ndim)          # 42  0

# 1-D : Vector
d1 = np.array([1, 2, 3, 4])
print(d1.ndim)              # 1

# 2-D : Matrix
d2 = np.array([[1,2,3],[4,5,6]])
print(d2.ndim)              # 2

# 3-D : Tensor
d3 = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
print(d3.ndim)              # 3

# Force minimum dimensions
d = np.array([1,2,3], ndmin=5)
print(d.ndim)               # 5
print(d.shape)              # (1,1,1,1,3)
```

### Understanding Axis (Critical for DS!)
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# axis=0 → operate DOWN the columns (collapse rows)
print(arr.sum(axis=0))      # [5 7 9]     column sums

# axis=1 → operate ACROSS the rows (collapse columns)
print(arr.sum(axis=1))      # [6 15]      row sums

# No axis → entire array
print(arr.sum())            # 21
```

```
        axis=1 →  (across columns)
       ┌─────────────────┐
axis=0 │  1    2    3    │  → row sum = 6
  ↓    │  4    5    6    │  → row sum = 15
       └─────────────────┘
          ↓    ↓    ↓
          5    7    9      column sums
```

> 💡 **Memory trick:** `axis=0` = **down** (row-wise collapse, gives per-column result). `axis=1` = **across** (column-wise collapse, gives per-row result).

### Real-World Meaning of Dimensions
| Dim | Example in Data Science |
|---|---|
| **1-D** | A single feature column: `[25, 30, 35]` |
| **2-D** | A dataset: rows = samples, columns = features |
| **3-D** | Color image: (height, width, RGB channels) |
| **4-D** | Batch of images: (batch, height, width, channels) |

---

## 4. `shape`

`.shape` returns a **tuple** giving the size of the array along each dimension.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a.shape)              # (5,)        → 5 elements, 1-D

b = np.array([[1,2,3], [4,5,6]])
print(b.shape)              # (2, 3)      → 2 rows, 3 columns

c = np.array([[[1,2],[3,4]], [[5,6],[7,8]]])
print(c.shape)              # (2, 2, 2)   → 2 blocks, 2 rows, 2 cols

d = np.zeros((4, 5, 3))
print(d.shape)              # (4, 5, 3)
```

### Reading a shape tuple
```
shape = (2, 3)
          │  └── number of COLUMNS (axis 1)
          └───── number of ROWS    (axis 0)

shape = (100, 28, 28, 3)
          │    │   │   └── channels (RGB)
          │    │   └────── width
          │    └────────── height
          └─────────────── number of images
```

### Practical Uses
```python
arr = np.array([[1,2,3], [4,5,6], [7,8,9]])

rows, cols = arr.shape
print(f"Rows: {rows}, Columns: {cols}")

# Shape of each dimension
print(arr.shape[0])         # 3 (rows)
print(arr.shape[1])         # 3 (columns)

# Changing shape directly (must match total elements)
a = np.arange(12)
a.shape = (3, 4)            # modifies in place
print(a)

# ⚠️ Common DS error - shape mismatch
X = np.array([[1,2],[3,4],[5,6]])    # (3,2)
y = np.array([1, 0, 1])              # (3,)
print(X.shape, y.shape)              # must align for ML models

# Converting 1-D to 2-D column vector (needed by sklearn)
y_2d = y.reshape(-1, 1)
print(y_2d.shape)                    # (3, 1)
```

---

## 5. `ndim`

`.ndim` returns the **number of dimensions (axes)** as an integer.

```python
import numpy as np

print(np.array(5).ndim)                       # 0
print(np.array([1,2,3]).ndim)                 # 1
print(np.array([[1,2],[3,4]]).ndim)           # 2
print(np.array([[[1],[2]],[[3],[4]]]).ndim)   # 3

# ndim always equals len(shape)
arr = np.zeros((2,3,4))
print(arr.ndim)             # 3
print(len(arr.shape))       # 3  - same thing
```

### Practical Use — Validation
```python
def process(data):
    arr = np.array(data)
    if arr.ndim == 1:
        print("1-D: treating as a single feature")
        arr = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        print(f"2-D: {arr.shape[0]} samples, {arr.shape[1]} features")
    else:
        raise ValueError(f"Expected 1-D or 2-D, got {arr.ndim}-D")
    return arr

process([1,2,3])
process([[1,2],[3,4]])
```

---

## 6. Array Attributes Summary

```python
import numpy as np
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(arr.shape)      # (2, 3)        dimensions as tuple
print(arr.ndim)       # 2             number of dimensions
print(arr.size)       # 6             TOTAL number of elements
print(arr.dtype)      # int64         data type
print(arr.itemsize)   # 8             bytes per element
print(arr.nbytes)     # 48            total bytes (size × itemsize)
print(arr.T)          # transpose
print(len(arr))       # 2             length of first axis only
```

| Attribute | Returns | Example output |
|---|---|---|
| `.shape` | Tuple of dimensions | `(2, 3)` |
| `.ndim` | Number of dimensions | `2` |
| `.size` | Total elements | `6` |
| `.dtype` | Data type | `int64` |
| `.itemsize` | Bytes per element | `8` |
| `.nbytes` | Total memory | `48` |
| `.T` | Transposed array | shape becomes `(3,2)` |

---

## 7. Indexing

Accessing individual elements. **Indexing starts at 0.**

### 7.1 1-D Indexing
```python
a = np.array([10, 20, 30, 40, 50])

print(a[0])       # 10   first
print(a[3])       # 40
print(a[-1])      # 50   last
print(a[-2])      # 40   second last

a[0] = 99         # modify
print(a)          # [99 20 30 40 50]
```

### 7.2 2-D Indexing
```python
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Syntax: arr[row, column]
print(b[0, 0])    # 1
print(b[1, 2])    # 6
print(b[-1, -1])  # 9

print(b[0])       # [1 2 3]   entire first row
print(b[:, 0])    # [1 4 7]   entire first column

# Old style (also works but slower)
print(b[1][2])    # 6
```

```
        col0  col1  col2
row0  [  1     2     3  ]     b[0,1] = 2
row1  [  4     5     6  ]     b[1,2] = 6
row2  [  7     8     9  ]     b[2,0] = 7
```

### 7.3 3-D Indexing
```python
c = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])

# Syntax: arr[block, row, column]
print(c[0, 1, 1])     # 4
print(c[1, 0, 0])     # 5
print(c[0])           # [[1 2] [3 4]]  first block
```

### 7.4 Fancy (Array) Indexing
```python
a = np.array([10, 20, 30, 40, 50])

# Pass a list of indices
print(a[[0, 2, 4]])       # [10 30 50]
print(a[[-1, -2]])        # [50 40]

b = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(b[[0, 2]])          # rows 0 and 2
print(b[[0, 1], [1, 2]])  # [2 6]  elements at (0,1) and (1,2)
```

### 7.5 Boolean Indexing / Masking (Very Important in DS!)
```python
a = np.array([10, 25, 3, 47, 18, 62])

# Create a boolean mask
mask = a > 20
print(mask)               # [False  True False  True False  True]

# Filter using the mask
print(a[mask])            # [25 47 62]
print(a[a > 20])          # same, in one line

# Multiple conditions (use & | ~, NOT and/or/not)
print(a[(a > 10) & (a < 50)])      # [25 47 18]
print(a[(a < 10) | (a > 50)])      # [ 3 62]
print(a[~(a > 20)])                # [10  3 18]  NOT

# Conditional replacement
a[a > 50] = 50            # cap values
print(a)

# np.where - vectorized if-else
scores = np.array([45, 88, 72, 30, 95])
result = np.where(scores >= 50, "Pass", "Fail")
print(result)             # ['Fail' 'Pass' 'Pass' 'Fail' 'Pass']

# Get indices where condition is True
print(np.where(scores >= 50))      # (array([1, 2, 4]),)

# Count matching elements
print(np.sum(scores >= 50))        # 3
print(np.any(scores > 90))         # True
print(np.all(scores > 20))         # True
```

---

## 8. Slicing

Extracting a **sub-array**. Syntax: `arr[start : stop : step]` — **stop is EXCLUSIVE**.

### 8.1 1-D Slicing
```python
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(a[2:5])       # [2 3 4]      index 2 to 4
print(a[:4])        # [0 1 2 3]    start to index 3
print(a[6:])        # [6 7 8 9]    index 6 to end
print(a[:])         # entire array
print(a[::2])       # [0 2 4 6 8]  every 2nd element
print(a[1::2])      # [1 3 5 7 9]  odd indices
print(a[::-1])      # [9 8 ... 0]  reversed
print(a[-3:])       # [7 8 9]      last 3 elements
print(a[:-3])       # [0..6]       all except last 3
```

### 8.2 2-D Slicing
```python
b = np.array([[ 1,  2,  3,  4],
              [ 5,  6,  7,  8],
              [ 9, 10, 11, 12],
              [13, 14, 15, 16]])

# Syntax: arr[row_slice, column_slice]
print(b[0:2, 0:2])      # [[1 2] [5 6]]     top-left 2x2 block
print(b[:, 1])          # [2 6 10 14]       column 1 (as 1-D)
print(b[:, 1:2])        # column 1 as a 2-D column vector
print(b[1, :])          # [5 6 7 8]         row 1
print(b[1:3, :])        # rows 1 and 2
print(b[:, ::2])        # every 2nd column
print(b[::-1, :])       # rows reversed
print(b[-2:, -2:])      # bottom-right 2x2 block
```

### 8.3 3-D Slicing
```python
c = np.arange(24).reshape(2, 3, 4)

print(c[0])             # first block (3x4)
print(c[0, 1])          # second row of first block
print(c[:, 0, :])       # first row of every block
print(c[..., 0])        # ellipsis: first column of everything
```

### 8.4 ⚠️ Views vs Copies (CRITICAL)
```python
a = np.array([1, 2, 3, 4, 5])

# Slicing returns a VIEW - shares memory with the original!
view = a[1:4]
view[0] = 99
print(a)            # [ 1 99  3  4  5]  ← original CHANGED!

# Use .copy() to get an independent array
a = np.array([1, 2, 3, 4, 5])
copy = a[1:4].copy()
copy[0] = 99
print(a)            # [1 2 3 4 5]  ← original unchanged ✅

# Check if it's a view
print(view.base is not None)     # True  = it's a view
print(copy.base is None)         # True  = it's a copy
```

> 💡 **Note:** This differs from Python lists, where slicing always copies. NumPy avoids copying for performance.

---

## 9. Reshaping

Changing the shape of an array **without changing its data**. The total number of elements must stay the same.

### 9.1 reshape()
```python
a = np.arange(12)         # [0 1 2 ... 11], shape (12,)

print(a.reshape(3, 4))    # 3 rows, 4 columns
print(a.reshape(4, 3))    # 4 rows, 3 columns
print(a.reshape(2, 6))
print(a.reshape(2, 3, 2)) # 3-D

# ❌ a.reshape(5, 3)      # ValueError: 12 elements ≠ 15
```

### 9.2 The magic `-1` (auto-calculate)
```python
a = np.arange(12)

print(a.reshape(3, -1).shape)     # (3, 4)   NumPy computes 4
print(a.reshape(-1, 6).shape)     # (2, 6)
print(a.reshape(2, -1, 2).shape)  # (2, 3, 2)

# Very common in ML - convert 1-D to a column vector
y = np.array([1, 2, 3, 4])
print(y.reshape(-1, 1))
# [[1]
#  [2]
#  [3]
#  [4]]
print(y.reshape(1, -1))           # row vector: [[1 2 3 4]]
```

### 9.3 Flattening (N-D → 1-D)
```python
b = np.array([[1, 2, 3], [4, 5, 6]])

print(b.flatten())      # [1 2 3 4 5 6]   returns a COPY
print(b.ravel())        # [1 2 3 4 5 6]   returns a VIEW (faster)
print(b.reshape(-1))    # [1 2 3 4 5 6]

# Difference demo
flat = b.flatten()
flat[0] = 99
print(b[0,0])           # 1   ← unchanged (copy)

rav = b.ravel()
rav[0] = 99
print(b[0,0])           # 99  ← changed (view)
```

### 9.4 Transpose
```python
b = np.array([[1, 2, 3],
              [4, 5, 6]])          # shape (2,3)

print(b.T)                         # shape (3,2)
# [[1 4]
#  [2 5]
#  [3 6]]

print(np.transpose(b))             # same
print(b.swapaxes(0, 1))            # same for 2-D

# 3-D transpose with explicit axis order
c = np.arange(24).reshape(2,3,4)
print(c.transpose(1, 0, 2).shape)  # (3, 2, 4)
```

### 9.5 Adding / Removing Dimensions
```python
a = np.array([1, 2, 3])       # shape (3,)

# Add a dimension
print(np.expand_dims(a, axis=0).shape)   # (1, 3)
print(np.expand_dims(a, axis=1).shape)   # (3, 1)
print(a[np.newaxis, :].shape)            # (1, 3)
print(a[:, np.newaxis].shape)            # (3, 1)

# Remove dimensions of size 1
b = np.array([[[1], [2], [3]]])          # shape (1,3,1)
print(np.squeeze(b).shape)               # (3,)
```

### 9.6 Joining & Splitting Arrays
```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Joining
print(np.concatenate((a, b), axis=0))    # vertical (4x2)
print(np.concatenate((a, b), axis=1))    # horizontal (2x4)
print(np.vstack((a, b)))                 # vertical stack
print(np.hstack((a, b)))                 # horizontal stack
print(np.stack((a, b)).shape)            # (2,2,2) new axis

# Splitting
c = np.arange(12).reshape(4, 3)
print(np.split(c, 2, axis=0))            # split into 2 along rows
print(np.hsplit(c, 3))                   # split into 3 columns
print(np.vsplit(c, 2))                   # split into 2 row groups

# Add/remove elements
d = np.array([1,2,3])
print(np.append(d, [4,5]))               # [1 2 3 4 5]
print(np.insert(d, 1, 99))               # [1 99 2 3]
print(np.delete(d, 0))                   # [2 3]
```

---

## 10. Mathematical Operations

### 10.1 Element-wise Arithmetic (Vectorization)
```python
import numpy as np

a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])

print(a + b)      # [11 22 33 44]
print(a - b)      # [ 9 18 27 36]
print(a * b)      # [10 40 90 160]   element-wise, NOT matrix mult
print(a / b)      # [10. 10. 10. 10.]
print(a // b)     # floor division
print(a % b)      # modulus
print(a ** 2)     # [100 400 900 1600]

# With a scalar (broadcasting)
print(a + 5)      # [15 25 35 45]
print(a * 2)      # [20 40 60 80]

# Equivalent function forms
print(np.add(a, b))
print(np.subtract(a, b))
print(np.multiply(a, b))
print(np.divide(a, b))
print(np.power(a, 2))
print(np.mod(a, b))
```

### 10.2 Broadcasting
NumPy automatically expands smaller arrays to match shapes during operations.

**Broadcasting rules:** compare shapes from the right; dimensions are compatible if they are equal or one of them is 1.

```python
# Scalar with array
a = np.array([1, 2, 3])
print(a + 10)              # [11 12 13]

# (3,1) with (3,)  →  (3,3)
b = np.array([[1], [2], [3]])      # shape (3,1)
c = np.array([10, 20, 30])         # shape (3,)
print(b + c)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]

# Real DS use - normalize each column
data = np.array([[1, 200], [2, 300], [3, 400]])
col_means = data.mean(axis=0)               # [2. 300.]
col_stds  = data.std(axis=0)
normalized = (data - col_means) / col_stds  # broadcasting!
print(normalized)
```

```
Shape (3,1)     Shape (3,)        Result (3,3)
  [1]             [10 20 30]       [[11 21 31]
  [2]        +               =      [12 22 32]
  [3]                                [13 23 33]]
```

### 10.3 Mathematical Functions
```python
a = np.array([1, 4, 9, 16, 25])

print(np.sqrt(a))          # [1. 2. 3. 4. 5.]
print(np.square(a))
print(np.abs([-1, -2, 3])) # [1 2 3]
print(np.exp([0, 1, 2]))   # e^x
print(np.log(a))           # natural log
print(np.log10(a))         # base-10 log
print(np.log2(a))          # base-2 log
print(np.log1p(a))         # log(1+x) - safe for zeros, used in DS

# Rounding
b = np.array([1.234, 5.678, 9.999])
print(np.round(b, 2))      # [1.23 5.68 10.  ]
print(np.floor(b))         # [1. 5. 9.]
print(np.ceil(b))          # [2. 6. 10.]
print(np.trunc(b))         # [1. 5. 9.]

# Trigonometric
angles = np.array([0, np.pi/2, np.pi])
print(np.sin(angles))
print(np.cos(angles))
print(np.degrees(angles))
print(np.radians([0, 90, 180]))

# Sign & clip
print(np.sign([-5, 0, 5]))         # [-1  0  1]
print(np.clip([1, 50, 200], 0, 100))  # [  1  50 100]  cap values
```

### 10.4 Aggregate Functions
```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(np.sum(a))             # 21    all elements
print(np.sum(a, axis=0))     # [5 7 9]   column sums
print(np.sum(a, axis=1))     # [6 15]    row sums

print(np.prod(a))            # 720   product
print(np.cumsum(a))          # [1 3 6 10 15 21]  cumulative sum
print(np.cumprod([1,2,3,4])) # [1 2 6 24]

print(np.min(a), np.max(a))  # 1 6
print(np.argmin(a), np.argmax(a))   # 0 5  (index of min/max)
print(np.argmax(a, axis=1))  # [2 2]  index of max in each row

print(np.ptp(a))             # 5   peak-to-peak (max - min) = range
```

### 10.5 Matrix Operations (Linear Algebra)
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(A * B)            # element-wise multiplication
print(A @ B)            # MATRIX multiplication (preferred)
print(np.dot(A, B))     # same as @
print(np.matmul(A, B))  # same as @

print(A.T)                       # transpose
print(np.linalg.inv(A))          # inverse
print(np.linalg.det(A))          # determinant
print(np.trace(A))               # sum of diagonal
print(np.linalg.matrix_rank(A))  # rank

# Eigenvalues & eigenvectors (used in PCA)
values, vectors = np.linalg.eig(A)
print(values)

# Solve linear system Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(x)

# Norms
v = np.array([3, 4])
print(np.linalg.norm(v))         # 5.0  (Euclidean/L2 norm)
print(np.linalg.norm(v, ord=1))  # 7.0  (L1/Manhattan norm)
```

### 10.6 Comparison Operations
```python
a = np.array([1, 2, 3, 4])
b = np.array([1, 5, 3, 7])

print(a == b)               # [ True False  True False]
print(a > b)                # [False False False False]
print(np.array_equal(a, b)) # False  (entire array comparison)

print(np.maximum(a, b))     # [1 5 3 7]  element-wise max
print(np.minimum(a, b))     # [1 2 3 4]
```

### 10.7 Sorting & Searching
```python
a = np.array([3, 1, 4, 1, 5, 9, 2])

print(np.sort(a))           # [1 1 2 3 4 5 9]  returns a copy
print(np.argsort(a))        # indices that would sort the array
a.sort()                    # sorts IN PLACE

b = np.array([[3,1],[2,4]])
print(np.sort(b, axis=0))   # sort each column
print(np.sort(b, axis=1))   # sort each row

print(np.unique([1,1,2,2,3]))               # [1 2 3]
print(np.unique([1,1,2], return_counts=True))  # (array([1,2]), array([2,1]))

print(np.searchsorted([1,3,5,7], 4))        # 2 (insertion index)
print(np.isin([1,2,3], [2,3,4]))            # [False True True]
```

---

## 11. Statistical Functions

NumPy provides a full set of statistical functions — essential for EDA.

### Overview Table
| Function | Description |
|---|---|
| `np.mean()` | Arithmetic average |
| `np.median()` | Middle value |
| `np.std()` | Standard deviation |
| `np.var()` | Variance |
| `np.min()` / `np.max()` | Minimum / Maximum |
| `np.ptp()` | Range (max − min) |
| `np.percentile()` | Value at given percentile |
| `np.quantile()` | Value at given quantile (0–1) |
| `np.corrcoef()` | Correlation coefficient matrix |
| `np.cov()` | Covariance matrix |
| `np.average()` | Weighted average |
| `np.nanmean()` etc. | Ignore NaN values |
| `np.histogram()` | Frequency distribution |
| `np.bincount()` | Count occurrences of ints |

### 11.1 Basic Statistics
```python
import numpy as np

data = np.array([12, 15, 18, 22, 25, 28, 30, 35, 40, 100])

print("Count :", data.size)               # 10
print("Sum   :", np.sum(data))            # 325
print("Min   :", np.min(data))            # 12
print("Max   :", np.max(data))            # 100
print("Range :", np.ptp(data))            # 88
print("Mean  :", np.mean(data))           # 32.5
print("Median:", np.median(data))         # 26.5
print("Std   :", np.std(data))            # 24.32
print("Var   :", np.var(data))            # 591.45
```

### 11.2 Percentiles & Quantiles
```python
data = np.array([12, 15, 18, 22, 25, 28, 30, 35, 40, 100])

print(np.percentile(data, 25))      # Q1 (1st quartile)
print(np.percentile(data, 50))      # Q2 = median
print(np.percentile(data, 75))      # Q3 (3rd quartile)
print(np.percentile(data, [25, 50, 75]))

# Quantiles (0 to 1 instead of 0 to 100)
print(np.quantile(data, 0.25))
print(np.quantile(data, [0.1, 0.5, 0.9]))

# IQR & outlier detection (standard DS technique)
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = data[(data < lower) | (data > upper)]
print(f"IQR: {IQR}, Bounds: [{lower}, {upper}]")
print("Outliers:", outliers)         # [100]
```

### 11.3 Correlation & Covariance
```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

print(np.corrcoef(x, y))
# [[1.         0.7745967]
#  [0.7745967  1.       ]]
print("Correlation:", np.corrcoef(x, y)[0, 1])   # 0.7746

print(np.cov(x, y))        # covariance matrix
```

| Correlation value | Interpretation |
|---|---|
| `+1` | Perfect positive relationship |
| `+0.7 to +0.9` | Strong positive |
| `0` | No linear relationship |
| `-0.7 to -0.9` | Strong negative |
| `-1` | Perfect negative relationship |

### 11.4 Handling NaN (Missing Values)
```python
data = np.array([10, 20, np.nan, 40, 50])

print(np.mean(data))          # nan  ❌ normal functions fail
print(np.nanmean(data))       # 30.0 ✅ ignores NaN
print(np.nanmedian(data))     # 30.0
print(np.nanstd(data))        # ignores NaN
print(np.nansum(data))        # 120.0
print(np.nanmin(data), np.nanmax(data))

# Detect and handle NaN
print(np.isnan(data))                 # [F F T F F]
print(np.sum(np.isnan(data)))         # 1  count of NaN
clean = data[~np.isnan(data)]         # remove NaN
filled = np.nan_to_num(data, nan=0)   # replace NaN with 0
```

### 11.5 Weighted Average & Histogram
```python
# Weighted average
marks   = np.array([80, 90, 70])
weights = np.array([0.5, 0.3, 0.2])
print(np.average(marks, weights=weights))     # 81.0

# Histogram - frequency distribution
data = np.random.normal(50, 10, 1000)
counts, bin_edges = np.histogram(data, bins=5)
print("Counts:", counts)
print("Bin edges:", bin_edges.round(2))

# Counting occurrences of integers
labels = np.array([0, 1, 1, 2, 2, 2])
print(np.bincount(labels))     # [1 2 3]  → class distribution
```

### 11.6 Axis-wise Statistics on a Dataset
```python
# Simulating a dataset: rows = students, columns = subjects
marks = np.array([[85, 90, 78],
                  [70, 88, 92],
                  [95, 65, 80],
                  [60, 75, 85]])

print("Subject-wise averages (axis=0):", marks.mean(axis=0))
# [77.5  79.5  83.75]

print("Student-wise averages (axis=1):", marks.mean(axis=1))
# [84.33  83.33  80.  73.33]

print("Overall average:", marks.mean())
print("Subject-wise std:", marks.std(axis=0).round(2))
print("Best student index:", np.argmax(marks.mean(axis=1)))
print("Hardest subject index:", np.argmin(marks.mean(axis=0)))
```

---

## 12. Mean, Median, Standard Deviation

These three are the **core measures** used in every EDA.

### 12.1 Mean (Arithmetic Average)

**Formula:**
```
        Σxᵢ      x₁ + x₂ + ... + xₙ
mean = ───── = ────────────────────
          n              n
```

```python
import numpy as np

data = np.array([10, 20, 30, 40, 50])

print(np.mean(data))       # 30.0
print(data.mean())         # 30.0  (method form)

# Manual verification
print(np.sum(data) / len(data))    # 30.0

# 2-D mean
m = np.array([[1,2,3],[4,5,6]])
print(m.mean())            # 3.5     all elements
print(m.mean(axis=0))      # [2.5 3.5 4.5]   column means
print(m.mean(axis=1))      # [2. 5.]         row means

# Mean ignoring NaN
d = np.array([10, 20, np.nan, 40])
print(np.nanmean(d))       # 23.33
```

**⚠️ Weakness — mean is sensitive to outliers:**
```python
normal   = np.array([10, 20, 30, 40, 50])
with_out = np.array([10, 20, 30, 40, 1000])

print(np.mean(normal))     # 30.0
print(np.mean(with_out))   # 220.0  ← badly distorted!
```

### 12.2 Median (Middle Value)

The middle value when the data is **sorted**.
- Odd count → the middle element
- Even count → average of the two middle elements

```python
# Odd number of elements
a = np.array([10, 20, 30, 40, 50])
print(np.median(a))        # 30.0  (middle value)

# Even number of elements
b = np.array([10, 20, 30, 40])
print(np.median(b))        # 25.0  → (20+30)/2

# Unsorted data works fine (NumPy sorts internally)
c = np.array([50, 10, 30, 20, 40])
print(np.median(c))        # 30.0

# 2-D
m = np.array([[1,2,3],[4,5,6]])
print(np.median(m, axis=0))    # [2.5 3.5 4.5]
print(np.median(m, axis=1))    # [2. 5.]

# NaN-safe
print(np.nanmedian([10, 20, np.nan, 40]))
```

**✅ Strength — median resists outliers:**
```python
normal   = np.array([10, 20, 30, 40, 50])
with_out = np.array([10, 20, 30, 40, 1000])

print(np.median(normal))    # 30.0
print(np.median(with_out))  # 30.0  ← unchanged! Robust ✅
```

### 12.3 Standard Deviation (Spread)

Measures how **spread out** the data is from the mean.

**Formula (Population):**
```
        ┌─────────────────
        │  Σ(xᵢ − μ)²
σ  =   │  ────────────
      √        n
```

**Formula (Sample):** divides by `n−1` instead of `n` (Bessel's correction).

```python
data = np.array([10, 20, 30, 40, 50])

print(np.std(data))          # 14.14  POPULATION std (ddof=0, default)
print(np.std(data, ddof=1))  # 15.81  SAMPLE std (ddof=1)

print(np.var(data))          # 200.0  variance = std²
print(np.std(data) ** 2)     # 200.0

# 2-D
m = np.array([[1,2,3],[4,5,6]])
print(m.std(axis=0))         # [1.5 1.5 1.5]  column std
print(m.std(axis=1))         # [0.816 0.816]  row std

# NaN-safe
print(np.nanstd([10, 20, np.nan, 40]))
```

> 💡 **ddof rule:** Use `ddof=0` (default) when your data is the **entire population**. Use `ddof=1` when it's a **sample** from a larger population. Pandas defaults to `ddof=1`, NumPy defaults to `ddof=0` — a classic source of mismatched results.

**Interpreting standard deviation:**
```python
tight = np.array([48, 49, 50, 51, 52])     # values close together
loose = np.array([10, 30, 50, 70, 90])     # values spread out

print(f"Tight: mean={tight.mean()}, std={tight.std():.2f}")   # std ≈ 1.41
print(f"Loose: mean={loose.mean()}, std={loose.std():.2f}")   # std ≈ 28.28
# Same mean (50), completely different spread!
```

### 12.4 Mean vs Median — when to use which

| Situation | Use |
|---|---|
| Symmetric distribution, no outliers | **Mean** |
| Skewed distribution / has outliers | **Median** |
| Income, house prices, response times | **Median** (usually right-skewed) |
| Test scores, heights, measurement errors | **Mean** (usually normal) |

```python
# Skewness detection via mean vs median
salaries = np.array([30000, 35000, 40000, 45000, 500000])

mean_sal   = np.mean(salaries)
median_sal = np.median(salaries)

print(f"Mean  : ₹{mean_sal:,.0f}")     # ₹130,000  ← misleading
print(f"Median: ₹{median_sal:,.0f}")   # ₹40,000   ← representative

if mean_sal > median_sal:
    print("→ Right-skewed (positive skew): use median")
elif mean_sal < median_sal:
    print("→ Left-skewed (negative skew): use median")
else:
    print("→ Symmetric: mean is fine")
```

```
Right-skewed (mean > median)      Left-skewed (mean < median)
     ▁▃█▇▅▃▂▁▁▁                        ▁▁▁▂▃▅▇█▃▁
      ↑  ↑                                   ↑  ↑
   median mean                            mean median
```

### 12.5 The Empirical Rule (68-95-99.7)
For normally distributed data:
- ~68% of values fall within **1 std** of the mean
- ~95% within **2 std**
- ~99.7% within **3 std**

```python
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=10000)

mean, std = data.mean(), data.std()

within_1 = np.sum(np.abs(data - mean) < 1*std) / len(data)
within_2 = np.sum(np.abs(data - mean) < 2*std) / len(data)
within_3 = np.sum(np.abs(data - mean) < 3*std) / len(data)

print(f"Within 1σ: {within_1:.1%}")     # ~68%
print(f"Within 2σ: {within_2:.1%}")     # ~95%
print(f"Within 3σ: {within_3:.1%}")     # ~99.7%

# Z-score outlier detection based on this rule
z_scores = np.abs((data - mean) / std)
outliers = data[z_scores > 3]
print(f"Outliers found: {len(outliers)}")
```

### 12.6 Complete Summary Function
```python
def describe(arr, name="Data"):
    """Full statistical summary of a NumPy array (like pandas .describe())."""
    arr = np.asarray(arr)
    clean = arr[~np.isnan(arr)] if arr.dtype.kind == 'f' else arr

    print(f"\n=== {name} ===")
    print(f"Count    : {clean.size}")
    print(f"Mean     : {np.mean(clean):.2f}")
    print(f"Median   : {np.median(clean):.2f}")
    print(f"Std      : {np.std(clean, ddof=1):.2f}")
    print(f"Variance : {np.var(clean, ddof=1):.2f}")
    print(f"Min      : {np.min(clean):.2f}")
    print(f"25%      : {np.percentile(clean, 25):.2f}")
    print(f"50%      : {np.percentile(clean, 50):.2f}")
    print(f"75%      : {np.percentile(clean, 75):.2f}")
    print(f"Max      : {np.max(clean):.2f}")
    print(f"Range    : {np.ptp(clean):.2f}")
    print(f"IQR      : {np.percentile(clean,75) - np.percentile(clean,25):.2f}")

    skew_hint = ("Right-skewed" if np.mean(clean) > np.median(clean)
                 else "Left-skewed" if np.mean(clean) < np.median(clean)
                 else "Symmetric")
    print(f"Shape    : {skew_hint}")

describe(np.array([12., 15, 18, 22, 25, 28, 30, 35, 40, 100]), "Sample Data")
```

---

## 13. Practical Mini Project

```python
# ============================================================
# NUMPY MINI PROJECT: Student Performance Analysis
# ============================================================
import numpy as np

np.random.seed(42)

# --- 1. Create the dataset ---
students = np.array(["Ravi","Meera","Sam","Priya","Arjun",
                     "Kavya","Rohit","Neha"])
subjects = np.array(["Math","Physics","Chemistry","English"])

# 8 students × 4 subjects
marks = np.random.randint(35, 100, size=(8, 4))

print("Dataset shape:", marks.shape)
print("Dimensions   :", marks.ndim)
print("Total entries:", marks.size)
print("\nMarks Matrix:\n", marks)

# --- 2. Student-wise analysis (axis=1) ---
student_avg = marks.mean(axis=1)
student_max = marks.max(axis=1)
student_min = marks.min(axis=1)

print("\n--- Student-wise ---")
for i, name in enumerate(students):
    print(f"{name:8} Avg: {student_avg[i]:5.1f}  "
          f"Best: {student_max[i]:3}  Worst: {student_min[i]:3}")

# --- 3. Subject-wise analysis (axis=0) ---
subject_avg = marks.mean(axis=0)
subject_std = marks.std(axis=0)

print("\n--- Subject-wise ---")
for i, sub in enumerate(subjects):
    print(f"{sub:10} Mean: {subject_avg[i]:5.1f}  Std: {subject_std[i]:5.2f}")

# --- 4. Key insights ---
print("\n--- Insights ---")
print(f"Class topper      : {students[np.argmax(student_avg)]} "
      f"({student_avg.max():.1f})")
print(f"Needs support     : {students[np.argmin(student_avg)]} "
      f"({student_avg.min():.1f})")
print(f"Easiest subject   : {subjects[np.argmax(subject_avg)]}")
print(f"Hardest subject   : {subjects[np.argmin(subject_avg)]}")
print(f"Most consistent   : {subjects[np.argmin(subject_std)]} "
      f"(std={subject_std.min():.2f})")

# --- 5. Pass/Fail using boolean masking ---
PASS_MARK = 40
passed = marks >= PASS_MARK
print(f"\nTotal pass entries: {passed.sum()} / {marks.size}")
print(f"Overall pass rate : {passed.mean():.1%}")

all_passed = passed.all(axis=1)
print("Students passing all subjects:", students[all_passed])

any_failed = (~passed).any(axis=1)
print("Students failing at least one:", students[any_failed])

# --- 6. Grading with np.where ---
grades = np.where(marks >= 90, 'A',
         np.where(marks >= 75, 'B',
         np.where(marks >= 60, 'C',
         np.where(marks >= 40, 'D', 'F'))))
print("\nGrades:\n", grades)

# --- 7. Normalization (Z-score) ---
normalized = (marks - marks.mean(axis=0)) / marks.std(axis=0)
print("\nNormalized (first 3 rows):\n", normalized[:3].round(2))
print("Check → mean ≈ 0:", normalized.mean(axis=0).round(6))
print("Check → std  ≈ 1:", normalized.std(axis=0).round(6))

# --- 8. Min-Max scaling to [0,1] ---
mn, mx = marks.min(axis=0), marks.max(axis=0)
scaled = (marks - mn) / (mx - mn)
print("\nMin-Max scaled (first 3 rows):\n", scaled[:3].round(2))

# --- 9. Outlier detection using IQR ---
flat = marks.flatten()
Q1, Q3 = np.percentile(flat, [25, 75])
IQR = Q3 - Q1
outliers = flat[(flat < Q1 - 1.5*IQR) | (flat > Q3 + 1.5*IQR)]
print(f"\nIQR: {IQR:.1f} | Outliers: {outliers}")

# --- 10. Correlation between subjects ---
corr = np.corrcoef(marks.T)
print("\nSubject correlation matrix:\n", corr.round(2))
```

---

## 14. Cheatsheet

### Creation
```python
np.array([1,2,3])            np.zeros((2,3))        np.ones((2,3))
np.full((2,3), 7)            np.eye(3)              np.arange(0,10,2)
np.linspace(0,1,5)           np.random.rand(2,3)    np.random.randn(3)
np.random.randint(0,10,5)    np.random.seed(42)
```

### Attributes
```python
arr.shape    arr.ndim     arr.size     arr.dtype
arr.itemsize arr.nbytes   arr.T        len(arr)
```

### Indexing & Slicing
```python
arr[0]           arr[-1]          arr[1,2]         arr[:, 0]
arr[1:4]         arr[::2]         arr[::-1]        arr[0:2, 0:2]
arr[arr > 5]     arr[[0,2,4]]     np.where(cond, a, b)
```

### Reshaping
```python
arr.reshape(3,4)    arr.reshape(-1,1)   arr.flatten()   arr.ravel()
arr.T               np.expand_dims()    np.squeeze()
np.concatenate()    np.vstack()         np.hstack()     np.split()
```

### Math
```python
a+b  a-b  a*b  a/b  a**2  a@b
np.sqrt() np.exp() np.log() np.abs() np.round()
np.sum() np.prod() np.cumsum() np.min() np.max()
np.argmin() np.argmax() np.clip()
np.dot() np.linalg.inv() np.linalg.det() np.linalg.norm()
```

### Statistics
```python
np.mean()      np.median()     np.std()        np.var()
np.percentile() np.quantile()  np.ptp()        np.average()
np.corrcoef()  np.cov()        np.histogram()  np.bincount()
np.nanmean()   np.nanmedian()  np.nanstd()     np.isnan()
```

### Key Rules to Remember 🏆
1. `axis=0` → operate **down** columns; `axis=1` → operate **across** rows.
2. Slicing returns a **view**, not a copy — use `.copy()` when you need independence.
3. `reshape(-1, 1)` converts a 1-D array into a column vector (needed by sklearn).
4. Use `&`, `|`, `~` with boolean masks — **not** `and`, `or`, `not`.
5. NumPy `std` defaults to `ddof=0`; Pandas defaults to `ddof=1`.
6. Normal functions return `nan` if any `nan` is present — use `np.nanmean()` etc.
7. Always `np.random.seed()` for reproducible results.
8. Arrays are **homogeneous** — mixing types forces upcasting to a common type.

---

## 📌 Summary

| Concept | Key Point |
|---|---|
| **Array** | Homogeneous, fixed-type, fast N-dimensional container |
| **Dimensions** | 0-D scalar, 1-D vector, 2-D matrix, 3-D+ tensor |
| **shape** | Tuple of sizes per axis, e.g. `(rows, cols)` |
| **ndim** | Integer count of axes = `len(shape)` |
| **Indexing** | `arr[row, col]`; boolean masking for filtering |
| **Slicing** | `arr[start:stop:step]`, returns a **view** |
| **Reshaping** | `reshape()`, `-1` auto-computes, `flatten()`/`ravel()` |
| **Math ops** | Vectorized element-wise; broadcasting expands shapes |
| **Statistics** | `mean`, `median`, `std`, `var`, `percentile`, `corrcoef` |
| **Mean** | Average — sensitive to outliers |
| **Median** | Middle value — robust to outliers |
| **Std** | Spread around the mean; `ddof=1` for samples |

---

*📁 Previous: Part 3 → Python Basics*
*📁 Next: Part 5 → Pandas for Data Analysis*

---
### 🔖 Tags
`#NumPy` `#Python` `#DataScience` `#Statistics` `#Arrays` `#Notes`
