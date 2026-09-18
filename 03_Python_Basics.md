# 🐍 Data Science Notes — Part 3: Python Basics

> Complete Python fundamentals required for Data Science — theory, syntax, and runnable code examples.

---

## Table of Contents
1. [Variables](#1-variables)
2. [Data Types](#2-data-types)
3. [Operators](#3-operators)
4. [Conditions](#4-conditions)
5. [Loops](#5-loops)
6. [Functions](#6-functions)
7. [Lists](#7-lists)
8. [Tuples](#8-tuples)
9. [Sets](#9-sets)
10. [Dictionaries](#10-dictionaries)
11. [Exception Handling](#11-exception-handling)
12. [File Handling](#12-file-handling)
13. [OOP Basics](#13-oop-basics)
14. [Quick Reference Cheatsheet](#14-quick-reference-cheatsheet)

---

## 1. Variables

A **variable** is a named reference to a value stored in memory. Python is **dynamically typed** — you don't declare the type, it's inferred at runtime.

### Rules for naming
| Rule | ✅ Valid | ❌ Invalid |
|---|---|---|
| Start with letter or `_` | `name`, `_temp` | `2name` |
| Only letters, digits, `_` | `total_marks`, `x1` | `total-marks`, `my var` |
| Case-sensitive | `Age` ≠ `age` | — |
| Can't be a keyword | `value` | `class`, `for`, `if` |

### Syntax & Examples
```python
# Basic assignment
name = "Anita"
age = 23
height = 5.4
is_student = True

# Multiple assignment
x, y, z = 1, 2, 3
a = b = c = 100          # all point to same value

# Swapping (Pythonic)
x, y = y, x

# Check type and identity
print(type(age))         # <class 'int'>
print(id(age))           # memory address

# Dynamic typing - type can change
var = 10
print(type(var))         # int
var = "hello"
print(type(var))         # str

# Constants (convention only - Python doesn't enforce)
PI = 3.14159
MAX_USERS = 1000

# Delete a variable
del var
```

### Naming Conventions (PEP 8)
```python
snake_case_variable = 1      # variables & functions
CONSTANT_VALUE = 2           # constants
ClassName = 3                # classes (PascalCase)
_private_var = 4             # internal use
```

### Keywords (reserved words)
```python
import keyword
print(keyword.kwlist)
# ['False','None','True','and','as','assert','async','await','break','class',
#  'continue','def','del','elif','else','except','finally','for','from','global',
#  'if','import','in','is','lambda','nonlocal','not','or','pass','raise',
#  'return','try','while','with','yield']
```

---

## 2. Data Types

```
Python Data Types
│
├── Numeric      → int, float, complex
├── Sequence     → str, list, tuple, range
├── Mapping      → dict
├── Set          → set, frozenset
├── Boolean      → bool
├── Binary       → bytes, bytearray, memoryview
└── None         → NoneType
```

### 2.1 Numeric Types
```python
a = 10             # int   - whole numbers, unlimited precision
b = 3.14           # float - decimal numbers
c = 2 + 3j         # complex - real + imaginary

print(type(a), type(b), type(c))
print(c.real, c.imag)      # 2.0 3.0

# Large integers work natively
big = 2 ** 100
print(big)                 # 1267650600228229401496703205376

# Float precision caution
print(0.1 + 0.2)           # 0.30000000000000004  ⚠️
print(round(0.1 + 0.2, 2)) # 0.3
```

### 2.2 String (str)
```python
s1 = 'Single quotes'
s2 = "Double quotes"
s3 = """Multi-line
string"""

# Indexing & slicing
text = "Data Science"
print(text[0])       # 'D'
print(text[-1])      # 'e'
print(text[0:4])     # 'Data'
print(text[5:])      # 'Science'
print(text[::-1])    # 'ecneicS ataD'  (reverse)
print(text[::2])     # 'Dt cec'        (every 2nd char)

# Common string methods
print(text.upper())            # DATA SCIENCE
print(text.lower())            # data science
print(text.title())            # Data Science
print(text.replace("Data","Big"))  # Big Science
print(text.split())            # ['Data', 'Science']
print("  hi  ".strip())        # 'hi'
print(text.find("Sci"))        # 5
print(text.count("a"))         # 2
print(len(text))               # 12
print("-".join(["a","b","c"])) # a-b-c
print(text.startswith("Data")) # True

# f-strings (preferred formatting)
name, marks = "Ravi", 88.567
print(f"{name} scored {marks:.2f}")        # Ravi scored 88.57
print(f"{marks:>10.1f}")                   # right-aligned
print(f"{name=}")                          # name='Ravi' (debugging)

# Other formatting
print("Hello, {}".format(name))
print("Hello, %s" % name)
```

### 2.3 Boolean & None
```python
flag = True
done = False

print(True + True)      # 2  (True=1, False=0)
print(bool(0))          # False
print(bool(""))         # False
print(bool([]))         # False
print(bool(None))       # False
print(bool("text"))     # True

# Falsy values: 0, 0.0, '', [], (), {}, set(), None, False
x = None
print(x is None)        # True  (use 'is', not '==')
```

### 2.4 Type Conversion
```python
# Implicit (automatic)
result = 5 + 2.0        # int + float → float (7.0)

# Explicit (casting)
int("25")               # 25
int(3.99)               # 3  (truncates, doesn't round)
float("3.14")           # 3.14
str(100)                # '100'
bool(1)                 # True
list("abc")             # ['a','b','c']
tuple([1,2,3])          # (1,2,3)
set([1,1,2])            # {1,2}
dict([('a',1)])         # {'a': 1}

# Safe conversion
value = "abc"
try:
    num = int(value)
except ValueError:
    num = 0
```

### 2.5 Mutable vs Immutable

| Immutable (cannot change) | Mutable (can change) |
|---|---|
| `int`, `float`, `complex` | `list` |
| `str` | `dict` |
| `tuple` | `set` |
| `bool`, `frozenset` | `bytearray` |

```python
# Immutable - creates new object
s = "hello"
print(id(s))
s += " world"
print(id(s))       # DIFFERENT id

# Mutable - modifies in place
lst = [1,2,3]
print(id(lst))
lst.append(4)
print(id(lst))     # SAME id
```

---

## 3. Operators

### 3.1 Arithmetic Operators
| Operator | Name | Example | Result |
|---|---|---|---|
| `+` | Addition | `7 + 2` | `9` |
| `-` | Subtraction | `7 - 2` | `5` |
| `*` | Multiplication | `7 * 2` | `14` |
| `/` | Division (float) | `7 / 2` | `3.5` |
| `//` | Floor division | `7 // 2` | `3` |
| `%` | Modulus (remainder) | `7 % 2` | `1` |
| `**` | Exponent | `7 ** 2` | `49` |

```python
a, b = 7, 2
print(a + b, a - b, a * b)     # 9 5 14
print(a / b)                   # 3.5
print(a // b)                  # 3
print(a % b)                   # 1
print(a ** b)                  # 49
print(-7 // 2)                 # -4  (floors toward -infinity)
```

### 3.2 Comparison (Relational) Operators
```python
print(5 == 5)      # True
print(5 != 3)      # True
print(5 > 3)       # True
print(5 < 3)       # False
print(5 >= 5)      # True
print(5 <= 4)      # False

# Chained comparison (Pythonic)
age = 25
print(18 <= age <= 60)   # True
```

### 3.3 Logical Operators
```python
a, b = True, False
print(a and b)     # False - both must be True
print(a or b)      # True  - at least one True
print(not a)       # False - inverts

# Short-circuit evaluation
x = 0
print(x != 0 and 10/x > 1)   # False (10/x never runs → no error)

# Truth table
# and: T&T=T, T&F=F, F&T=F, F&F=F
# or : T|T=T, T|F=T, F|T=T, F|F=F
```

### 3.4 Assignment Operators
```python
x = 10
x += 5     # x = x + 5  → 15
x -= 3     # 12
x *= 2     # 24
x /= 4     # 6.0
x //= 2    # 3.0
x %= 2     # 1.0
x **= 3    # 1.0

# Walrus operator := (Python 3.8+) - assign inside expression
numbers = [1,2,3,4,5]
if (n := len(numbers)) > 3:
    print(f"List has {n} items")
```

### 3.5 Bitwise Operators
```python
a, b = 10, 4       # 1010, 0100 in binary
print(a & b)       # 0   AND
print(a | b)       # 14  OR
print(a ^ b)       # 14  XOR
print(~a)          # -11 NOT
print(a << 1)      # 20  Left shift  (×2)
print(a >> 1)      # 5   Right shift (÷2)
```

### 3.6 Membership & Identity Operators
```python
# Membership: in, not in
fruits = ["apple","banana"]
print("apple" in fruits)       # True
print("mango" not in fruits)   # True
print("Sci" in "Data Science") # True

# Identity: is, is not (compares memory location)
a = [1,2,3]
b = [1,2,3]
c = a
print(a == b)      # True  - same VALUE
print(a is b)      # False - different OBJECTS
print(a is c)      # True  - same object
```

### 3.7 Operator Precedence (high → low)
```
()  →  **  →  +x, -x, ~x  →  * / // %  →  + -  →  << >>
→  &  →  ^  →  |  →  comparisons/in/is  →  not  →  and  →  or
```
```python
print(2 + 3 * 4)        # 14 (not 20)
print((2 + 3) * 4)      # 20
print(2 ** 3 ** 2)      # 512 (right-associative: 2**(3**2))
```

---

## 4. Conditions

### 4.1 if / elif / else
```python
marks = 85

if marks >= 90:
    grade = "A"
elif marks >= 75:
    grade = "B"
elif marks >= 60:
    grade = "C"
else:
    grade = "F"

print(f"Grade: {grade}")     # Grade: B
```

### 4.2 Nested if
```python
age = 25
income = 50000

if age >= 18:
    if income > 30000:
        print("Eligible for loan")
    else:
        print("Income too low")
else:
    print("Underage")
```

### 4.3 Ternary (Conditional Expression)
```python
age = 20
status = "Adult" if age >= 18 else "Minor"
print(status)

# Nested ternary
n = 0
sign = "Positive" if n > 0 else ("Negative" if n < 0 else "Zero")
```

### 4.4 match-case (Python 3.10+)
```python
day = "Mon"

match day:
    case "Sat" | "Sun":
        print("Weekend")
    case "Mon":
        print("Start of week")
    case _:                      # default
        print("Weekday")
```

### 4.5 pass statement
```python
if marks > 90:
    pass          # placeholder - do nothing (avoids syntax error)
else:
    print("Keep going")
```

### DS-Relevant Example
```python
def categorize_age(age):
    if age < 13:
        return "Child"
    elif age < 20:
        return "Teen"
    elif age < 60:
        return "Adult"
    return "Senior"

ages = [5, 15, 30, 70]
print([categorize_age(a) for a in ages])
# ['Child', 'Teen', 'Adult', 'Senior']
```

---

## 5. Loops

### 5.1 for loop
```python
# Iterate over a sequence
for fruit in ["apple","banana","mango"]:
    print(fruit)

# range()
for i in range(5):           # 0,1,2,3,4
    print(i)

for i in range(2, 10, 2):    # start, stop, step → 2,4,6,8
    print(i)

for i in range(10, 0, -1):   # countdown
    print(i)

# With index - enumerate()
colors = ["red","green","blue"]
for idx, color in enumerate(colors):
    print(idx, color)

for idx, color in enumerate(colors, start=1):   # start at 1
    print(idx, color)

# Two lists together - zip()
names = ["Ravi","Meera"]
marks = [88, 92]
for n, m in zip(names, marks):
    print(f"{n}: {m}")

# Iterate over a string
for ch in "Data":
    print(ch)

# Iterate over a dictionary
person = {"name":"Anita", "age":23}
for key in person:                     # keys
    print(key)
for key, val in person.items():        # key-value pairs
    print(key, val)
```

### 5.2 while loop
```python
count = 1
while count <= 5:
    print(count)
    count += 1

# With user input
total = 0
while True:
    n = input("Enter number (q to quit): ")
    if n == 'q':
        break
    total += int(n)
print("Total:", total)
```

### 5.3 Loop Control Statements
```python
# break - exit loop entirely
for i in range(10):
    if i == 5:
        break
    print(i)          # 0 1 2 3 4

# continue - skip current iteration
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)          # 1 3 5 7 9

# pass - do nothing
for i in range(3):
    pass

# else with loop - runs if loop completed WITHOUT break
for i in range(5):
    if i == 10:
        break
else:
    print("Loop finished normally")     # This runs
```

### 5.4 Nested Loops
```python
# Multiplication table
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i}x{j}={i*j}", end="  ")
    print()

# Pattern printing
for i in range(1, 5):
    print("*" * i)
# *
# **
# ***
# ****
```

### 5.5 Comprehensions (Pythonic loops)
```python
# List comprehension
squares = [x**2 for x in range(1, 6)]              # [1,4,9,16,25]
evens   = [x for x in range(20) if x % 2 == 0]     # filtering
labels  = ["even" if x%2==0 else "odd" for x in range(5)]

# Nested
matrix = [[i*j for j in range(3)] for i in range(3)]

# Dict comprehension
sq_dict = {x: x**2 for x in range(1, 5)}           # {1:1, 2:4, 3:9, 4:16}

# Set comprehension
unique = {x % 3 for x in range(10)}                # {0,1,2}

# Generator expression (memory-efficient)
gen = (x**2 for x in range(1000000))
print(sum(gen))
```

---

## 6. Functions

A reusable block of code that performs a specific task. Follows **DRY** (Don't Repeat Yourself).

### 6.1 Basic Syntax
```python
def function_name(parameters):
    """Docstring: describes what the function does."""
    # body
    return value

# Example
def greet(name):
    """Return a greeting message."""
    return f"Hello, {name}!"

print(greet("Anita"))      # Hello, Anita!
print(greet.__doc__)       # prints the docstring
```

### 6.2 Types of Arguments
```python
# 1. Positional arguments
def add(a, b):
    return a + b
print(add(3, 5))

# 2. Keyword arguments
def info(name, age):
    return f"{name} is {age}"
print(info(age=25, name="Ravi"))     # order doesn't matter

# 3. Default arguments
def power(base, exp=2):
    return base ** exp
print(power(3))        # 9  (uses default)
print(power(3, 3))     # 27

# 4. *args - variable positional arguments (tuple)
def total(*numbers):
    return sum(numbers)
print(total(1,2,3,4,5))     # 15

# 5. **kwargs - variable keyword arguments (dict)
def details(**info):
    for k, v in info.items():
        print(f"{k}: {v}")
details(name="Meera", age=22, city="Pune")

# All combined (order matters!)
def demo(a, b=10, *args, **kwargs):
    print(a, b, args, kwargs)
demo(1, 2, 3, 4, x=5, y=6)     # 1 2 (3,4) {'x':5,'y':6}
```

> ⚠️ **Mutable default argument trap:**
> ```python
> def bad(items=[]):     # ❌ list is shared across calls
>     items.append(1)
>     return items
>
> def good(items=None):  # ✅ correct pattern
>     if items is None:
>         items = []
>     items.append(1)
>     return items
> ```

### 6.3 Return Values
```python
# Single value
def square(x):
    return x * x

# Multiple values (returns a tuple)
def stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

lo, hi, avg = stats([10, 20, 30])
print(lo, hi, avg)      # 10 30 20.0

# No return → returns None
def show(x):
    print(x)
result = show(5)
print(result)           # None
```

### 6.4 Lambda (Anonymous) Functions
```python
# Syntax: lambda arguments: expression
square = lambda x: x**2
print(square(5))                # 25

add = lambda a, b: a + b
print(add(3, 4))                # 7

# Common use with map/filter/reduce/sorted
nums = [1,2,3,4,5,6]

squares = list(map(lambda x: x**2, nums))          # [1,4,9,16,25,36]
evens   = list(filter(lambda x: x%2==0, nums))     # [2,4,6]

from functools import reduce
product = reduce(lambda a,b: a*b, nums)            # 720

# Sorting with lambda key
students = [("Ravi",88), ("Meera",92), ("Sam",79)]
print(sorted(students, key=lambda s: s[1], reverse=True))
# [('Meera',92), ('Ravi',88), ('Sam',79)]
```

### 6.5 Scope (LEGB Rule)
```
Local → Enclosing → Global → Built-in
```
```python
x = "global"

def outer():
    x = "enclosing"
    def inner():
        x = "local"
        print(x)     # local
    inner()
    print(x)         # enclosing

outer()
print(x)             # global

# Modifying global variable
counter = 0
def increment():
    global counter
    counter += 1
increment()
print(counter)       # 1

# nonlocal for enclosing scope
def outer2():
    n = 0
    def inner2():
        nonlocal n
        n += 1
    inner2()
    return n
print(outer2())      # 1
```

### 6.6 Recursion
```python
def factorial(n):
    if n <= 1:           # base case - ALWAYS needed
        return 1
    return n * factorial(n-1)    # recursive case

print(factorial(5))      # 120

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print([fibonacci(i) for i in range(10)])
# [0,1,1,2,3,5,8,13,21,34]
```

### 6.7 Decorators (bonus)
```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.4f}s")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

slow_function()      # slow_function took 1.0012s
```

---

## 7. Lists

**Ordered, mutable, allows duplicates, indexed.**

### 7.1 Creating & Accessing
```python
lst = [1, 2, 3, 4, 5]
mixed = [1, "two", 3.0, True, [5,6]]
empty = []
from_range = list(range(5))       # [0,1,2,3,4]

# Indexing
print(lst[0])       # 1  (first)
print(lst[-1])      # 5  (last)
print(lst[-2])      # 4

# Slicing  lst[start:stop:step]
print(lst[1:4])     # [2,3,4]
print(lst[:3])      # [1,2,3]
print(lst[2:])      # [3,4,5]
print(lst[::2])     # [1,3,5]
print(lst[::-1])    # [5,4,3,2,1]  reverse
```

### 7.2 List Methods
| Method | Description | Example |
|---|---|---|
| `append(x)` | Add item at end | `lst.append(6)` |
| `insert(i,x)` | Insert at index | `lst.insert(0, 0)` |
| `extend(iter)` | Add multiple items | `lst.extend([7,8])` |
| `remove(x)` | Remove first occurrence | `lst.remove(3)` |
| `pop(i)` | Remove & return item | `lst.pop()` |
| `clear()` | Remove all items | `lst.clear()` |
| `index(x)` | Find index of value | `lst.index(4)` |
| `count(x)` | Count occurrences | `lst.count(2)` |
| `sort()` | Sort in place | `lst.sort(reverse=True)` |
| `reverse()` | Reverse in place | `lst.reverse()` |
| `copy()` | Shallow copy | `new = lst.copy()` |

```python
lst = [3, 1, 4, 1, 5]

lst.append(9)              # [3,1,4,1,5,9]
lst.insert(0, 0)           # [0,3,1,4,1,5,9]
lst.extend([2,6])          # [0,3,1,4,1,5,9,2,6]
lst.remove(1)              # removes FIRST 1
popped = lst.pop()         # removes & returns last
popped = lst.pop(0)        # removes & returns index 0
print(lst.count(1))        # frequency
print(lst.index(4))        # position

lst.sort()                 # ascending, in place
lst.sort(reverse=True)     # descending
sorted_new = sorted(lst)   # returns NEW list, original unchanged
lst.reverse()
```

### 7.3 Operations & Built-ins
```python
a = [1,2,3]
b = [4,5,6]

print(a + b)          # [1,2,3,4,5,6]  concatenation
print(a * 2)          # [1,2,3,1,2,3]  repetition
print(3 in a)         # True
print(len(a))         # 3
print(max(a), min(a), sum(a))    # 3 1 6

# Unpacking
x, y, z = [1,2,3]
first, *rest = [1,2,3,4]         # first=1, rest=[2,3,4]
```

### 7.4 Copying (Important!)
```python
original = [1, 2, [3, 4]]

# ❌ Assignment - NOT a copy, same object
wrong = original
wrong.append(5)
print(original)      # [1,2,[3,4],5]  ← original changed!

# ✅ Shallow copy
shallow = original.copy()      # or list(original) or original[:]

# ✅ Deep copy (for nested structures)
import copy
deep = copy.deepcopy(original)
deep[2].append(99)
print(original)      # unchanged
```

### 7.5 Sorting Complex Data
```python
students = [
    {"name":"Ravi",  "marks":88},
    {"name":"Meera", "marks":92},
    {"name":"Sam",   "marks":79}
]

# Sort by marks descending
top = sorted(students, key=lambda s: s["marks"], reverse=True)
for s in top:
    print(s["name"], s["marks"])

# Sort by multiple keys
data = [("A", 2), ("B", 1), ("A", 1)]
print(sorted(data, key=lambda x: (x[0], x[1])))
```

---

## 8. Tuples

**Ordered, IMMUTABLE, allows duplicates, indexed.**

### Why use tuples?
- Faster than lists (fixed size)
- Can be used as dictionary keys (hashable)
- Protects data from accidental modification

```python
t = (1, 2, 3)
single = (5,)             # ⚠️ comma required for single-element tuple
not_tuple = (5)           # this is just int 5
empty = ()
no_parens = 1, 2, 3       # parentheses optional

# Access (same as list)
print(t[0], t[-1])
print(t[0:2])             # (1,2)

# ❌ Cannot modify
# t[0] = 10               # TypeError

# But mutable objects INSIDE a tuple can change
t2 = (1, [2,3])
t2[1].append(4)
print(t2)                 # (1, [2,3,4])

# Methods (only 2)
print(t.count(2))         # 1
print(t.index(3))         # 2

# Unpacking
a, b, c = (1, 2, 3)
x, *y = (1, 2, 3, 4)      # x=1, y=[2,3,4]

# Tuples as dict keys (lists can't be)
locations = {(12.97, 77.59): "Bengaluru", (19.07, 72.87): "Mumbai"}
print(locations[(12.97, 77.59)])

# Common use: multiple return values
def min_max(nums):
    return min(nums), max(nums)      # returns a tuple
lo, hi = min_max([4,1,9])

# Named tuples (readable)
from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])
p = Point(3, 4)
print(p.x, p.y)           # 3 4
```

### List vs Tuple
| Feature | List | Tuple |
|---|---|---|
| Syntax | `[1,2,3]` | `(1,2,3)` |
| Mutable | ✅ Yes | ❌ No |
| Speed | Slower | Faster |
| Memory | More | Less |
| Dict key | ❌ No | ✅ Yes |
| Methods | ~11 | 2 |
| Use case | Data that changes | Fixed/constant data |

---

## 9. Sets

**Unordered, mutable, NO duplicates, unindexed.**

```python
s = {1, 2, 3, 4}
from_list = set([1,1,2,2,3])       # {1,2,3}  auto-removes duplicates
empty = set()                       # ⚠️ NOT {} - that's a dict!

# ❌ No indexing
# print(s[0])        # TypeError

# Methods
s.add(5)                  # add single element
s.update([6,7])           # add multiple
s.remove(3)               # removes; KeyError if absent
s.discard(100)            # removes; NO error if absent
popped = s.pop()          # removes arbitrary element
s.clear()                 # empty the set
```

### Set Operations (very useful in DS)
```python
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

print(A | B)    # Union         {1,2,3,4,5,6}   or A.union(B)
print(A & B)    # Intersection  {3,4}           or A.intersection(B)
print(A - B)    # Difference    {1,2}           or A.difference(B)
print(A ^ B)    # Symmetric diff {1,2,5,6}      or A.symmetric_difference(B)

print(A.issubset({1,2,3,4,5}))     # True
print(A.issuperset({1,2}))         # True
print(A.isdisjoint({9,10}))        # True (no common elements)
```

### Practical DS Uses
```python
# 1. Remove duplicates fast
data = [1,2,2,3,3,3,4]
unique = list(set(data))            # [1,2,3,4]

# 2. Fast membership testing - O(1) vs O(n) for lists
valid_ids = set(range(1000000))
print(999999 in valid_ids)          # very fast

# 3. Find common/missing columns between datasets
cols_train = {"age","salary","city","target"}
cols_test  = {"age","salary","city"}
print(cols_train - cols_test)       # {'target'} - missing in test

# frozenset - immutable set
fs = frozenset([1,2,3])
```

---

## 10. Dictionaries

**Key-Value pairs. Ordered (Python 3.7+), mutable, keys must be unique & immutable.**

```python
person = {"name": "Anita", "age": 23, "city": "Pune"}
empty = {}
from_pairs = dict([("a",1), ("b",2)])
from_kwargs = dict(name="Ravi", age=25)

# Access
print(person["name"])              # Anita
print(person.get("age"))           # 23
print(person.get("email", "N/A"))  # N/A  (safe - no KeyError)
# print(person["email"])           # ❌ KeyError

# Add / Update
person["email"] = "a@x.com"        # add new key
person["age"] = 24                 # update existing
person.update({"phone": "12345", "age": 25})

# Delete
del person["phone"]
removed = person.pop("email")      # remove & return value
person.popitem()                   # remove & return last item
# person.clear()                   # remove everything
```

### Dictionary Methods
```python
d = {"a":1, "b":2, "c":3}

print(d.keys())      # dict_keys(['a','b','c'])
print(d.values())    # dict_values([1,2,3])
print(d.items())     # dict_items([('a',1),('b',2),('c',3)])

print(list(d.keys()))
print(len(d))
print("a" in d)      # True (checks KEYS)

# setdefault - get, or set if missing
d.setdefault("e", 5)

# Merge (Python 3.9+)
d2 = {"d": 4}
merged = d | d2
d |= d2              # in-place merge
```

### Iterating
```python
d = {"Ravi":88, "Meera":92, "Sam":79}

for key in d:
    print(key, d[key])

for key, value in d.items():
    print(f"{key} scored {value}")

for value in d.values():
    print(value)

# Sort by value
top = sorted(d.items(), key=lambda x: x[1], reverse=True)
print(top)       # [('Meera',92), ('Ravi',88), ('Sam',79)]
```

### Nested Dictionaries
```python
students = {
    "S001": {"name":"Ravi",  "marks": {"math":88, "sci":91}},
    "S002": {"name":"Meera", "marks": {"math":92, "sci":85}}
}

print(students["S001"]["marks"]["math"])    # 88

for sid, info in students.items():
    avg = sum(info["marks"].values()) / len(info["marks"])
    print(f"{info['name']}: {avg:.1f}")
```

### Dict Comprehension & Practical Uses
```python
# Comprehension
squares = {x: x**2 for x in range(1,6)}
filtered = {k:v for k,v in d.items() if v > 80}
inverted = {v:k for k,v in d.items()}

# Counting frequency
text = "data science is data driven"
freq = {}
for word in text.split():
    freq[word] = freq.get(word, 0) + 1
print(freq)

# Better: Counter
from collections import Counter
print(Counter(text.split()))
print(Counter(text.split()).most_common(2))

# defaultdict - no KeyError
from collections import defaultdict
dd = defaultdict(list)
dd["fruits"].append("apple")     # key auto-created

# Dict → DataFrame (the DS bridge)
import pandas as pd
data = {"name":["Ravi","Meera"], "marks":[88,92]}
df = pd.DataFrame(data)
print(df)
```

### Collections Comparison
| Type | Ordered | Mutable | Duplicates | Indexed | Syntax |
|---|---|---|---|---|---|
| **List** | ✅ | ✅ | ✅ | ✅ | `[1,2,3]` |
| **Tuple** | ✅ | ❌ | ✅ | ✅ | `(1,2,3)` |
| **Set** | ❌ | ✅ | ❌ | ❌ | `{1,2,3}` |
| **Dict** | ✅ | ✅ | ❌ (keys) | By key | `{'a':1}` |

---

## 11. Exception Handling

Handling **runtime errors** gracefully so the program doesn't crash.

### Common Built-in Exceptions
| Exception | Cause |
|---|---|
| `ZeroDivisionError` | Division by zero |
| `ValueError` | Right type, wrong value — `int("abc")` |
| `TypeError` | Wrong type — `"a" + 1` |
| `NameError` | Undefined variable |
| `IndexError` | List index out of range |
| `KeyError` | Dictionary key not found |
| `FileNotFoundError` | File doesn't exist |
| `AttributeError` | Object has no such attribute |
| `ImportError` | Module not found |

### 11.1 Basic try-except
```python
try:
    num = int(input("Enter a number: "))
    result = 100 / num
    print(result)
except ValueError:
    print("Please enter a valid number")
except ZeroDivisionError:
    print("Cannot divide by zero")
```

### 11.2 Full Structure
```python
try:
    # code that may raise an error
    f = open("data.txt")
    data = f.read()
except FileNotFoundError as e:
    print(f"File error: {e}")
except Exception as e:                 # catch-all (use last)
    print(f"Unexpected error: {type(e).__name__}: {e}")
else:
    print("Success! No exceptions occurred")    # runs ONLY if no error
finally:
    print("This ALWAYS runs - cleanup goes here")
    # f.close()
```

### 11.3 Multiple Exceptions Together
```python
try:
    value = int("abc")
except (ValueError, TypeError) as e:
    print(f"Conversion failed: {e}")
```

### 11.4 Raising Exceptions
```python
def set_age(age):
    if not isinstance(age, int):
        raise TypeError("Age must be an integer")
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age

try:
    set_age(-5)
except ValueError as e:
    print("Error:", e)

# assert - for debugging/validation
def divide(a, b):
    assert b != 0, "Denominator cannot be zero"
    return a / b
```

### 11.5 Custom Exceptions
```python
class InsufficientDataError(Exception):
    """Raised when dataset has too few rows for modeling."""
    pass

def train_model(df):
    if len(df) < 100:
        raise InsufficientDataError(
            f"Need at least 100 rows, got {len(df)}"
        )
    print("Training...")

try:
    import pandas as pd
    train_model(pd.DataFrame({"a":[1,2,3]}))
except InsufficientDataError as e:
    print("Custom error:", e)
```

### 11.6 DS-Relevant Example
```python
import pandas as pd

def safe_load_csv(path):
    """Load a CSV with proper error handling."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return None
    except pd.errors.EmptyDataError:
        print("❌ File is empty")
        return None
    except pd.errors.ParserError:
        print("❌ File is malformed")
        return None
    else:
        print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        return df

df = safe_load_csv("data.csv")

# Safe numeric conversion in a column
def safe_convert(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
```

---

## 12. File Handling

### 12.1 File Modes
| Mode | Meaning |
|---|---|
| `'r'` | Read (default) — error if file missing |
| `'w'` | Write — **overwrites** existing / creates new |
| `'a'` | Append — adds to end / creates new |
| `'x'` | Exclusive create — error if file exists |
| `'r+'` | Read + Write |
| `'b'` | Binary mode (e.g. `'rb'`, `'wb'`) |
| `'t'` | Text mode (default) |

### 12.2 Reading Files
```python
# ✅ Best practice - with statement (auto-closes file)
with open("data.txt", "r") as f:
    content = f.read()              # entire file as one string
print(content)

with open("data.txt", "r") as f:
    line = f.readline()             # one line
    lines = f.readlines()           # list of all lines

# Line by line (memory efficient for large files)
with open("data.txt", "r") as f:
    for line in f:
        print(line.strip())         # strip removes \n

# Manual open/close (not recommended)
f = open("data.txt", "r")
data = f.read()
f.close()                           # must remember to close!
```

### 12.3 Writing Files
```python
# Write (overwrites!)
with open("output.txt", "w") as f:
    f.write("Hello World\n")
    f.write("Second line\n")
    f.writelines(["line3\n", "line4\n"])

# Append
with open("output.txt", "a") as f:
    f.write("Appended line\n")

# Write a list of records
records = ["Ravi,88", "Meera,92", "Sam,79"]
with open("marks.txt", "w") as f:
    for r in records:
        f.write(r + "\n")
```

### 12.4 Working with CSV
```python
import csv

# Reading
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    header = next(reader)           # skip header row
    for row in reader:
        print(row)                  # row is a list

# Reading as dictionary
with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"], row["marks"])

# Writing
with open("out.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name","marks"])
    writer.writerows([["Ravi",88], ["Meera",92]])

# Writing dictionaries
with open("out.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name","marks"])
    writer.writeheader()
    writer.writerow({"name":"Ravi", "marks":88})
```

### 12.5 Working with JSON
```python
import json

data = {"name":"Anita", "skills":["Python","SQL"], "exp":2}

# Write JSON
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)

# Read JSON
with open("data.json", "r") as f:
    loaded = json.load(f)
print(loaded["skills"])

# String conversion
json_string = json.dumps(data)        # dict → JSON string
back_to_dict = json.loads(json_string)  # JSON string → dict
```

### 12.6 File Operations (os / pathlib)
```python
import os

print(os.path.exists("data.txt"))     # True/False
print(os.path.getsize("data.txt"))    # size in bytes
print(os.listdir("."))                # files in current directory
os.rename("old.txt", "new.txt")
os.remove("temp.txt")                 # delete file
os.makedirs("new_folder", exist_ok=True)

# Modern approach - pathlib
from pathlib import Path

p = Path("data.txt")
print(p.exists(), p.suffix, p.stem, p.parent)
text = p.read_text()
p.write_text("new content")

# List all CSVs in a folder
for csv_file in Path("data/").glob("*.csv"):
    print(csv_file)
```

### 12.7 DS-Relevant Example
```python
import pandas as pd
from pathlib import Path

# Read all CSVs in a folder and combine them
folder = Path("datasets/")
all_dfs = []

for file in folder.glob("*.csv"):
    try:
        df = pd.read_csv(file)
        df['source_file'] = file.name
        all_dfs.append(df)
        print(f"✅ Loaded {file.name}: {df.shape}")
    except Exception as e:
        print(f"❌ Failed {file.name}: {e}")

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("combined_data.csv", index=False)
    print(f"Final shape: {combined.shape}")
```

---

## 13. OOP Basics

**Object-Oriented Programming** organizes code into objects that bundle **data (attributes)** and **behavior (methods)**.

### 13.1 Class & Object
```python
class Student:
    # Class variable - shared by ALL instances
    school = "ABC School"

    # Constructor - runs when object is created
    def __init__(self, name, marks):
        self.name = name       # instance variable
        self.marks = marks

    # Instance method
    def display(self):
        return f"{self.name} scored {self.marks}"

    def is_pass(self):
        return self.marks >= 40

# Creating objects (instances)
s1 = Student("Ravi", 88)
s2 = Student("Meera", 35)

print(s1.display())       # Ravi scored 88
print(s2.is_pass())       # False
print(s1.school)          # ABC School
print(Student.school)     # ABC School
```

### 13.2 The Four Pillars of OOP

#### A) Encapsulation — bundling data + hiding internals
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance       # __ = private (name mangled)

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
        else:
            raise ValueError("Amount must be positive")

    def get_balance(self):             # getter
        return self.__balance

acc = BankAccount(1000)
acc.deposit(500)
print(acc.get_balance())     # 1500
# print(acc.__balance)       # ❌ AttributeError
```

**Access modifiers (by convention):**
| Prefix | Meaning |
|---|---|
| `name` | Public |
| `_name` | Protected (internal use — convention only) |
| `__name` | Private (name-mangled) |

#### B) Inheritance — reuse code from a parent class
```python
# Parent / Base class
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def intro(self):
        return f"I am {self.name}, {self.age} years old"

# Child / Derived class
class Employee(Person):
    def __init__(self, name, age, salary):
        super().__init__(name, age)     # call parent constructor
        self.salary = salary

    def intro(self):                    # override parent method
        return f"{super().intro()} and I earn {self.salary}"

e = Employee("Ravi", 30, 50000)
print(e.intro())
print(isinstance(e, Person))       # True
print(issubclass(Employee, Person)) # True
```

**Types of inheritance:**
```python
# Single      : class B(A)
# Multilevel  : class C(B), class B(A)
# Multiple    : class C(A, B)
# Hierarchical: class B(A), class C(A)

class A:
    def hello(self): return "A"
class B:
    def hi(self): return "B"
class C(A, B):      # multiple inheritance
    pass

c = C()
print(c.hello(), c.hi())
print(C.__mro__)    # Method Resolution Order
```

#### C) Polymorphism — same interface, different behavior
```python
# Method overriding
class Shape:
    def area(self):
        raise NotImplementedError("Subclass must implement area()")

class Circle(Shape):
    def __init__(self, r): self.r = r
    def area(self): return 3.14159 * self.r ** 2

class Rectangle(Shape):
    def __init__(self, l, b): self.l, self.b = l, b
    def area(self): return self.l * self.b

# Same method call, different results
shapes = [Circle(5), Rectangle(4, 6)]
for s in shapes:
    print(f"{type(s).__name__}: {s.area():.2f}")

# Duck typing - "if it walks like a duck..."
class Dog:
    def speak(self): return "Woof"
class Cat:
    def speak(self): return "Meow"

for animal in [Dog(), Cat()]:
    print(animal.speak())
```

#### D) Abstraction — hide complexity, expose only essentials
```python
from abc import ABC, abstractmethod

class MLModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class MyRegressor(MLModel):
    def fit(self, X, y):
        self.mean = sum(y) / len(y)
        return self

    def predict(self, X):
        return [self.mean] * len(X)

m = MyRegressor()
m.fit([[1],[2],[3]], [10, 20, 30])
print(m.predict([[4],[5]]))       # [20.0, 20.0]

# model = MLModel()   # ❌ TypeError: can't instantiate abstract class
```

### 13.3 Special (Dunder / Magic) Methods
```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):                 # for print()
        return f"Vector({self.x}, {self.y})"

    def __repr__(self):                # for developers/debugging
        return f"Vector(x={self.x}, y={self.y})"

    def __add__(self, other):          # operator overloading: +
        return Vector(self.x + other.x, self.y + other.y)

    def __eq__(self, other):           # ==
        return self.x == other.x and self.y == other.y

    def __len__(self):                 # len()
        return 2

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)          # Vector(4, 6)
print(v1 == v2)         # False
print(len(v1))          # 2
```

| Dunder | Triggered by |
|---|---|
| `__init__` | Object creation |
| `__str__` | `print(obj)`, `str(obj)` |
| `__repr__` | `repr(obj)`, console output |
| `__len__` | `len(obj)` |
| `__add__` / `__sub__` | `+` / `-` |
| `__eq__` / `__lt__` | `==` / `<` |
| `__getitem__` | `obj[key]` |
| `__call__` | `obj()` |

### 13.4 Class Methods, Static Methods, Properties
```python
class Employee:
    raise_percent = 1.05
    count = 0

    def __init__(self, name, salary):
        self.name = name
        self._salary = salary
        Employee.count += 1

    # Instance method - works with the object
    def apply_raise(self):
        self._salary *= Employee.raise_percent

    # Class method - works with the class
    @classmethod
    def from_string(cls, emp_str):
        name, salary = emp_str.split("-")
        return cls(name, int(salary))

    @classmethod
    def set_raise(cls, amount):
        cls.raise_percent = amount

    # Static method - utility, no self/cls needed
    @staticmethod
    def is_workday(day):
        return day not in ("Sat", "Sun")

    # Property - method accessed like an attribute
    @property
    def salary(self):
        return self._salary

    @salary.setter
    def salary(self, value):
        if value < 0:
            raise ValueError("Salary cannot be negative")
        self._salary = value

e1 = Employee("Ravi", 50000)
e2 = Employee.from_string("Meera-60000")     # alternative constructor

print(e1.salary)           # 50000 (no parentheses - property)
e1.salary = 55000          # uses setter with validation
print(Employee.is_workday("Sun"))    # False
print(Employee.count)      # 2
```

### 13.5 DS-Relevant OOP Example
```python
import pandas as pd
import numpy as np

class DataCleaner:
    """Reusable data cleaning pipeline."""

    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.log = []

    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        self.log.append(f"Removed {before - len(self.df)} duplicates")
        return self                      # enables method chaining

    def fill_missing(self, strategy="median"):
        for col in self.df.select_dtypes(include=np.number).columns:
            if self.df[col].isnull().any():
                value = (self.df[col].median() if strategy == "median"
                         else self.df[col].mean())
                self.df[col] = self.df[col].fillna(value)
                self.log.append(f"Filled {col} with {strategy}")
        return self

    def remove_outliers(self, column):
        Q1, Q3 = self.df[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        before = len(self.df)
        self.df = self.df[
            (self.df[column] >= Q1 - 1.5*IQR) &
            (self.df[column] <= Q3 + 1.5*IQR)
        ]
        self.log.append(f"Removed {before - len(self.df)} outliers from {column}")
        return self

    def report(self):
        print(f"Shape: {self.original_shape} → {self.df.shape}")
        for entry in self.log:
            print(f"  • {entry}")
        return self.df

# Usage - method chaining
df = pd.DataFrame({
    "age":    [25, 30, np.nan, 35, 200, 25],
    "salary": [50000, 60000, 55000, np.nan, 70000, 50000]
})

clean_df = (DataCleaner(df)
            .remove_duplicates()
            .fill_missing("median")
            .remove_outliers("age")
            .report())
```

---

## 14. Quick Reference Cheatsheet

### Data Structure Selection Guide
| Need | Use |
|---|---|
| Ordered, changeable collection | **List** |
| Fixed data / dict key / fast | **Tuple** |
| Unique items / fast lookup / set math | **Set** |
| Key-value mapping | **Dict** |

### Most-Used Built-in Functions
```python
len(x)         type(x)        print(x)       input()
int() float() str() bool() list() tuple() set() dict()
range(start, stop, step)
enumerate(iterable, start=0)
zip(a, b)
sorted(iterable, key=..., reverse=...)
reversed(iterable)
sum() min() max() abs() round()
any(iterable)   all(iterable)
map(func, iterable)      filter(func, iterable)
isinstance(obj, type)
dir(obj)       help(obj)      id(obj)
```

### Common Patterns
```python
# Swap
a, b = b, a

# Reverse a string/list
s[::-1]

# Flatten a nested list
flat = [item for sub in nested for item in sub]

# Count frequency
from collections import Counter
Counter(items)

# Remove duplicates keeping order
list(dict.fromkeys(items))

# Check all/any condition
all(x > 0 for x in nums)
any(x < 0 for x in nums)

# Merge dicts
merged = {**d1, **d2}      # or d1 | d2 (3.9+)

# Conditional assignment
value = x if condition else y

# Multiple conditions
if all([cond1, cond2, cond3]):
    ...
```

### Common Beginner Mistakes ⚠️
| Mistake | Fix |
|---|---|
| `if x = 5:` | Use `==` for comparison |
| Mutable default arg `def f(x=[])` | Use `def f(x=None)` |
| `lst2 = lst1` thinking it copies | Use `lst1.copy()` or `deepcopy` |
| `{}` for empty set | Use `set()` |
| Modifying a list while looping it | Loop over a copy: `for x in lst[:]` |
| Forgetting `f.close()` | Use `with open(...) as f:` |
| `==` for `None` | Use `is None` |
| Indentation mixing tabs/spaces | Use 4 spaces consistently |

---

## 📌 Summary

| Topic | Key Takeaway |
|---|---|
| **Variables** | Dynamically typed; snake_case naming |
| **Data Types** | int, float, str, bool, list, tuple, set, dict, None |
| **Operators** | Arithmetic, comparison, logical, membership, identity |
| **Conditions** | `if/elif/else`, ternary, `match-case` |
| **Loops** | `for`, `while`, `break`/`continue`, comprehensions |
| **Functions** | `def`, `*args`/`**kwargs`, lambda, scope (LEGB) |
| **List** | Ordered, mutable — `[]` |
| **Tuple** | Ordered, immutable — `()` |
| **Set** | Unordered, unique — `{}` via `set()` |
| **Dict** | Key-value, ordered (3.7+) — `{k:v}` |
| **Exceptions** | `try/except/else/finally`, `raise`, custom exceptions |
| **Files** | `with open()`, CSV, JSON, pathlib |
| **OOP** | Class, Object, Encapsulation, Inheritance, Polymorphism, Abstraction |

---

*📁 Previous: Part 2 → Data Science Workflow*
*📁 Next: Part 4 → NumPy & Pandas for Data Science*

---
### 🔖 Tags
`#Python` `#PythonBasics` `#DataScience` `#OOP` `#Notes` `#Beginner`
