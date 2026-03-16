# Pandas Masterclass — From Basics to Advanced

> **Prerequisites:** Python basics, NumPy fundamentals  
> **Install:** `pip install pandas openpyxl xlrd`  
> **Import convention:** `import pandas as pd`

---

## Table of Contents

1. [What is Pandas and Why Use It?](#1-what-is-pandas-and-why-use-it)
2. [Series — The 1D Structure](#2-series--the-1d-structure)
3. [DataFrame — The 2D Structure](#3-dataframe--the-2d-structure)
4. [Loading and Saving Data](#4-loading-and-saving-data)
5. [Exploring Your Data](#5-exploring-your-data)
6. [Selecting and Filtering](#6-selecting-and-filtering)
7. [Adding and Modifying Columns](#7-adding-and-modifying-columns)
8. [Sorting](#8-sorting)
9. [Cleaning Data — Handling Missing Values](#9-cleaning-data--handling-missing-values)
10. [Cleaning Data — Duplicates and Types](#10-cleaning-data--duplicates-and-types)
11. [String Operations](#11-string-operations)
12. [Datetime Operations](#12-datetime-operations)
13. [Groupby and Aggregation](#13-groupby-and-aggregation)
14. [Merging, Joining and Concatenating](#14-merging-joining-and-concatenating)
15. [Pivot Tables and Reshaping](#15-pivot-tables-and-reshaping)
16. [Apply, Map and Lambda Functions](#16-apply-map-and-lambda-functions)
17. [Plotting with Pandas](#17-plotting-with-pandas)
18. [Performance Tips](#18-performance-tips)

---

## 1. What is Pandas and Why Use It?

Pandas provides two core data structures — `Series` (1D) and `DataFrame` (2D) — that make working with labeled, tabular data as natural as working with a spreadsheet, but with the full power of Python.

### What Pandas gives you

- Read data from CSV, Excel, JSON, SQL, HTML, and more
- Clean and transform messy real-world data
- Filter, group, aggregate, join datasets
- Time series analysis
- Integrates seamlessly with NumPy, Matplotlib, and Scikit-learn

### Setup

```python
import pandas as pd
import numpy as np

# Check version
print(pd.__version__)
```

---

## 2. Series — The 1D Structure

A `Series` is a one-dimensional labeled array. Think of it as a column with an index.

### Creating a Series

```python
# From a list — index defaults to 0, 1, 2, ...
s = pd.Series([10, 20, 30, 40, 50])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# 4    50
# dtype: int64

# With a custom index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
# a    10
# b    20
# c    30

# From a dictionary (keys become index)
s = pd.Series({'python': 90, 'java': 75, 'c++': 60})
print(s)
# python    90
# java      75
# c++       60

# From a scalar (fills all positions)
s = pd.Series(42, index=['x', 'y', 'z'])
# x    42
# y    42
# z    42
```

### Accessing Series values

```python
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

# By label (index name)
s['a']        # 10
s[['a', 'c']] # returns Series with 'a' and 'c'

# By position
s.iloc[0]     # 10
s.iloc[-1]    # 40
s.iloc[1:3]   # 'b' and 'c'

# Attribute access (if index is a valid Python identifier)
s.a           # 10
```

### Series operations

```python
s = pd.Series([1, 2, 3, 4, 5])

# All NumPy-style operations work
s * 2          # 2 4 6 8 10
s + 10         # 11 12 13 14 15
s ** 2         # 1 4 9 16 25
s[s > 3]       # 4, 5

# Alignment on index — missing = NaN
s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['b', 'c', 'd'])
s1 + s2        # a=NaN, b=22, c=53, d=NaN

# Statistics
s.sum()        # 15
s.mean()       # 3.0
s.std()        # 1.58
s.min()        # 1
s.max()        # 5
s.describe()   # summary stats

# Value operations
s.unique()          # array of unique values
s.nunique()         # count of unique values
s.value_counts()    # frequency of each value
s.sort_values()     # sort by value
s.sort_index()      # sort by index
```

---

## 3. DataFrame — The 2D Structure

A `DataFrame` is a 2D table — rows and columns, each column being a `Series` sharing the same index.

### Creating a DataFrame

```python
# From a dictionary (most common)
df = pd.DataFrame({
    'name':   ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
    'age':    [25, 30, 35, 28, 22],
    'salary': [55000, 72000, 85000, 61000, 48000],
    'dept':   ['Eng', 'Mkt', 'Eng', 'Mkt', 'HR'],
    'active': [True, True, True, False, True]
})

# From a list of dicts (each dict = one row)
df = pd.DataFrame([
    {'name': 'Alice', 'age': 25, 'city': 'NYC'},
    {'name': 'Bob',   'age': 30, 'city': 'LA'},
    {'name': 'Carol', 'age': 35},   # missing 'city' → NaN
])

# From a NumPy array
arr = np.random.randint(0, 100, (5, 3))
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# From a CSV (covered in section 4)
df = pd.read_csv('file.csv')

# Empty DataFrame with defined columns
df = pd.DataFrame(columns=['name', 'age', 'salary'])
```

### DataFrame structure

```python
df = pd.DataFrame({
    'name':   ['Alice', 'Bob', 'Carol'],
    'age':    [25, 30, 35],
    'salary': [55000, 72000, 85000]
})

df.columns    # Index(['name', 'age', 'salary'])
df.index      # RangeIndex(start=0, stop=3, step=1)
df.dtypes     # name: object, age: int64, salary: int64
df.values     # NumPy array of all values
df.shape      # (3, 3)
df.size       # 9  (total elements)
df.ndim       # 2
```

---

## 4. Loading and Saving Data

### Reading files

```python
# CSV — most common
df = pd.read_csv('data.csv')

# With options
df = pd.read_csv('data.csv',
    sep=';',                      # different separator (e.g., semicolon)
    header=0,                     # row number to use as column names
    index_col='id',               # use this column as the index
    usecols=['name', 'age'],      # only load specific columns
    nrows=100,                    # only load first 100 rows
    skiprows=[1, 2],              # skip specific rows
    dtype={'age': int, 'salary': float},   # force column types
    parse_dates=['date', 'created_at'],    # auto-parse as datetime
    na_values=['?', 'N/A', 'none'],        # treat as NaN
    encoding='utf-8',             # file encoding
    thousands=',',                # '1,000' → 1000
    decimal='.',                  # decimal separator
)

# Excel
df = pd.read_excel('data.xlsx')
df = pd.read_excel('data.xlsx', sheet_name='Sheet2')
df = pd.read_excel('data.xlsx', sheet_name=0)   # first sheet

# JSON
df = pd.read_json('data.json')

# From clipboard (great for pasting from spreadsheets!)
df = pd.read_clipboard()

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM employees', conn)

# HTML table on a webpage
tables = pd.read_html('https://example.com/table-page')
df = tables[0]   # first table found

# From a URL directly
df = pd.read_csv('https://raw.githubusercontent.com/..../data.csv')

# Parquet (fast columnar format)
df = pd.read_parquet('data.parquet')
```

### Saving files

```python
# CSV
df.to_csv('output.csv', index=False)         # index=False avoids saving the row numbers
df.to_csv('output.csv', sep='\t')            # tab-separated
df.to_csv('output.csv', columns=['name', 'age'])  # only these columns

# Excel
df.to_excel('output.xlsx', index=False)
df.to_excel('output.xlsx', sheet_name='Results', index=False)

# Multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)

# JSON
df.to_json('output.json', orient='records')  # list of records

# Parquet
df.to_parquet('output.parquet', index=False)

# SQL
df.to_sql('table_name', conn, if_exists='replace', index=False)
```

---

## 5. Exploring Your Data

Always do these steps after loading new data.

```python
df = pd.read_csv('employees.csv')

# --- Shape and size ---
df.shape              # (rows, cols)  e.g. (1000, 8)
len(df)               # number of rows
df.columns.tolist()   # list of column names

# --- First/last rows ---
df.head()             # first 5 rows (default)
df.head(10)           # first 10 rows
df.tail(3)            # last 3 rows
df.sample(5)          # 5 random rows
df.sample(frac=0.1)   # 10% random sample

# --- Data types and memory ---
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1000 entries, 0 to 999
# Data columns (total 4 columns):
#  #  Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   name     1000 non-null   object
#  1   age      980 non-null    float64  ← 20 missing values!
#  2   salary   1000 non-null   int64
#  3   dept     1000 non-null   object

df.dtypes             # column data types
df.memory_usage(deep=True)  # memory per column

# --- Statistics ---
df.describe()
# count, mean, std, min, 25%, 50%, 75%, max for numeric cols

df.describe(include='all')           # include non-numeric
df.describe(include=['object'])      # only string columns
df['salary'].describe()              # for one column

# --- Missing data ---
df.isnull().sum()          # count NaN per column
df.isnull().mean() * 100   # % missing per column
df.isnull().any(axis=1)    # True for rows with any NaN
df.notnull().all(axis=1)   # True for complete rows

# --- Unique values ---
df['dept'].unique()           # array of unique values
df['dept'].nunique()          # count of unique values
df['dept'].value_counts()     # frequency table (sorted)
df['dept'].value_counts(normalize=True)  # as proportions
df['dept'].value_counts(dropna=False)    # include NaN count

# --- Correlations ---
df.corr()                          # correlation matrix (numeric cols)
df['salary'].corr(df['age'])       # correlation between two cols
```

---

## 6. Selecting and Filtering

### Column selection

```python
# Single column → returns Series
df['name']
df.name           # dot notation (only works if name is a valid Python identifier)

# Multiple columns → returns DataFrame
df[['name', 'salary']]
df[['name', 'age', 'dept']]
```

### Row selection with `.loc` (label-based)

```python
# .loc[row_selector, col_selector]
# row_selector: index label, list, slice, or boolean array
# col_selector: column name, list, or slice

df.loc[0]                     # row with index label 0
df.loc[2]                     # row with index label 2
df.loc[[0, 2, 4]]             # rows with labels 0, 2, 4
df.loc[0:3]                   # rows 0 THROUGH 3 (INCLUSIVE, unlike Python!)
df.loc[0, 'name']             # single value
df.loc[0, ['name', 'age']]    # row 0, specific columns
df.loc[:, 'age']              # all rows, 'age' column
df.loc[:, 'age':'salary']     # all rows, columns age through salary
df.loc[0:2, 'name':'dept']    # rows 0-2, columns name through dept
```

### Row selection with `.iloc` (position-based)

```python
# .iloc[row_pos, col_pos]  — always uses integers, end is EXCLUSIVE

df.iloc[0]           # first row
df.iloc[-1]          # last row
df.iloc[[0, 2, 4]]   # rows at positions 0, 2, 4
df.iloc[0:3]         # rows 0, 1, 2  (3 is EXCLUDED)
df.iloc[0, 1]        # row 0, col 1
df.iloc[:, 0]        # all rows, first column
df.iloc[0:3, 1:4]    # rows 0-2, cols 1-3
df.iloc[-5:]         # last 5 rows
```

> **loc vs iloc summary:**  
> `.loc` uses **labels** (index values, column names) — stop is **inclusive**  
> `.iloc` uses **integers** (positions 0, 1, 2...) — stop is **exclusive**

### Boolean filtering

```python
# Condition creates a boolean Series
df[df['salary'] > 60000]
df[df['dept'] == 'Eng']
df[df['active'] == True]
df[df['name'] != 'Bob']

# Multiple conditions — use & (and), | (or), ~ (not)
# MUST use parentheses around each condition!
df[(df['salary'] > 60000) & (df['dept'] == 'Eng')]
df[(df['age'] < 30) | (df['salary'] > 80000)]
df[~(df['dept'] == 'HR')]    # all except HR

# .query() — cleaner syntax for complex filters
df.query('salary > 60000')
df.query('dept == "Eng"')
df.query('age > 25 and salary > 60000')
df.query('dept in ["Eng", "HR"]')
df.query('salary > @threshold')    # reference a Python variable with @

# .isin() — check membership
df[df['dept'].isin(['Eng', 'Mkt'])]
df[~df['dept'].isin(['HR'])]       # exclude HR

# .between() — range filter (inclusive)
df[df['age'].between(25, 35)]
df[df['salary'].between(50000, 80000)]

# String methods in filters
df[df['name'].str.startswith('A')]
df[df['name'].str.contains('ali', case=False)]
```

---

## 7. Adding and Modifying Columns

```python
df = pd.DataFrame({
    'name':   ['Alice', 'Bob', 'Carol'],
    'salary': [55000, 72000, 85000],
    'age':    [25, 30, 35]
})

# Add a new column
df['bonus'] = df['salary'] * 0.10
df['total'] = df['salary'] + df['bonus']
df['senior'] = df['age'] >= 30              # boolean column
df['constant'] = 100                         # scalar broadcasts

# Modify existing column
df['salary'] = df['salary'] * 1.05          # 5% raise
df['name'] = df['name'].str.upper()

# Conditional column with np.where
df['level'] = np.where(df['salary'] > 70000, 'Senior', 'Junior')

# Multiple conditions with np.select
conditions = [
    df['salary'] < 60000,
    df['salary'].between(60000, 80000),
    df['salary'] > 80000
]
choices = ['Junior', 'Mid', 'Senior']
df['level'] = np.select(conditions, choices, default='Unknown')

# rename columns
df.rename(columns={'name': 'full_name', 'salary': 'annual_salary'}, inplace=True)

# rename all columns at once
df.columns = ['full_name', 'annual_salary', 'age', 'bonus', 'total', 'senior', 'constant', 'level']

# drop columns
df.drop(columns=['constant', 'bonus'], inplace=True)
# or equivalently:
df = df.drop(columns=['constant'])

# reorder columns
df = df[['name', 'dept', 'salary', 'age']]   # select only these, in this order

# insert column at specific position
df.insert(2, 'city', ['NYC', 'LA', 'Chicago'])
```

---

## 8. Sorting

```python
df = pd.DataFrame({
    'name':   ['Alice', 'Bob', 'Carol', 'Dave'],
    'age':    [25, 30, 25, 28],
    'salary': [55000, 72000, 85000, 61000],
    'dept':   ['Eng', 'Mkt', 'Eng', 'Mkt']
})

# Sort by a single column
df.sort_values('salary')                      # ascending (default)
df.sort_values('salary', ascending=False)     # descending

# Sort by multiple columns
df.sort_values(['dept', 'salary'])            # by dept, then salary within dept
df.sort_values(['dept', 'salary'],
               ascending=[True, False])       # dept asc, salary desc

# Sort by index
df.sort_index()
df.sort_index(ascending=False)

# inplace vs returning new
df.sort_values('salary', inplace=True)        # modifies df
new_df = df.sort_values('salary')             # returns new sorted df

# nlargest / nsmallest — get top/bottom N rows
df.nlargest(3, 'salary')        # top 3 by salary
df.nsmallest(2, 'age')          # bottom 2 by age
df.nlargest(3, ['salary', 'age'])  # sort by multiple cols

# reset index after sorting
df.sort_values('salary').reset_index(drop=True)
```

---

## 9. Cleaning Data — Handling Missing Values

Missing data in Pandas is represented as `NaN` (Not a Number) for numeric columns, or `None`/`NaN` for objects.

### Detecting missing values

```python
df.isnull()              # boolean DataFrame, True where NaN
df.notnull()             # opposite
df.isnull().sum()        # count NaN per column
df.isnull().sum(axis=1)  # count NaN per row
df.isnull().any()        # True if column has ANY NaN
df.isnull().all()        # True if column is ALL NaN

# Find rows with any missing data
df[df.isnull().any(axis=1)]

# Percentage missing
(df.isnull().sum() / len(df)) * 100
```

### Dropping missing values

```python
# Drop rows with ANY NaN
df.dropna()

# Drop rows where a SPECIFIC column is NaN
df.dropna(subset=['age'])
df.dropna(subset=['age', 'salary'])   # drop if EITHER is NaN

# Drop rows only if ALL values are NaN
df.dropna(how='all')

# Keep rows with at least N non-NaN values
df.dropna(thresh=3)   # keep row only if >= 3 non-null values

# Drop COLUMNS with any NaN
df.dropna(axis=1)

# inplace
df.dropna(inplace=True)
```

### Filling missing values

```python
# Fill all NaN with a single value
df.fillna(0)
df.fillna('unknown')
df.fillna(False)

# Fill specific columns differently
df.fillna({'age': 0, 'salary': df['salary'].mean(), 'name': 'Unknown'})

# Fill with column statistics
df['age'].fillna(df['age'].mean())
df['age'].fillna(df['age'].median())
df['dept'].fillna(df['dept'].mode()[0])   # most common value

# Forward fill — propagate last valid value forward
df.fillna(method='ffill')    # fill NaN with the value above it
df['salary'].ffill()         # forward fill a column

# Backward fill — propagate next valid value backward
df.fillna(method='bfill')

# Interpolate — for time series / sequential data
df['temperature'].interpolate()           # linear interpolation
df['temperature'].interpolate(method='polynomial', order=2)

# Limit — only fill a certain number of consecutive NaNs
df.fillna(method='ffill', limit=2)
```

---

## 10. Cleaning Data — Duplicates and Types

### Handling duplicates

```python
# Find duplicate rows
df.duplicated()                     # True for duplicate rows
df.duplicated(subset=['name'])      # duplicated based on 'name' only
df.duplicated(keep='first')         # keep=first: marks all but first as dup
df.duplicated(keep='last')          # keep=last: marks all but last
df.duplicated(keep=False)           # marks ALL copies as duplicates

# Count duplicates
df.duplicated().sum()

# View duplicated rows
df[df.duplicated()]

# Drop duplicates
df.drop_duplicates()                     # drop all duplicate rows
df.drop_duplicates(subset=['name'])      # drop based on 'name' column
df.drop_duplicates(subset=['name', 'dept'])
df.drop_duplicates(keep='last')          # keep the last occurrence
df.drop_duplicates(inplace=True)
```

### Type conversion

```python
df.dtypes   # inspect current types

# Convert to numeric
df['salary'] = df['salary'].astype(int)
df['salary'] = df['salary'].astype(float)
df['price'] = pd.to_numeric(df['price'])
df['price'] = pd.to_numeric(df['price'], errors='coerce')  # NaN on failure

# Convert to string
df['code'] = df['code'].astype(str)

# Convert to boolean
df['active'] = df['active'].astype(bool)

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'], errors='coerce')   # NaN on failure

# Convert to category (saves memory for low-cardinality string cols)
df['dept'] = df['dept'].astype('category')
df['dept'].cat.categories    # view categories
df['dept'].cat.codes         # numeric code for each category

# Replace values
df.replace('N/A', np.nan)
df.replace({'dept': {'Eng': 'Engineering', 'Mkt': 'Marketing'}})
df['salary'].replace(0, np.nan)
```

---

## 11. String Operations

Access string methods via the `.str` accessor on a Series.

```python
s = pd.Series(['Alice Smith', '  Bob Jones  ', 'carol Brown', 'DAVE WILSON'])

# Case
s.str.upper()             # 'ALICE SMITH', ...
s.str.lower()             # 'alice smith', ...
s.str.title()             # 'Alice Smith', ...

# Strip whitespace
s.str.strip()             # remove leading/trailing spaces
s.str.lstrip()            # remove leading only
s.str.rstrip()            # remove trailing only

# Check contents (returns boolean Series)
s.str.contains('Smith')                  # case-sensitive
s.str.contains('smith', case=False)      # case-insensitive
s.str.contains(r'\d+', regex=True)       # regex: contains digits
s.str.startswith('A')
s.str.endswith('n')
s.str.isdigit()
s.str.isalpha()
s.str.isnumeric()

# Extract and modify
s.str.replace('Smith', 'S.')
s.str.replace(r'\s+', ' ', regex=True)       # collapse multiple spaces
s.str.split(' ')                              # list of words
s.str.split(' ', expand=True)                # split into separate columns
s.str.split(' ').str[0]                      # first word only
s.str.get(0)                                 # first character

# Length and padding
s.str.len()
s.str.pad(width=20, side='right', fillchar='-')
s.str.zfill(5)                               # zero-pad (for numbers as strings)

# Slicing
s.str[0:5]              # first 5 characters
s.str[-3:]              # last 3 characters

# Extract with regex
emails = pd.Series(['alice@example.com', 'bob@gmail.com'])
emails.str.extract(r'(\w+)@(\w+\.\w+)')     # extract groups
emails.str.extractall(r'(\w+)')              # extract all matches
emails.str.findall(r'\w+')                   # find all matches

# Count occurrences
s.str.count('l')         # count of 'l' in each string

# Join
parts = pd.Series([['hello', 'world'], ['foo', 'bar']])
parts.str.join('-')      # 'hello-world', 'foo-bar'
```

---

## 12. Datetime Operations

```python
# Parse dates
df['date'] = pd.to_datetime(df['date'])

# Create date range
pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')   # daily
pd.date_range(start='2024-01-01', periods=12, freq='ME')         # monthly end
pd.date_range(start='2024-01-01', periods=52, freq='W')          # weekly

# Access components via .dt accessor
df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
df['date'].dt.hour
df['date'].dt.minute
df['date'].dt.second
df['date'].dt.day_of_week    # 0=Monday, 6=Sunday
df['date'].dt.day_name()     # 'Monday', 'Tuesday', ...
df['date'].dt.month_name()   # 'January', ...
df['date'].dt.quarter        # 1, 2, 3, or 4
df['date'].dt.is_month_end   # boolean
df['date'].dt.is_month_start

# Date arithmetic
df['date'] + pd.Timedelta(days=7)      # add 7 days
df['date'] - pd.Timedelta(weeks=1)     # subtract 1 week
df['end_date'] - df['start_date']      # timedelta between dates

# Time offsets
df['date'] + pd.DateOffset(months=1)   # add 1 month
df['date'] + pd.DateOffset(years=1)    # add 1 year

# Filtering by date
df[df['date'] > '2024-06-01']
df[df['date'].between('2024-01-01', '2024-06-30')]
df[df['date'].dt.year == 2024]
df[df['date'].dt.month == 3]           # all March rows

# Resample — aggregate by time period (requires datetime index)
df = df.set_index('date')
df.resample('ME').sum()         # monthly totals
df.resample('W').mean()         # weekly averages
df.resample('Q').max()          # quarterly max
df.resample('h').count()        # hourly counts

# Rolling window
df['sales'].rolling(7).mean()   # 7-day moving average
df['sales'].rolling(30).sum()   # 30-day rolling total
df['sales'].expanding().mean()  # cumulative mean
```

---

## 13. Groupby and Aggregation

`groupby` is one of Pandas' most powerful features. The pattern is: **split → apply → combine**.

### Basic groupby

```python
df = pd.DataFrame({
    'name':   ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank'],
    'dept':   ['Eng', 'Mkt', 'Eng', 'Mkt', 'HR', 'Eng'],
    'city':   ['NYC', 'NYC', 'LA', 'LA', 'NYC', 'LA'],
    'salary': [80000, 72000, 95000, 61000, 48000, 88000],
    'age':    [25, 30, 35, 28, 22, 32]
})

# Group by one column, apply one aggregation
df.groupby('dept')['salary'].mean()
# dept
# Eng    87666.67
# HR     48000.00
# Mkt    66500.00

df.groupby('dept')['salary'].sum()
df.groupby('dept')['salary'].max()
df.groupby('dept')['salary'].count()
df.groupby('dept')['salary'].std()

# Multiple columns
df.groupby('dept')[['salary', 'age']].mean()
df.groupby('dept').size()       # number of rows per group
```

### Multiple aggregations with `.agg()`

```python
# Multiple aggregations on one column
df.groupby('dept')['salary'].agg(['mean', 'min', 'max', 'count'])

# Different aggregations per column
df.groupby('dept').agg({
    'salary': ['mean', 'max', 'std'],
    'age':    ['mean', 'min', 'max'],
    'name':   'count'
})

# Custom aggregations with lambda
df.groupby('dept')['salary'].agg(
    avg_salary='mean',
    highest='max',
    range=lambda x: x.max() - x.min()
)

# Named aggregations (cleaner column names)
df.groupby('dept').agg(
    avg_salary=('salary', 'mean'),
    max_age=('age', 'max'),
    headcount=('name', 'count')
)
```

### Group by multiple columns

```python
df.groupby(['dept', 'city'])['salary'].mean()
df.groupby(['dept', 'city']).agg({'salary': 'mean', 'age': 'count'})

# Access a specific group
grouped = df.groupby('dept')
grouped.get_group('Eng')       # all Eng rows

# Iterate over groups
for dept_name, group_df in df.groupby('dept'):
    print(f"\n{dept_name}:")
    print(group_df)
```

### `transform` — keep original shape

```python
# transform returns a Series with same length as the original df
# Perfect for adding computed columns back

df['dept_avg_salary'] = df.groupby('dept')['salary'].transform('mean')
df['dept_total'] = df.groupby('dept')['salary'].transform('sum')
df['salary_rank'] = df.groupby('dept')['salary'].transform('rank')

# Salary above or below dept average?
df['above_avg'] = df['salary'] > df.groupby('dept')['salary'].transform('mean')

# Normalize salary within department
df['salary_norm'] = df.groupby('dept')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

### `filter` — include or exclude entire groups

```python
# Keep only departments with more than 2 employees
df.groupby('dept').filter(lambda x: len(x) > 2)

# Keep only groups where max salary > 80000
df.groupby('dept').filter(lambda x: x['salary'].max() > 80000)
```

---

## 14. Merging, Joining and Concatenating

### `pd.merge` — SQL-style joins

```python
employees = pd.DataFrame({
    'emp_id':  [1, 2, 3, 4],
    'name':    ['Alice', 'Bob', 'Carol', 'Dave'],
    'dept_id': [10, 20, 10, 30]
})
departments = pd.DataFrame({
    'dept_id':   [10, 20, 40],
    'dept_name': ['Engineering', 'Marketing', 'Finance']
})

# INNER join — only matching rows in both (default)
pd.merge(employees, departments, on='dept_id')
# Alice, Bob, Carol — Dave is excluded (dept_id 30 not in departments)

# LEFT join — all rows from left, NaN where no match on right
pd.merge(employees, departments, on='dept_id', how='left')
# All 4 employees, Dave gets NaN for dept_name

# RIGHT join — all rows from right, NaN where no match on left
pd.merge(employees, departments, on='dept_id', how='right')
# Engineering, Marketing, Finance — Finance has no employees (NaN)

# OUTER join — all rows from both
pd.merge(employees, departments, on='dept_id', how='outer')

# Join on columns with different names
pd.merge(employees, departments,
         left_on='dept_id', right_on='id')

# Join on multiple columns
pd.merge(df1, df2, on=['name', 'date'])

# Suffix when column names clash
pd.merge(df1, df2, on='id', suffixes=('_left', '_right'))
```

### `pd.concat` — stacking DataFrames

```python
# Stack vertically (add rows) — same columns
df_all = pd.concat([df_jan, df_feb, df_mar])

# Reset index (avoid duplicate index values)
df_all = pd.concat([df_jan, df_feb], ignore_index=True)

# Add a key to identify each source
df_all = pd.concat([df1, df2], keys=['file1', 'file2'])

# Stack horizontally (add columns) — same rows
df_wide = pd.concat([df1, df2], axis=1)

# Handling mismatched columns
df_all = pd.concat([df1, df2], join='outer')   # NaN for missing (default)
df_all = pd.concat([df1, df2], join='inner')   # only common columns
```

### `.join` — index-based merge

```python
df1 = pd.DataFrame({'salary': [80000, 72000]}, index=['Alice', 'Bob'])
df2 = pd.DataFrame({'age': [25, 30]}, index=['Alice', 'Bob'])

df1.join(df2)         # join on index
df1.join(df2, how='left')
```

---

## 15. Pivot Tables and Reshaping

### Pivot table

```python
df = pd.DataFrame({
    'dept':    ['Eng', 'Eng', 'Mkt', 'Mkt', 'HR'],
    'city':    ['NYC', 'LA',  'NYC', 'LA',  'NYC'],
    'salary':  [80000, 95000, 72000, 61000, 48000],
    'headcount': [5, 3, 4, 2, 3]
})

# Like Excel pivot tables
pd.pivot_table(df,
    values='salary',        # what to aggregate
    index='dept',           # row grouping
    columns='city',         # column grouping
    aggfunc='mean',         # aggregation function
    fill_value=0            # replace NaN with 0
)
#       city        LA       NYC
# dept
# Eng          95000.0   80000.0
# HR               0.0   48000.0
# Mkt          61000.0   72000.0

# Multiple values
pd.pivot_table(df,
    values=['salary', 'headcount'],
    index='dept',
    columns='city',
    aggfunc={'salary': 'mean', 'headcount': 'sum'}
)
```

### `melt` — wide to long format

```python
# Wide format (one row per person, multiple time columns)
wide = pd.DataFrame({
    'name':  ['Alice', 'Bob'],
    'Jan':   [100, 200],
    'Feb':   [150, 250],
    'Mar':   [120, 230]
})

# Convert to long (tidy) format
long = pd.melt(wide,
    id_vars=['name'],              # columns to keep
    value_vars=['Jan', 'Feb', 'Mar'],  # columns to melt
    var_name='month',              # name for the new "variable" column
    value_name='sales'             # name for the new "value" column
)
#     name month  sales
# 0  Alice   Jan    100
# 1    Bob   Jan    200
# 2  Alice   Feb    150
# ...
```

### `stack` and `unstack`

```python
# unstack — move row index level to columns
df.set_index(['dept', 'city'])['salary'].unstack()

# stack — move column level to row index
df.stack()     # columns become row labels
```

### `crosstab` — frequency table

```python
pd.crosstab(df['dept'], df['city'])           # counts
pd.crosstab(df['dept'], df['city'],
            values=df['salary'], aggfunc='mean')  # with values
pd.crosstab(df['dept'], df['city'], normalize='index')  # row percentages
```

---

## 16. Apply, Map and Lambda Functions

### `.apply()` — apply a function to rows or columns

```python
df = pd.DataFrame({
    'salary': [55000, 72000, 85000],
    'tax_rate': [0.2, 0.25, 0.3]
})

# Apply to a column (element-wise for Series)
df['salary'].apply(lambda x: x * 1.1)       # 10% raise
df['salary'].apply(lambda x: f'${x:,}')     # format as string

# Apply a named function
def categorize(salary):
    if salary < 60000:
        return 'Junior'
    elif salary < 80000:
        return 'Mid'
    else:
        return 'Senior'

df['level'] = df['salary'].apply(categorize)

# Apply across rows (axis=1) — each row is passed as a Series
df['take_home'] = df.apply(
    lambda row: row['salary'] * (1 - row['tax_rate']),
    axis=1
)

# Apply across columns (axis=0 default) — returns aggregate per column
df.apply('mean')         # mean of each column
df.apply(lambda x: x.max() - x.min())   # range of each column
```

### `.map()` — element-wise mapping on Series

```python
s = pd.Series(['Eng', 'Mkt', 'HR', 'Eng'])

# Map with a dictionary (like a lookup/replace)
dept_map = {'Eng': 'Engineering', 'Mkt': 'Marketing', 'HR': 'Human Resources'}
s.map(dept_map)

# Map with a function
s.map(lambda x: x.lower())
s.map(len)              # length of each string

# Missing values in dict → NaN
s.map({'Eng': 'Engineering'})   # Mkt and HR → NaN
```

### `.applymap()` / `.map()` on DataFrame — element-wise

```python
# Apply a function to every single element in a DataFrame
df.map(lambda x: round(x, 2) if isinstance(x, float) else x)
df.map(str)     # convert everything to string
```

### Vectorized operations vs apply — performance

```python
# SLOW — use apply only when necessary
df['salary'].apply(lambda x: x * 1.1)

# FAST — use vectorized operations instead
df['salary'] * 1.1

# apply is needed when:
# 1. Your logic is too complex for vectorized operations
# 2. You need to use multiple columns per row (axis=1)
# 3. You're calling an external function
```

---

## 17. Plotting with Pandas

Pandas has built-in plotting that wraps Matplotlib. Always import Matplotlib too.

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'month':    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'revenue':  [12000, 15000, 11000, 18000, 21000, 17000],
    'expenses': [8000, 9000, 10000, 11000, 13000, 12000]
})

# Line plot
df.plot(x='month', y='revenue', figsize=(10, 5))
df.plot(x='month', y=['revenue', 'expenses'])  # multiple lines
plt.title('Monthly Revenue')
plt.show()

# Bar chart
df.plot.bar(x='month', y='revenue')
df.plot.barh(x='month', y='revenue')     # horizontal
df.plot.bar(x='month', stacked=True)     # stacked bars

# Scatter
df.plot.scatter(x='expenses', y='revenue')

# Histogram
df['revenue'].plot.hist(bins=20)

# Box plot
df[['revenue', 'expenses']].plot.box()

# Pie chart
df['revenue'].plot.pie()

# Area chart
df.plot.area(x='month', y=['revenue', 'expenses'])

# Plot directly on a Series
s = pd.Series([1, 4, 9, 16, 25])
s.plot()
s.plot.bar()

# Correlation heatmap (with seaborn)
import seaborn as sns
numeric_df = df[['revenue', 'expenses']]
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

## 18. Performance Tips

### Use vectorized operations

```python
# SLOW
for i, row in df.iterrows():
    df.at[i, 'bonus'] = row['salary'] * 0.1

# FAST
df['bonus'] = df['salary'] * 0.1
```

### Use efficient types

```python
# Check memory usage
df.memory_usage(deep=True)

# Convert string columns with few unique values to category
df['dept'] = df['dept'].astype('category')    # can save up to 90% memory

# Use smaller integer/float types when possible
df['age'] = df['age'].astype('int8')          # age fits in int8 (-128 to 127)
df['salary'] = df['salary'].astype('float32') # less precision, less memory
```

### Prefer query/loc over chained indexing

```python
# SLOW and may cause SettingWithCopyWarning
df[df['dept'] == 'Eng']['salary'] = 90000  # BAD

# CORRECT
df.loc[df['dept'] == 'Eng', 'salary'] = 90000  # GOOD
```

### Avoid iterrows — use vectorized or apply

```python
# SLOW (Python-speed loop)
for i, row in df.iterrows():
    ...

# MUCH FASTER
df.apply(func, axis=1)       # still Python, but optimized
df['col'] * 2                # fully vectorized (fastest)
```

### Read large files efficiently

```python
# Read in chunks for huge files
chunk_size = 100_000
chunks = []
for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    # process each chunk
    chunk = chunk[chunk['salary'] > 50000]
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# Use parquet for fast I/O on large datasets
df.to_parquet('data.parquet')            # save
df = pd.read_parquet('data.parquet')     # load (10–100x faster than CSV)
```

---

## 19. GroupBy — Advanced Patterns

### `groupby().apply()` — run any custom function on each group

`agg` returns one row per group. `transform` returns the same shape as the input. `apply` is the most flexible — it can return **anything**: a scalar, a Series, or a whole DataFrame.

```python
df = pd.DataFrame({
    'dept':   ['Eng','Eng','Eng','Sales','Sales','HR'],
    'name':   ['Priya','Rohan','Anjali','Vikram','Sneha','Karthik'],
    'salary': [95000, 80000, 110000, 72000, 65000, 48000],
    'age':    [28, 32, 26, 35, 29, 41]
})

# Example 1 — return a scalar per group (same as agg)
df.groupby('dept')['salary'].apply(lambda g: g.max() - g.min())
# dept
# Eng      30000
# HR           0
# Sales     7000

# Example 2 — return a filtered sub-DataFrame per group
# Get top 1 earner from each department
def top_earner(group_df):
    return group_df.nlargest(1, 'salary')

df.groupby('dept').apply(top_earner, include_groups=False)

# Example 3 — return a transformed DataFrame (custom logic)
def normalize_salary(group_df):
    group_df = group_df.copy()
    group_df['norm_salary'] = (
        (group_df['salary'] - group_df['salary'].mean())
        / group_df['salary'].std()
    )
    return group_df

df.groupby('dept').apply(normalize_salary, include_groups=False)

# Example 4 — compute something that needs multiple columns
def salary_to_age_ratio(group_df):
    return (group_df['salary'] / group_df['age']).mean()

df.groupby('dept').apply(salary_to_age_ratio)
```

> **When to use which:**
> - `agg` → reduce each group to one summary value per column
> - `transform` → add group stats back to original DataFrame (same shape)
> - `apply` → anything else — full custom logic, multiple columns, returning DataFrames

---

### `pd.NamedAgg` — explicit named aggregations

A cleaner alternative to the dict style for naming output columns precisely.

```python
result = df.groupby('dept').agg(
    total_employees  = pd.NamedAgg(column='name',   aggfunc='count'),
    avg_salary       = pd.NamedAgg(column='salary', aggfunc='mean'),
    max_salary       = pd.NamedAgg(column='salary', aggfunc='max'),
    salary_range     = pd.NamedAgg(column='salary', aggfunc=lambda x: x.max() - x.min()),
    youngest_age     = pd.NamedAgg(column='age',    aggfunc='min'),
)
print(result)
#        total_employees  avg_salary  max_salary  salary_range  youngest_age
# dept
# Eng                  3    95000.0      110000         30000            26
# HR                   1    48000.0       48000             0            41
# Sales                2    68500.0       72000          7000            29
```

---

### `agg` with custom user-defined functions

```python
# Define reusable custom aggregation functions
def salary_range(series):
    return series.max() - series.min()

def coeff_of_variation(series):
    """std / mean — measures relative variability"""
    return series.std() / series.mean()

def above_median_count(series):
    """How many values are above the group median"""
    return (series > series.median()).sum()

# Use them just like built-in functions
df.groupby('dept')['salary'].agg([
    'mean',
    'std',
    salary_range,          # custom function
    coeff_of_variation,    # custom function
    above_median_count,    # custom function
])

# Mix built-ins and custom in named agg:
df.groupby('dept').agg(
    avg_sal   = ('salary', 'mean'),
    sal_range = ('salary', salary_range),
    cv        = ('salary', coeff_of_variation),
)
```

---

## 20. Pivot Tables — Advanced

### `pivot_table` with multiple `aggfunc`

```python
sales = pd.DataFrame({
    'region':   ['North','North','South','South','East'],
    'product':  ['Laptop','Mouse','Laptop','Keyboard','Mouse'],
    'quantity': [2, 5, 1, 3, 4],
    'revenue':  [180000, 6000, 90000, 12000, 5000],
    'discount': [10, 0, 5, 0, 15],
})

# Multiple aggfuncs — returns MultiIndex columns
pd.pivot_table(sales,
    values=['revenue', 'quantity'],
    index='region',
    columns='product',
    aggfunc={'revenue': 'sum', 'quantity': 'mean'},
    fill_value=0
)

# Multiple aggfuncs as a list — same func applied to all values
pd.pivot_table(sales,
    values='revenue',
    index='region',
    columns='product',
    aggfunc=['sum', 'mean', 'count'],
    fill_value=0
)
```

### `pivot_table` with `margins=True` — automatic grand totals

```python
pivot = pd.pivot_table(sales,
    values='revenue',
    index='region',
    columns='product',
    aggfunc='sum',
    fill_value=0,
    margins=True,          # adds "All" row and column
    margins_name='Total'   # custom label instead of "All"
)
print(pivot)
#          product  Keyboard  Laptop   Mouse   Total
# region
# East                    0       0    5000    5000
# North                   0  180000    6000  186000
# South               12000   90000       0  102000
# Total               12000  270000   11000  293000
```

---

## 21. Crosstab — All Variations

`pd.crosstab` is a specialised pivot for **frequency tables**. It has more options than most people use.

### Basic count crosstab

```python
import pandas as pd

df = pd.DataFrame({
    'dept':      ['Eng','Eng','Sales','Sales','HR','Eng','Sales'],
    'level':     ['Senior','Junior','Mid','Senior','Junior','Mid','Junior'],
    'gender':    ['M','F','M','F','F','M','M'],
    'city':      ['Mumbai','Delhi','Mumbai','Bangalore','Delhi','Mumbai','Delhi'],
})

# Count of each combination
pd.crosstab(df['dept'], df['level'])
#        level  Junior  Mid  Senior
# dept
# Eng         1    1       1
# HR          1    0       0
# Sales       1    1       1
```

### `normalize` — all four options

```python
# normalize='index'   → row percentages (each row sums to 1.0)
pd.crosstab(df['dept'], df['level'], normalize='index').round(2)

# normalize='columns' → column percentages (each column sums to 1.0)
pd.crosstab(df['dept'], df['level'], normalize='columns').round(2)

# normalize='all'     → % of grand total (entire table sums to 1.0)
pd.crosstab(df['dept'], df['level'], normalize='all').round(3)

# No normalize        → raw counts (default)
pd.crosstab(df['dept'], df['level'])
```

### `margins=True` — row and column totals

```python
pd.crosstab(df['dept'], df['level'],
            margins=True,
            margins_name='Total')
#        level  Junior  Mid  Senior  Total
# dept
# Eng         1    1       1       3
# HR          1    0       0       1
# Sales       1    1       1       3
# Total       3    2       2       7
```

### `values` + `aggfunc` — aggregate a third column

```python
salaries = pd.DataFrame({
    'dept':   ['Eng','Eng','Sales','Sales','HR','Eng'],
    'level':  ['Senior','Junior','Mid','Senior','Junior','Mid'],
    'salary': [110000, 60000, 75000, 90000, 45000, 80000],
})

# Average salary per dept × level
pd.crosstab(salaries['dept'], salaries['level'],
            values=salaries['salary'],
            aggfunc='mean').round(0)

# Count AND sum together using a dict
pd.crosstab(salaries['dept'], salaries['level'],
            values=salaries['salary'],
            aggfunc=['mean', 'count'])
```

### Three-way crosstab — multiple row levels

```python
# Two index columns — dept AND gender as row index
pd.crosstab([df['dept'], df['gender']], df['level'])
#                 level  Junior  Mid  Senior
# dept  gender
# Eng   F              1    0       0
#       M              0    1       1
# HR    F              1    0       0
# Sales M              1    1       1
```

---

## 22. Window Functions — shift, diff, pct_change, rolling, expanding

These are essential for time-series and sequential analysis.

```python
sales_daily = pd.DataFrame({
    'date':    pd.date_range('2024-01-01', periods=8, freq='D'),
    'revenue': [12000, 15000, 11000, 18000, 21000, 17000, 22000, 19000],
    'units':   [10, 12, 9, 15, 17, 14, 18, 16],
})
```

### `shift` — look at previous or next rows

```python
# Shift down by 1 — brings yesterday's value into today's row
sales_daily['prev_day_revenue'] = sales_daily['revenue'].shift(1)
# Row 0 gets NaN (no previous day)

# Shift up by 1 — brings tomorrow's value into today's row
sales_daily['next_day_revenue'] = sales_daily['revenue'].shift(-1)

# Shift by 7 — same day last week
sales_daily['last_week_revenue'] = sales_daily['revenue'].shift(7)
```

### `diff` — difference from previous row

```python
# Day-over-day change in revenue (absolute)
sales_daily['revenue_change'] = sales_daily['revenue'].diff()
# diff(1) = current - previous

# diff(n) — compare to n periods ago
sales_daily['revenue_change_3d'] = sales_daily['revenue'].diff(3)
```

### `pct_change` — percentage change from previous row

```python
# Day-over-day % change
sales_daily['revenue_growth_pct'] = (
    sales_daily['revenue'].pct_change() * 100
).round(2)

# Compare to 3 periods ago
sales_daily['revenue_growth_3d'] = (
    sales_daily['revenue'].pct_change(periods=3) * 100
).round(2)

print(sales_daily[['date','revenue','revenue_growth_pct']])
#         date  revenue  revenue_growth_pct
# 0 2024-01-01    12000                 NaN
# 1 2024-01-02    15000               25.00
# 2 2024-01-03    11000              -26.67
# 3 2024-01-04    18000               63.64
```

### `rolling` — sliding window calculations

```python
# 3-day rolling average (smooths out daily noise)
sales_daily['rolling_3d_avg'] = sales_daily['revenue'].rolling(window=3).mean()

# 3-day rolling sum
sales_daily['rolling_3d_sum'] = sales_daily['revenue'].rolling(window=3).sum()

# Rolling max and min
sales_daily['rolling_3d_max'] = sales_daily['revenue'].rolling(3).max()
sales_daily['rolling_3d_min'] = sales_daily['revenue'].rolling(3).min()

# Rolling std — volatility measure
sales_daily['rolling_3d_std'] = sales_daily['revenue'].rolling(3).std()

# min_periods — compute even when window isn't full yet
sales_daily['rolling_avg_partial'] = sales_daily['revenue'].rolling(
    window=3, min_periods=1
).mean()

# Rolling on grouped data (e.g., rolling avg per product)
df_products = pd.DataFrame({
    'product': ['A','A','A','B','B','B'],
    'month':   [1, 2, 3, 1, 2, 3],
    'revenue': [100, 150, 120, 200, 180, 220],
})
df_products['rolling_avg'] = (
    df_products.groupby('product')['revenue']
    .transform(lambda x: x.rolling(2, min_periods=1).mean())
)
```

### `expanding` — cumulative from the start

```python
# Cumulative average (expands from row 0 to current row)
sales_daily['cumulative_avg'] = sales_daily['revenue'].expanding().mean()

# Cumulative sum (running total)
sales_daily['cumulative_sum'] = sales_daily['revenue'].expanding().sum()

# Cumulative max — running highest value seen so far
sales_daily['running_max'] = sales_daily['revenue'].expanding().max()

print(sales_daily[['date','revenue','cumulative_avg','running_max']])
```

---

## 23. Method Chaining — `assign` and `pipe`

### `assign` — add columns without breaking the chain

`assign` returns a new DataFrame with added/modified columns, making it perfect for chaining.

```python
df = pd.DataFrame({
    'name':   ['Priya','Rohan','Anjali','Vikram'],
    'salary': [95000, 80000, 110000, 72000],
    'age':    [28, 32, 26, 35],
    'dept':   ['Eng','Eng','Eng','Sales'],
})

# Without assign — messy multi-step mutations
df['salary_lpa']  = df['salary'] / 100000
df['senior']      = df['age'] >= 30
df['tax']         = df['salary'] * 0.30

# With assign — clean, readable chain
result = (df
    .assign(salary_lpa  = lambda x: x['salary'] / 100000)
    .assign(senior      = lambda x: x['age'] >= 30)
    .assign(tax         = lambda x: x['salary'] * 0.30)
    .assign(take_home   = lambda x: x['salary'] - x['tax'])
    .sort_values('salary', ascending=False)
    .reset_index(drop=True)
)

# Multiple assigns in one call (columns can reference each other left to right)
result = df.assign(
    salary_lpa = lambda x: x['salary'] / 100000,
    tax        = lambda x: x['salary'] * 0.30,
    take_home  = lambda x: x['salary'] - x['salary'] * 0.30,
)
```

### `pipe` — plug custom functions into a chain

`pipe` lets you pass the whole DataFrame through a function, keeping the chain unbroken.

```python
# Define reusable transformation functions
def add_salary_band(df):
    df = df.copy()
    df['salary_band'] = pd.cut(df['salary'],
        bins=[0, 600000, 1500000, float('inf')],
        labels=['Entry','Mid','Senior'])
    return df

def flag_high_performers(df, threshold=4.0):
    df = df.copy()
    df['high_performer'] = df['performance_score'] >= threshold
    return df

def remove_inactive(df):
    return df[df['status'] == 'Active'].reset_index(drop=True)

# Chain them all with pipe — reads like a pipeline
clean_df = (employees
    .pipe(remove_inactive)
    .pipe(add_salary_band)
    .pipe(flag_high_performers, threshold=4.2)
    .assign(email = lambda x: x['name'].str.lower().str.replace(' ', '.') + '@company.in')
    .sort_values('salary', ascending=False)
    .reset_index(drop=True)
)
```

> `pipe` is especially powerful when you have a library of transformation functions you want to mix and match across projects.

---

## 24. `explode` — one row per list element

When a column contains **lists**, `explode` splits each list into separate rows.

```python
df = pd.DataFrame({
    'emp_id':  [1001, 1002, 1003],
    'name':    ['Priya', 'Rohan', 'Anjali'],
    'skills':  [
        ['Python', 'SQL', 'Tableau'],
        ['Java', 'AWS'],
        ['Python', 'ML', 'Docker', 'Kubernetes']
    ]
})

print(df)
#    emp_id    name                             skills
# 0    1001   Priya          [Python, SQL, Tableau]
# 1    1002   Rohan                      [Java, AWS]
# 2    1003  Anjali  [Python, ML, Docker, Kubernetes]

# Explode — one row per skill
exploded = df.explode('skills').reset_index(drop=True)
print(exploded)
#    emp_id    name      skills
# 0    1001   Priya      Python
# 1    1001   Priya         SQL
# 2    1001   Priya     Tableau
# 3    1002   Rohan        Java
# 4    1002   Rohan         AWS
# 5    1003  Anjali      Python
# 6    1003  Anjali          ML
# 7    1003  Anjali      Docker
# 8    1003  Anjali  Kubernetes

# Now you can do groupby on skills!
exploded.groupby('skills')['emp_id'].count().sort_values(ascending=False)
# Python    2
# AWS       1
# Docker    1
# ...

# Real-world: reading from CSV where skills column has comma-separated values
df['skills'] = df['skills_str'].str.split(',')
df.explode('skills')
```

---

## 25. MultiIndex — Working with Hierarchical Indices

When you `groupby` multiple columns, the result has a **MultiIndex**. Here's how to navigate it.

```python
df = pd.DataFrame({
    'dept':   ['Eng','Eng','Eng','Sales','Sales','HR'],
    'level':  ['Senior','Junior','Senior','Mid','Senior','Junior'],
    'city':   ['Mumbai','Delhi','Bangalore','Mumbai','Delhi','Mumbai'],
    'salary': [110000, 60000, 105000, 75000, 92000, 48000],
})

# groupby two columns → MultiIndex result
multi = df.groupby(['dept','level'])['salary'].mean()
print(multi)
# dept   level
# Eng    Junior     60000.0
#        Senior    107500.0
# HR     Junior     48000.0
# Sales  Mid        75000.0
#        Senior     92000.0

# ── Accessing MultiIndex data ────────────────────────────────────────────────

# Access by outer index only
multi['Eng']
# level
# Junior     60000.0
# Senior    107500.0

# Access by both levels
multi['Eng']['Senior']    # 107500.0
multi['Eng', 'Senior']    # 107500.0  (tuple indexing)

# .xs — cross-section (cleaner for inner levels)
multi.xs('Senior', level='level')        # all Senior rows
multi.xs('Eng',    level='dept')         # all Eng rows

# ── Flattening a MultiIndex ──────────────────────────────────────────────────

# unstack — inner index becomes columns
wide = multi.unstack(level='level')
print(wide)
# level       Junior     Senior
# dept
# Eng        60000.0   107500.0
# HR         48000.0        NaN
# Sales          NaN    92000.0

# reset_index — flatten to regular columns
flat = multi.reset_index()
# dept  level   salary
# Eng   Junior  60000.0
# ...

# Flatten MultiIndex column names (after pivot/unstack)
pivot = df.groupby(['dept','level'])['salary'].agg(['mean','max']).reset_index()
# If columns are MultiIndex after pivot:
pivot.columns = ['_'.join(col).strip('_') for col in pivot.columns]
```

---

## Quick Reference Card

```python
# ── CREATING ─────────────────────────────────────────
pd.Series([1,2,3], index=['a','b','c'])
pd.DataFrame({'col1': [...], 'col2': [...]})
pd.read_csv('f.csv')  pd.read_excel('f.xlsx')

# ── EXPLORING ────────────────────────────────────────
df.head()  df.tail()  df.sample(n)
df.shape   df.info()  df.describe()
df.dtypes  df.isnull().sum()  df['col'].value_counts()

# ── SELECTING ────────────────────────────────────────
df['col']           df[['a','b']]
df.loc[0, 'col']    df.loc[0:3, 'a':'c']
df.iloc[0, 1]       df.iloc[0:3, 1:4]
df[df['col'] > 5]   df.query('col > 5')
df[df['col'].isin(['x','y'])]

# ── ADDING/MODIFYING ─────────────────────────────────
df['new'] = df['a'] * 2
df['level'] = np.where(df['sal']>60000, 'Senior', 'Junior')
df.rename(columns={'old': 'new'})
df.drop(columns=['col'])

# ── CLEANING ─────────────────────────────────────────
df.isnull().sum()         df.dropna()
df.fillna(0)              df.fillna(df['col'].mean())
df.drop_duplicates()      df['col'].astype(int)
df.replace('?', np.nan)

# ── STRINGS ──────────────────────────────────────────
df['col'].str.lower()   .str.strip()   .str.contains('x')
df['col'].str.split(' ', expand=True)  .str.extract(r'(\w+)')

# ── GROUPBY ──────────────────────────────────────────
df.groupby('col')['val'].mean()
df.groupby('col').agg({'val': ['mean','max']})
df.groupby('col')['val'].transform('mean')

# ── MERGING ──────────────────────────────────────────
pd.merge(df1, df2, on='key', how='left')
pd.concat([df1, df2], ignore_index=True)

# ── RESHAPING ────────────────────────────────────────
pd.pivot_table(df, values='v', index='r', columns='c')
pd.melt(df, id_vars=['name'], value_vars=['Jan','Feb'])

# ── SAVING ───────────────────────────────────────────
df.to_csv('f.csv', index=False)
df.to_excel('f.xlsx', index=False)

# ── ADVANCED GROUPBY ─────────────────────────────────
df.groupby('col').apply(custom_fn)               # full custom logic
df.groupby('col').agg(alias=pd.NamedAgg('col','mean'))
df.groupby('col')['val'].agg(custom_function)    # user-defined function

# ── PIVOT TABLE ADVANCED ─────────────────────────────
pd.pivot_table(df, values='v', index='r', columns='c',
               aggfunc=['sum','mean'], margins=True, margins_name='Total')

# ── CROSSTAB FULL OPTIONS ────────────────────────────
pd.crosstab(df['a'], df['b'])                          # count
pd.crosstab(df['a'], df['b'], normalize='index')       # row %
pd.crosstab(df['a'], df['b'], normalize='columns')     # col %
pd.crosstab(df['a'], df['b'], normalize='all')         # grand total %
pd.crosstab(df['a'], df['b'], margins=True)            # with totals
pd.crosstab(df['a'], df['b'], values=df['v'], aggfunc='mean')  # aggregate

# ── WINDOW FUNCTIONS ─────────────────────────────────
df['col'].shift(1)                   # previous row value
df['col'].diff(1)                    # difference from previous row
df['col'].pct_change()               # % change from previous row
df['col'].rolling(7).mean()          # 7-row sliding average
df['col'].rolling(7, min_periods=1).mean()  # allow partial windows
df['col'].expanding().mean()         # cumulative average

# ── METHOD CHAINING ───────────────────────────────────
df.assign(new_col=lambda x: x['a'] * 2)             # add col in chain
df.pipe(my_function)                                 # pass df to function
df.pipe(my_function, extra_arg=value)                # with arguments

# ── EXPLODE ───────────────────────────────────────────
df['list_col'] = df['str_col'].str.split(',')
df.explode('list_col').reset_index(drop=True)

# ── MULTIINDEX ────────────────────────────────────────
multi = df.groupby(['a','b'])['val'].mean()
multi['outer_key']                   # select outer level
multi['outer', 'inner']              # select both levels
multi.xs('inner_val', level='b')     # cross-section
multi.unstack(level='b')             # inner index → columns
multi.reset_index()                  # flatten to regular df
```

---

*End of Pandas Masterclass*
