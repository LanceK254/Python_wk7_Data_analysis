# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# -------------------------------
# Task 1: Load and Explore the Dataset
# -------------------------------

# Load the Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!\n")
except Exception as e:
    print("Error loading dataset:", e)

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head(), "\n")

# Check data types and missing values
print("Dataset Info:")
print(df.info(), "\n")

print("Missing values:")
print(df.isnull().sum(), "\n")

# Clean the dataset (no missing values in this dataset)
# But for example:
# df.fillna(df.mean(), inplace=True)

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

# Basic statistics
print("Basic Statistical Summary:")
print(df.describe(), "\n")

# Grouping by species and calculating mean
group_means = df.groupby('species').mean()
print("Mean values per species:")
print(group_means, "\n")

# Observations:
print("Observation:")
print("→ Iris-virginica tends to have the largest petal and sepal dimensions on average.\n")

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

sns.set(style="whitegrid")

# Line Chart (using sepal length averages to simulate trend)
plt.figure(figsize=(8, 5))
df_sorted = df.sort_values(by=iris.feature_names[0])
plt.plot(df_sorted[iris.feature_names[0]].values, label='Sepal Length Trend', color='green')
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Index (sorted by sepal length)")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# Bar Chart (average petal length per species)
plt.figure(figsize=(6, 4))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None, palette='Set2')
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram (distribution of sepal width)
plt.figure(figsize=(6, 4))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='skyblue')
plt.title("Histogram: Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot (sepal length vs petal length)
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='Set1')
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()
