# scripts/main.py

import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ----------------------------------------------------------------------------
# STEP 1 — Download Dataset Automatically
# ----------------------------------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/Auto.csv"
DATA_PATH = "data/Auto.csv"

if not os.path.exists("data"):
    os.makedirs("data")

# Download only if not already downloaded
if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    response = requests.get(DATA_URL)
    with open(DATA_PATH, "wb") as f:
        f.write(response.content)
    print("Dataset downloaded successfully.")
else:
    print("Dataset already exists.")

# ----------------------------------------------------------------------------
# STEP 2 — Load Data
# ----------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# ----------------------------------------------------------------------------
# STEP 3 — Data Information & Missing Values
# ----------------------------------------------------------------------------

print("\nData Info:")
print(df.info())

print("\nChecking missing values:")
print(df.isnull().sum())

# ----------------------------------------------------------------------------
# STEP 4 — Visualization
# ----------------------------------------------------------------------------

if not os.path.exists("output"):
    os.makedirs("output")

# Pairplot of a few numeric features vs mpg
plt.figure(figsize=(10, 6))
sns.scatterplot(x="horsepower", y="mpg", data=df)
plt.title("Horsepower vs MPG")
plt.savefig("output/hp_vs_mpg.png")
plt.close()

# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("output/correlation_matrix.png")
plt.close()

print("\nVisualizations saved in output/")

# ----------------------------------------------------------------------------
# STEP 5 — Preprocessing
# ----------------------------------------------------------------------------

# Drop rows with missing values if any
df = df.dropna()

# Drop non-numeric "name" column
df = df.drop("name", axis=1)

# Features and target
X = df.drop("mpg", axis=1)
y = df["mpg"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nPreprocessing complete.")

# ----------------------------------------------------------------------------
# STEP 6 — Train/Test Split
# ----------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ----------------------------------------------------------------------------
# STEP 7 — Train Linear Regression
# ----------------------------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training completed.")

# ----------------------------------------------------------------------------
# STEP 8 — Model Evaluation
# ----------------------------------------------------------------------------

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# ----------------------------------------------------------------------------
# STEP 9 — Save Model & Predictions
# ----------------------------------------------------------------------------

# Save model
joblib.dump(model, "output/linear_regression_model.pkl")
joblib.dump(scaler, "output/scaler.pkl")
print("\nModel & scaler saved.")

# Save predictions
results_df = pd.DataFrame({
    "Actual MPG": y_test,
    "Predicted MPG": y_pred
})
results_df.to_csv("output/predictions.csv", index=False)
print("Predictions saved as output/predictions.csv")


