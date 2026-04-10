import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
df = pd.read_csv("superstore.csv")

# -------------------------------
# 1. Data Cleaning
# -------------------------------
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

# Remove missing values
df = df.dropna()

# -------------------------------
# 2. Feature Engineering
# -------------------------------

# Extract new features from date
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month

# Profit Margin
df["Profit Margin"] = df["Profit"] / df["Sales"]

# -------------------------------
# 3. Encoding Categorical Data
# -------------------------------
le = LabelEncoder()

df["Category"] = le.fit_transform(df["Category"])
df["Region"] = le.fit_transform(df["Region"])
df["Segment"] = le.fit_transform(df["Segment"])

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()

df[["Sales", "Profit", "Discount", "Quantity"]] = scaler.fit_transform(
    df[["Sales", "Profit", "Discount", "Quantity"]]
)

# -------------------------------
# 5. Final Dataset
# -------------------------------
print("\nProcessed Data:")
print(df.head())

print("\nColumns:")
print(df.columns)
