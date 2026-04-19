# ==============================
# Expense Tracker Project
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Step 1: Generate Synthetic Data
# ------------------------------

np.random.seed(42)

dates = pd.date_range(start="2024-01-01", end="2024-12-31")
categories = ["Food", "Rent", "Travel", "Shopping", "Bills", "Entertainment"]
payment_methods = ["Cash", "UPI", "Card"]

data = {
    "Date": np.random.choice(dates, 500),
    "Category": np.random.choice(categories, 500),
    "Amount": np.random.randint(100, 5000, 500),
    "Payment_Method": np.random.choice(payment_methods, 500)
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Data Cleaning
# ------------------------------

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")

# ------------------------------
# Step 3: Feature Engineering
# ------------------------------

df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
# Feature Engineering

df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.weekday
df["Is_Weekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)

# Sort data
df = df.sort_values("Date")

# Rolling Average (7 days)
df["Rolling_Avg"] = df["Amount"].rolling(window=7).mean()

# Lag Feature (previous expense)
df["Lag_1"] = df["Amount"].shift(1)

df = df.dropna()

# ------------------------------
# Step 4: Analysis
# ------------------------------

category_spending = df.groupby("Category")["Amount"].sum()
monthly_spending = df.groupby("Month")["Amount"].sum()
#----------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Features & Target
X = df[["Day", "Weekday", "Is_Weekend", "Lag_1"]]
y = df["Amount"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Model Trained Successfully")

#---------------------------------------------------------------
#ANOMALY DETECTION
# Z-score
df["Z_Score"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

# Threshold
threshold = 2

df["Anomaly"] = df["Z_Score"].apply(lambda x: 1 if abs(x) > threshold else 0)

# Show anomalies
anomalies = df[df["Anomaly"] == 1]

print("Anomalies Detected:")
print(anomalies[["Date", "Amount", "Category"]])

# ------------------------------
# Step 5: Visualization
# ------------------------------

plt.figure(figsize=(10,5))
category_spending.plot(kind="bar")
plt.title("Category-wise Spending")
plt.ylabel("Amount")
plt.show()

plt.figure(figsize=(10,5))
monthly_spending.plot(kind="line", marker="o")
plt.title("Monthly Spending Trend")
plt.ylabel("Amount")
plt.show()

# Pie Chart
plt.figure(figsize=(6,6))
category_spending.plot(kind="pie", autopct="%1.1f%%")
plt.title("Expense Distribution")
plt.ylabel("")
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(x=df["Date"], y=df["Amount"], hue=df["Anomaly"])
plt.title("Anomaly Detection")
plt.show()

##----RECOMMENDATION-------------------------------

monthly_avg = df.groupby("Month")["Amount"].sum().mean()

category_avg = df.groupby("Category")["Amount"].mean()

recommended_budget = monthly_avg * 0.9  # reduce 10%

print("Average Monthly Spending:", monthly_avg)
print("Recommended Budget:", recommended_budget)

print("\nCategory Suggestions:")
print(category_avg.sort_values(ascending=False))
#---------------SMART INSIGHT
top_category = df.groupby("Category")["Amount"].sum().idxmax()

if top_category == "Food":
    print("⚠️ You are spending too much on Food. Consider cooking at home.")

elif top_category == "Shopping":
    print("⚠️ Reduce impulse purchases.")

else:
    print("✅ Spending looks balanced.")
# ------------------------------
# Step 6: Insights
# ------------------------------

print("Top Spending Category:", category_spending.idxmax())
print("Highest Spending Month:", monthly_spending.idxmax())