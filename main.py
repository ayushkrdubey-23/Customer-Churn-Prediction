import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from xgboost import XGBClassifier

# -----------------------------
# STEP 1: CREATE OUTPUT FOLDER
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# STEP 2: LOAD DATA
# -----------------------------
df = pd.read_csv("data/churn.csv")

print("Dataset Loaded Successfully\n")
print(df.head())

# -----------------------------
# STEP 3: DATA CLEANING
# -----------------------------
# Drop customerID if exists
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing values
df = df.dropna()

# Convert target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# STEP 4: SPLIT DATA
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 5: XGBOOST MODEL
# -----------------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# -----------------------------
# STEP 6: PREDICTION
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------
# STEP 7: EVALUATION
# -----------------------------
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", acc)
print("\nROC-AUC Score:", roc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save metrics
with open("outputs/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"ROC-AUC: {roc}\n\n")
    f.write(classification_report(y_test, y_pred))

# -----------------------------
# STEP 8: SAVE PREDICTIONS
# -----------------------------
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Churn_Probability": y_prob
})

results.to_csv("outputs/predictions.csv", index=False)

# -----------------------------
# STEP 9: HIGH-RISK CUSTOMERS
# -----------------------------
high_risk = results[results["Churn_Probability"] > 0.7]
high_risk.to_csv("outputs/high_risk_customers.csv", index=False)

print(f"\nHigh Risk Customers: {len(high_risk)}")

# -----------------------------
# STEP 10: FEATURE IMPORTANCE
# -----------------------------
importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(14,6))
plt.bar(feature_names, importances)
plt.xticks(rotation=90)
plt.title("Feature Importance")

plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.show()

print("\nAll outputs saved in 'outputs' folder")


#Save trained model
# Create models folder if not exists
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")




# Save model, scaler, and feature names
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

print(" Model, scaler, features saved")
print("Model saved successfully")

