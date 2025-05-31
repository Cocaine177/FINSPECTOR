# train_model.py
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Ensure Models directory exists
os.makedirs("Models", exist_ok=True)

# Step 1: Load your dataset
try:
    df = pd.read_csv("final_dataset.csv")
    print("📂 Loaded 'final_dataset.csv'")
except FileNotFoundError:
    # Create a dummy dataset if none exists
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['anomaly'] = y
    df.to_csv("final_dataset.csv", index=False)
    print("⚠️ Created dummy 'final_dataset.csv' with random data.")

# Step 2: Prepare features and target
X = df.drop(columns=["anomaly"])
y = df["anomaly"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Save the model and feature names
joblib.dump(model, "Models/rf_stock_model.pkl")
joblib.dump(X.columns.tolist(), "Models/feature_names.pkl")
print("\n✅ Model and feature names saved in 'Models/' folder")
