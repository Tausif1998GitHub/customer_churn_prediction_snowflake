# snowflake_churn_ml_v2.py
# Author: Tausif | POC for AI/ML Engineer Assignment

from snowflake.snowpark import Session
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# =============== 1️⃣ Snowflake connection config ===================
connection_parameters = {
    "account": "vg01336.ap-south-1.aws",   # your Snowflake account name
    "user": "TAUSIF",
    "password": "XXXXXXXX",  # replace safely or use env var
    "role": "ACCOUNTADMIN",
    "warehouse": "POC_WH",
    "database": "POC_DB",
    "schema": "PUBLIC"
}

session = Session.builder.configs(connection_parameters).create()

# =============== 2️⃣ Load data =====================================
# Option A: Directly from Snowflake
table_name = "CUSTOMER_CHURN"
try:
    sdf = session.table(table_name)
    df = sdf.to_pandas()
except Exception as e:
    print("⚠️ Could not fetch from Snowflake, loading local CSV instead...")
    df = pd.read_csv("customer_churn.csv")

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]
print("✅ Data loaded:", df.shape)
print(df.head())

# =============== 3️⃣ Data Cleaning =================================
df = df.dropna()

# =============== 4️⃣ Feature Engineering ============================
df["is_fiber"] = (df["internet_service"] == "Fiber optic").astype(int)
df["is_monthly_contract"] = (df["contract_type"] == "Month-to-month").astype(int)
df = pd.get_dummies(df, columns=["gender", "payment_method"], drop_first=True)

# =============== 5️⃣ Train/Test Split ===============================
features = [
    "age", "tenure_months", "monthly_charges", "num_support_calls",
    "is_fiber", "is_monthly_contract"
] + [c for c in df.columns if c.startswith("gender_") or c.startswith("payment_method_")]

X = df[features]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============== 6️⃣ Train Model ===================================
clf = LogisticRegression(max_iter=300)
clf.fit(X_train, y_train)

# =============== 7️⃣ Evaluate ======================================
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))

# Save model locally
joblib.dump(clf, "snowflake_churn_model.joblib")

# =============== 8️⃣ Save Predictions to Snowflake ==================
results = pd.DataFrame({
    "churn_actual": y_test,
    "churn_pred": y_pred,
    "churn_prob": y_prob
})

# Create table if not exists
session.sql("""
    CREATE OR REPLACE TABLE CHURN_PREDICTIONS (
        CHURN_ACTUAL INT,
        CHURN_PRED INT,
        CHURN_PROB FLOAT
    );
""").collect()

# Fix casing + index
results.columns = [c.upper() for c in results.columns]
results.reset_index(drop=True, inplace=True)

# Write cleanly
session.write_pandas(results, "CHURN_PREDICTIONS")

print("✅ Predictions uploaded to Snowflake table: CHURN_PREDICTIONS")

# =============== 9️⃣ Verify from Python =============================
preview = session.table("CHURN_PREDICTIONS").limit(5).to_pandas()
print("\n🔍 Preview from Snowflake:\n", preview)

session.close()
print("\n🎯 End-to-end POC completed successfully.")
