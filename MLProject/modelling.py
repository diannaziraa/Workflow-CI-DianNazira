import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# =====================================
# LOAD DATASET
# =====================================

print("=" * 50)
print("MEMUAT DATASET")
print("=" * 50)

df = pd.read_csv("telco_churn_preprocessing.csv")

print(f"Shape Dataset: {df.shape}")

# =====================================
# FEATURE & TARGET
# =====================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# AUTOLOG
# =====================================

mlflow.autolog()

# =====================================
# TRAINING
# =====================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# =====================================
# PREDIKSI
# =====================================

y_pred = model.predict(X_test)

# =====================================
# EVALUASI
# =====================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n===== HASIL EVALUASI =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nTraining selesai")