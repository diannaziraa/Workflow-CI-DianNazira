import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =====================================
# LOAD DATASET PREPROCESSING
# =====================================

print("=" * 50)
print("MEMUAT DATASET PREPROCESSING")
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
# ENABLE AUTOLOG
# =====================================

mlflow.autolog()

# =====================================
# TRAIN MODEL
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
# EVALUASI
# =====================================

score = model.score(
    X_test,
    y_test
)

print("\n===== HASIL EVALUASI =====")
print(f"Accuracy : {score:.4f}")

print("\nTraining selesai")