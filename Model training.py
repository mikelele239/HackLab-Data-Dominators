# Churn Prediction Pipeline (Google Colab‑ready)
# =================================================
# ✨ Enhancements (May 2025)
# • **Removed** hard 95 % accuracy goal – the pipeline now always reports metrics without threshold‑gating.
# • **NEW** Top‑3 churn drivers for every scored customer, using per‑row SHAP‑style contributions from XGBoost
#   (leveraging `predict(..., pred_contribs=True)`).
# • Drivers + their signed impact are appended to the output DataFrame returned by `test_model_on_new()`.
# -------------------------------------------------
# 0️⃣  Install & Imports
# -------------------------------------------------
!pip -q install scikit-learn xgboost joblib  # add shap if you prefer full SHAP plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb  # needed for pred_contribs

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
)
from xgboost import XGBClassifier

# -------------------------------------------------
# 1️⃣  Upload your training CSV
# -------------------------------------------------
print("🔼  Please choose the pre‑processed dataset CSV (includes the target column).")
try:
    from google.colab import files
    uploaded = files.upload()
    if not uploaded:
        raise ValueError("No file uploaded – execution stopped.")
    DATA_PATH = next(iter(uploaded))
except ModuleNotFoundError:
    # If not in Colab, fall back to a local file path
    DATA_PATH = "Data final.csv"
    print(f"Colab not detected – using {DATA_PATH}")

# -------------------------------------------------
# 2️⃣  Load data + basic setup
# -------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded data → {df.shape[0]:,} rows, {df.shape[1]} columns")

TARGET_COL = "Churn"          # <-- EDIT if your target has a different name
ID_COLS     = ["customerID"]  # <-- Drop ID‑like columns as features

# Binary‑encode the target if needed
y = df[TARGET_COL].map({"Yes": 1, "No": 0}) if df[TARGET_COL].dtype == "O" else df[TARGET_COL]
X = df.drop(ID_COLS + [TARGET_COL], axis=1)

# Detect feature types
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
print(f"📊 Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}")

# -------------------------------------------------
# 3️⃣  Preprocessing pipeline
# -------------------------------------------------
numeric_transformer = Pipeline([
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols),
])

# -------------------------------------------------
# 4️⃣  Train / test split (stratified)
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"🔹 Train: {X_train.shape}, 🔸 Test: {X_test.shape}")

# -------------------------------------------------
# 5️⃣  Model & hyper‑parameter search (XGBoost)
# -------------------------------------------------
param_grid = {
    "learning_rate":  [0.01, 0.05, 0.1],
    "max_depth":      [3, 4, 5, 6, 8],
    "n_estimators":   [300, 500, 800],
    "subsample":      [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "gamma":          [0, 1, 5],
}

xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_grid,
    n_iter=30,
    scoring="accuracy",
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)

clf = Pipeline([
    ("pre", preprocessor),
    ("model", search),
])

print("🚀 Training (this may take a few minutes)…")
clf.fit(X_train, y_train)

best_model: XGBClassifier = clf.named_steps["model"].best_estimator_
print(f"🏆 Best XGB params: {clf.named_steps['model'].best_params_}")

# -------------------------------------------------
# 6️⃣  Evaluation (no hard accuracy threshold)
# -------------------------------------------------
print("\n===== Evaluation =====")

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc      = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_proba)
cm       = confusion_matrix(y_test, y_pred)
report   = classification_report(y_test, y_pred)

print(f"Accuracy : {acc:.4f}")
print(f"ROC‑AUC  : {roc_auc:.4f}\n")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", cm)

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve – XGBoost Churn Classifier")
plt.show()

# Feature importance (top 20 – global)
enc = clf.named_steps["pre"].named_transformers_["cat"].named_steps["encoder"]
encoded_cat_features = enc.get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(encoded_cat_features)

importances = best_model.feature_importances_
imp_series = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
imp_series[::-1].plot(kind="barh")
plt.title("Top‑20 Feature Importances (XGBoost)")
plt.xlabel("Importance score")
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 7️⃣  Save the trained pipeline
# -------------------------------------------------
joblib.dump(clf, "churn_pipeline.pkl")
print("💾 Pipeline saved as churn_pipeline.pkl (download via Files sidebar)")

# -------------------------------------------------
# 🔍 Helper functions for per‑customer drivers
# -------------------------------------------------

def _get_top_drivers(raw_df: pd.DataFrame, model_pipeline: Pipeline, top_n: int = 3):
    """Return lists of top‑N driver names & their signed impacts for each row in raw_df."""

    # Decompose pipeline
    pre = model_pipeline.named_steps["pre"]
    booster: XGBClassifier = model_pipeline.named_steps["model"].best_estimator_

    # Build full feature name list (needs to mirror training order)
    enc = pre.named_transformers_["cat"].named_steps["encoder"]
    encoded_cat_features = enc.get_feature_names_out(cat_cols)
    all_feats = num_cols + list(encoded_cat_features)

    # Apply same preprocessing
    X_trans = pre.transform(raw_df)

    # Compute per‑row contributions (SHAP values) via XGBoost booster
    dmat = xgb.DMatrix(X_trans, feature_names=all_feats)
    contribs = booster.get_booster().predict(dmat, pred_contribs=True)  # shape: (n_samples, n_features + 1)

    driver_names = []
    driver_impacts = []

    for row in contribs:
        row_contribs = row[:-1]  # exclude bias term
        top_idx = np.argsort(np.abs(row_contribs))[::-1][:top_n]
        driver_names.append([all_feats[i] for i in top_idx])
        driver_impacts.append([float(row_contribs[i]) for i in top_idx])

    return driver_names, driver_impacts

# -------------------------------------------------
# 8️⃣  Interactively test on new data – returns drivers
# -------------------------------------------------

def test_model_on_new():
    """Upload a CSV *without* the target column ⇒ get churn predictions + top‑3 drivers."""
    try:
        from google.colab import files
        print("🔼  Upload a CSV with the same feature columns you used for training (no target column)…")
        new_upload = files.upload()
        if not new_upload:
            print("No file uploaded.")
            return None
        new_path = next(iter(new_upload))
    except ModuleNotFoundError:
        print("Colab not detected – provide a file path instead of uploading.")
        return None

    new_df = pd.read_csv(new_path)

    preds = clf.predict(new_df)
    probs = clf.predict_proba(new_df)[:, 1]

    # --- Top‑3 churn drivers ---
    names_list, impacts_list = _get_top_drivers(new_df, clf, top_n=3)

    out = new_df.copy()
    out["PredictedChurn"] = preds
    out["ChurnProbability"] = probs

    # Unpack drivers into separate columns for clarity
    for i in range(3):
        out[f"Driver{i+1}"] = [names[i] if len(names) > i else None for names in names_list]
        out[f"Impact{i+1}"] = [impacts[i] if len(impacts) > i else None for impacts in impacts_list]

    print("\n🔍  Predictions with top‑3 churn drivers (first 10 rows):")
    display(out.head(10))
    return out

# ➡️  After running all cells:
# result_df = test_model_on_new()
