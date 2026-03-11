"""
Hackathon Template (EDA + Train/Validation Split)

Purpose
-------
This script provides a reusable, dataset-agnostic template for:
1) Loading a tabular dataset from a CSV file
2) Performing fast, practical EDA (Exploratory Data Analysis)
3) Creating a train/validation split suitable for hackathons

How to use:
-------------------
1) SET The Configuration variables to fit the dataset
2) Run the file. You will get:
   - Shape / head / dtypes / columns
   - Missing values percentage
   - Duplicate count
   - Target distribution
   - Numeric summary
   - Categorical value counts (top K)
   - Train/validation split summary
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
import re
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

TARGET = "readmitted"  # Change this to whatever the Target is in your dataset
#=======Helper FUnctions=======
def icd9_group(code):
    if pd.isna(code):
        return "Missing"
    s = str(code).strip()
    if s in ["?", ""]:
        return "Missing"
    if s.startswith("V"):
        return "V_Code"
    if s.startswith("E"):
        return "E_Code"
    m = re.match(r"^(\d+)", s)
    if not m:
        return "Other"
    x = int(m.group(1))

    if x == 250: return "Diabetes"
    if 390 <= x <= 459: return "Circulatory"
    if 460 <= x <= 519: return "Respiratory"
    if 520 <= x <= 579: return "Digestive"
    if 580 <= x <= 629: return "Genitourinary"
    if 140 <= x <= 239: return "Neoplasms"
    if 240 <= x <= 279: return "Endocrine"
    if 290 <= x <= 319: return "Mental"
    if 320 <= x <= 389: return "Neuro_Sense"
    if 800 <= x <= 999: return "Injury"
    return "Other"

def engineer_features_like_train(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the SAME feature engineering you used on the training data."""
    df = df.copy()

    # '?' -> NaN (same as train) :contentReference[oaicite:2]{index=2}
    df = df.replace(r"^\s*\?\s*$", np.nan, regex=True)

    # Treat ID codes as categorical strings (same as train) :contentReference[oaicite:3]{index=3}
    ID_CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
    for c in ID_CODE_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # diag grouping + drop raw diag cols (same as train) :contentReference[oaicite:4]{index=4}
    for k in [1, 2, 3]:
        col = f"diag_{k}"
        if col in df.columns:
            df[f"{col}_grp"] = df[col].apply(icd9_group)

    df = df.drop(columns=[c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns], errors="ignore")

    # measured flags (same as train) :contentReference[oaicite:5]{index=5}
    for col in ["weight", "A1Cresult", "max_glu_serum"]:
        if col in df.columns:
            df[col + "_measured"] = df[col].notna().astype(int)

    return df


def fill_final_target_from_model(
    final_csv_path: str,
    trained_pipeline,
    train_feature_cols,
    target_col: str = TARGET,
    out_csv_path: str | None = None,
) -> pd.DataFrame:

    """
    Load FINAL csv (no target), do feature engineering, align columns to training X,
    predict with trained pipeline, and add target column to the final dataframe.
    """
    df_final_raw = pd.read_csv(final_csv_path)

    # Apply same feature engineering
    df_final = engineer_features_like_train(df_final_raw)

    # Build X_final exactly like training X (same drops you used) :contentReference[oaicite:6]{index=6}
    DROP_COLS_FINAL = ["weight", "max_glu_serum", "A1Cresult",
                      "encounter_id", "patient_nbr", "examide", "citoglipton", target_col]
    X_final = df_final.drop(columns=DROP_COLS_FINAL, errors="ignore")

    # Ensure FINAL has all columns seen in training (missing -> NaN), drop extras, order same
    for c in train_feature_cols:
        if c not in X_final.columns:
            X_final[c] = np.nan
    X_final = X_final.drop(columns=[c for c in X_final.columns if c not in train_feature_cols], errors="ignore")
    X_final = X_final[train_feature_cols]

    # Predict using the trained pipeline (preprocess + model)
    preds = trained_pipeline.predict(X_final)

    # Fill/attach predictions into a copy of the ORIGINAL final file
    df_out = df_final_raw.copy()
    df_out[target_col] = preds

    # Optional probabilities (if model supports predict_proba)
    if hasattr(trained_pipeline, "predict_proba"):
        df_out[target_col + "_proba_yes"] = trained_pipeline.predict_proba(X_final)[:, 1]

    if out_csv_path:
        df_out.to_csv(out_csv_path, index=False)

    return df_out

# ---Display Options---
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ---CONFIGS---
DATA_PATH = r"*************************************"

TOP_MISSING_N = 50  # How many columns to show for missing % check
TOP_CAT_N = 10  # How many most frequent values per categorical column

def plot_model_diagnostics(model, X_test, y_test, title: str) -> None:
    """Plot confusion matrix + ROC + PR for a fitted sklearn classifier/pipeline."""
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, values_format="d")
    ax.set_title(f"{title} | Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ROC / PR only if proba exists
    if hasattr(model, "predict_proba"):
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title(f"{title} | ROC Curve")
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title(f"{title} | Precision-Recall Curve")
        plt.tight_layout()
        plt.show()

# Carmel

# ----------------START HERE-------------------
# ---Load Data---
data = pd.read_csv(DATA_PATH)

# Handle Missing Values in the data by replacing any missing values presented as ('*?*') in the dataset with NaN so Missing Values Check would work correctly.
data = data.replace(r"^\s*\?\s*$", np.nan, regex=True)

# ---Target: 3-class -> binary (YES=<30 days, NO otherwise)---
target_map = {"<30": "Yes", ">30": "No", "NO": "No"}
data[TARGET] = data[TARGET].map(target_map)

# Converting the target to Binary
data[TARGET] = (data[TARGET] == "Yes").astype(int)

# While they look like a numeric value they are categorical value
ID_CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
for c in ID_CODE_COLS:
    if c in data.columns:
        data[c] = data[c].astype("string")


# Compressing diagnosis codes into medical chapters to prevent
for k in [1, 2, 3]:
    col = f"diag_{k}"
    if col in data.columns:
        data[f"{col}_grp"] = data[col].apply(icd9_group)

# drop raw diagnosis columns after grouping
drop_diags = [c for c in ["diag_1","diag_2","diag_3"] if c in data.columns]
data = data.drop(columns=drop_diags)


# adding a measured flag rather than ignore the features completely.
for col in ["weight", "A1Cresult", "max_glu_serum"]:
    if col in data.columns:
        data[col + "_measured"] = data[col].notna().astype(int)

# ---EDA---
print("\nShape:")
print(data.shape)

#print("\nHead:")
#print(data.head())

#print("\ndtypes:")
#print(data.dtypes)

print("\nColumns:")
print(data.columns.tolist())

print(f"\nMissing % (Top {TOP_MISSING_N}):")
missing_pct = (data.isna().mean() * 100).sort_values(ascending=False)
print(missing_pct.head(TOP_MISSING_N))

print("\nDuplicates:")
print(data.duplicated().sum())

print("\nTarget Distribution")
print(data[TARGET].value_counts(dropna=False))
print(data[TARGET].value_counts(normalize=True, dropna=False).round(4))

print("\nNumeric Summary:")
print(data.describe(include=np.number).T)

print(f"\nCategorical Summary (top {TOP_CAT_N} cols):")
cat_columns = data.select_dtypes(include=['str', 'category']).columns.tolist()
for col in cat_columns:
    print(f"\n[{col} unique = {data[col].nunique(dropna=True)}]")
    print(data[col].value_counts(dropna=False).head(TOP_CAT_N))


X = data.drop(columns=[TARGET,'weight','max_glu_serum','A1Cresult','encounter_id','patient_nbr','examide','citoglipton'])
# Check encoding of weight,max_glu and A1C to boolean weight measured?
y = data[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


numeric_cols = X_train.select_dtypes(include=[np.number]).columns
categorical_cols = X_train.columns.difference(numeric_cols)

# Preprocessing for numeric and categorical

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),
                                      ("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[
     ("imputer", SimpleImputer(strategy="most_frequent")),
     ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))
 ])

preprocess = ColumnTransformer(
     transformers=[
         ("num", numeric_transformer, numeric_cols),
         ("cat", categorical_transformer, categorical_cols)
     ],
     remainder="drop"
 )

# Baseline Model Logistic Regression

baseline_model = Pipeline(steps=[
     ("preprocess", preprocess),
     ("model", DummyClassifier(strategy="most_frequent"))
 ])


baseline_model.fit(X_train, y_train)

baseline_model_scores = cross_val_score(
    baseline_model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro"
)

print("\nCV Macro-F1 mean:", baseline_model_scores.mean().round(4),"std:", baseline_model_scores.std().round(4))

base_model_predict = baseline_model.predict(X_test)

print("\n=== Baseline (most_frequent) ===")
print("Accuracy:", round(accuracy_score(y_test, base_model_predict), 4))
print("Macro F1:", round(f1_score(y_test, base_model_predict, average="macro"), 4))

plot_model_diagnostics(baseline_model, X_test, y_test, title="Baseline (most_frequent)")

Xt = preprocess.fit_transform(X_train, y_train)
print("Transformed shape:", Xt.shape)


log_reg_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        C=0.5
    ))
])

# Fit + predict
log_reg_model.fit(X_train, y_train)

# ===== Feature importance for Logistic Regression (by absolute coefficient) =====
pre = log_reg_model.named_steps["preprocess"]
clf = log_reg_model.named_steps["model"]

# names after ColumnTransformer (works in newer sklearn)
feature_names = pre.get_feature_names_out()
coefs = clf.coef_.ravel()

imp = pd.Series(np.abs(coefs), index=feature_names).sort_values(ascending=False)

top_k = 25
fig, ax = plt.subplots(figsize=(10, 7))
imp.head(top_k).sort_values().plot(kind="barh", ax=ax)
ax.set_title(f"LogReg | Top {top_k} Features by |coef|")
ax.set_xlabel("|coefficient|")
plt.tight_layout()
plt.show()

log_reg_val_score = cross_val_score(
    log_reg_model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro"
)
pred_test_val = log_reg_model.predict(X_test)

print("\n=== Logistic Regression ===")
print("Accuracy:", round(accuracy_score(y_test, pred_test_val), 4))
print("Macro F1:", round(f1_score(y_test, pred_test_val, average="macro"), 4))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred_test_val))
print("\nClassification report:\n", classification_report(y_test, pred_test_val))
print("\nCV Macro-F1 mean:", log_reg_val_score.mean().round(4),"std:", log_reg_val_score.std().round(4))

plot_model_diagnostics(log_reg_model, X_test, y_test, title="Logistic Regression")

# IMPORTANT: use the SAME training feature columns you trained with
TRAIN_FEATURE_COLS = X.columns.tolist()  # X is the training features dataframe :contentReference[oaicite:7]{index=7}

Final_Data_Path = r"C:\Users\Carmel\Desktop\Carmel\Codes\Python\Hackathon\health_final_exam_input.csv"
out_path = r"C:\Users\Carmel\Desktop\Carmel\Codes\Python\Hackathon\health_final_exam_filled.csv"

final_filled = fill_final_target_from_model(
    final_csv_path=Final_Data_Path,
    trained_pipeline=log_reg_model,      # the trained Pipeline(preprocess+model)
    train_feature_cols=TRAIN_FEATURE_COLS,
    target_col=TARGET,
    out_csv_path=out_path
)

print(final_filled[[TARGET]].head())
print("Saved:", out_path)


