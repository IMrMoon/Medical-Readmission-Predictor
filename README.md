# Medical-Readmission-Predictor

This repository contains a machine learning classification project built to predict whether a patient will be readmitted to the hospital within 30 days. The prediction is based on various clinical and administrative data points.

The project is structured as a clear, easy-to-run Python script that covers the entire process:
- Quick exploratory data analysis (EDA) to find missing values, duplicates, and check data distribution.
- Practical feature engineering specific to medical data (like grouping ICD-9 diagnosis codes and creating flags for missing measurements).
- A clean preprocessing pipeline (handling missing data, scaling, and one-hot encoding).
- Model training and comparison between a baseline DummyClassifier and a Logistic Regression model.
- Performance evaluation using Accuracy, Macro F1, Confusion Matrix, and ROC/PR plots.
- A built-in utility to automatically process and predict on new, unlabelled data for final CSV submissions.

---

## Technical Decisions & Workflow

### 1. Data Cleaning
- **Handling Missing Values:** The raw dataset uses `"?"` to mark missing data. We convert these to standard `NaN` values so pandas and scikit-learn can process them correctly.
- **Target Variable:** The original target column has three states (`"<30"`, `">30"`, `"NO"`). Since the goal is early readmission detection, we simplified this into a binary format:
  - `1` = Readmitted in **<30 days**
  - `0` = Otherwise (later than 30 days, or not at all)

### 2. Feature Engineering
- **Grouping Diagnosis Codes:** Raw diagnosis codes (`diag_1`, `diag_2`, `diag_3`) have too many unique values, making it hard for the model to find patterns. We compressed these specific ICD-9 codes into broader medical categories (e.g., Diabetes, Circulatory, Respiratory) and dropped the raw columns.
- **Handling Sparse Clinical Tests ("Measured" Flags):** Some attributes like `weight`, `A1Cresult`, and `max_glu_serum` are missing for most patients. Instead of trying to guess these values, we created boolean columns (e.g., `weight_measured` = True/False). This way, we keep the valuable information of *whether a test was ordered or not*, without injecting bad data.
- **Categorical IDs:** Columns like `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` are numbers, but they represent categories, not mathematical values. We cast them to strings so the pipeline knows to one-hot encode them.

### 3. Feature Selection
We removed certain columns to keep the model focused and prevent data leakage:
- **Identifiers:** `encounter_id` and `patient_nbr` (these don't help the model learn general patterns).
- **Sparse/Irrelevant Data:** Columns that were entirely empty, mostly empty, or had no predictive value for this specific task (like `examide` and `citoglipton`).

### 4. Preprocessing Pipeline
We used a scikit-learn `ColumnTransformer` to handle different data types smoothly:

- **Numeric Pipeline:** Fills missing values with the median (`SimpleImputer`) and scales the data (`StandardScaler`).
- **Categorical Pipeline:** Fills missing values with the most common category (`SimpleImputer`) and converts categories into binary columns (`OneHotEncoder`). 
  - *Note:* We used `handle_unknown="ignore"` to prevent errors if the model sees a new category in the test set, and `min_frequency=0.01` to group extremely rare categories together.

### 5. Modeling Strategy
- **Baseline Check:** We used a `DummyClassifier(strategy="most_frequent")` to establish a baseline. It shows us what happens if a model just guesses the most common outcome every time.
- **Main Model:** We chose **Logistic Regression** because it's fast, performs well on tabular data, and is highly interpretable (you can actually look at the coefficients to see what influenced the prediction).
  - *Configuration:* `solver="lbfgs"`, `max_iter=2000`, and `C=0.5` for mild regularization.

---

## Why Accuracy Isn't Enough (Handling Class Imbalance)
During testing, a standard Logistic Regression model gave us high overall Accuracy but completely failed to identify the patients who actually returned to the hospital (the minority class). 

This is a common trap in medical datasets. To fix this, we shifted our evaluation focus to the **Macro F1 score** and enabled `class_weight="balanced"` in the model. This forces the algorithm to pay equal attention to the minority class, resulting in much better real-world predictions.

---

## Evaluation Metrics
When you run the script, it outputs:
- Accuracy & Macro F1 Score
- Detailed Classification Report (Precision, Recall, F1 for each class)
- Stratified 10-fold cross-validation scores
- Visualizations: Confusion Matrix, and optional ROC / Precision-Recall curves.

---

## Generating the Final Submission
We built a helper function `fill_final_target_from_model(...)` to make predicting on new data frictionless. It automatically:
1. Loads the unseen test dataset.
2. Applies the exact same feature engineering used during training.
3. Aligns the columns so they match the trained model perfectly.
4. Makes predictions and generates a ready-to-submit CSV with the `readmitted` predictions (and optional probability scores).

---

## How to Run the Project

1. **Install the required libraries:**
   ```bash
   pip install -r requirements.txt

2. **Run the main script:**
   ```bash
   python HackathonTabularClassification.py

Note on Reproducibility: All random states for train/test splits and cross-validation are locked to random_state=42 to ensure consistent results on every run.
   ׳
