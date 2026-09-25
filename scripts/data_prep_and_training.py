"""
End-to-end pipeline: feature engineering, model training, and saving.

Usage:
    python scripts/data_prep_and_training.py

Reads:  preprocessing_data/processed_texts_final.csv
Writes: experiments/feature_engineering.csv
        models/*.joblib
        experiments/all_model_results.csv
"""

import os
import pandas as pd
import numpy as np
import textstat
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "preprocessing_data", "processed_texts_final.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "experiments", "feature_engineering.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_PATH = os.path.join(BASE_DIR, "experiments", "all_model_results.csv")

FEATURE_COLUMNS = [
    "word_count", "average_word_length", "percent_complex_words",
    "percent_difficult_words", "sentence_count", "syllables_per_word",
    "polysyllable_count", "flesch_reading_ease", "automated_readability_index",
    "dale_chall_readability_score", "difficult_words", "linsear_write_formula",
    "spache_readability", "reading_time",
]
TARGET_COLUMN = "fk_estimated_uk_age_num"

# ── Feature Engineering ──────────────────────────────────────────────────────

def engineer_features(df):
    print("Engineering features...")
    df = df.copy()
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df["average_word_length"] = df["text"].apply(
        lambda x: sum(len(w) for w in str(x).split()) / max(len(str(x).split()), 1)
    )
    df["complex_word_count"] = df["text"].apply(
        lambda x: sum(1 for w in str(x).split() if textstat.syllable_count(w) >= 3)
    )
    df["percent_complex_words"] = df["complex_word_count"] / df["word_count"].replace(0, 1)
    df["difficult_word_count"] = df["text"].apply(lambda x: textstat.difficult_words(str(x)))
    df["percent_difficult_words"] = df["difficult_word_count"] / df["word_count"].replace(0, 1)
    return df


# ── Training Helpers ─────────────────────────────────────────────────────────

def save_result(name, mae, mse, rmse, r2, params="Default", notes=""):
    row = pd.DataFrame([{
        "Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse,
        "R2": r2, "Params": str(params), "Notes": notes,
    }])
    if os.path.isfile(RESULTS_PATH):
        row.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(RESULTS_PATH, index=False)


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"  {name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.2f}")
    return mae, mse, rmse, r2


# ── Model Training ───────────────────────────────────────────────────────────

def train_all(X_train, X_test, y_train, y_test):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Clear previous results
    if os.path.isfile(RESULTS_PATH):
        os.remove(RESULTS_PATH)

    # Linear Regression
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(MODELS_DIR, "linear_regression.joblib"))
    save_result("Linear Regression", *evaluate("Linear Regression", lr, X_test, y_test))

    # Ridge Regression
    print("Training Ridge Regression...")
    alphas = np.arange(0.01, 10, 0.1)
    ridge = GridSearchCV(
        Ridge(max_iter=20000, random_state=4),
        {"alpha": alphas}, cv=cv, scoring="r2", n_jobs=-1
    )
    ridge.fit(X_train, y_train)
    best_ridge = ridge.best_estimator_
    joblib.dump(best_ridge, os.path.join(MODELS_DIR, "ridge_regression.joblib"))
    save_result("Ridge Regression", *evaluate("Ridge Regression", best_ridge, X_test, y_test),
                params=ridge.best_params_)

    # Lasso Regression
    print("Training Lasso Regression...")
    lasso = GridSearchCV(
        Lasso(max_iter=20000, random_state=4),
        {"alpha": alphas}, cv=cv, scoring="r2", n_jobs=-1
    )
    lasso.fit(X_train, y_train)
    best_lasso = lasso.best_estimator_
    joblib.dump(best_lasso, os.path.join(MODELS_DIR, "lasso_regression.joblib"))
    save_result("Lasso Regression", *evaluate("Lasso Regression", best_lasso, X_test, y_test),
                params=lasso.best_params_)

    # Decision Tree
    print("Training Decision Tree...")
    dt = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        {"max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 5, 10]},
        cv=5, scoring="r2", n_jobs=-1
    )
    dt.fit(X_train, y_train)
    best_dt = dt.best_estimator_
    joblib.dump(best_dt, os.path.join(MODELS_DIR, "decision_tree_regressor.joblib"))
    save_result("Decision Tree", *evaluate("Decision Tree", best_dt, X_test, y_test),
                params=dt.best_params_)

    # Random Forest
    print("Training Random Forest...")
    rf = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
        cv=5, scoring="r2", n_jobs=-1
    )
    rf.fit(X_train, y_train)
    best_rf = rf.best_estimator_
    joblib.dump(best_rf, os.path.join(MODELS_DIR, "random_forest_regressor.joblib"))
    save_result("Random Forest", *evaluate("Random Forest", best_rf, X_test, y_test),
                params=rf.best_params_)

    # XGBoost
    print("Training XGBoost...")
    xgb = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        cv=5, scoring="r2", n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    best_xgb = xgb.best_estimator_
    joblib.dump(best_xgb, os.path.join(MODELS_DIR, "xgb_model.joblib"))
    save_result("XGBoost", *evaluate("XGBoost", best_xgb, X_test, y_test),
                params=xgb.best_params_)

    # Stacking Regressor
    print("Training Stacking Regressor...")
    estimators = [
        ("ridge", Ridge(alpha=0.11)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("xgb", XGBRegressor(n_estimators=100, random_state=42, verbosity=0)),
    ]
    stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    stacking.fit(X_train, y_train)
    joblib.dump(stacking, os.path.join(MODELS_DIR, "stacking_regressor.joblib"))
    save_result("Stacking Regressor", *evaluate("Stacking Regressor", stacking, X_test, y_test))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows loaded")

    df = engineer_features(df)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"  Features saved to {FEATURES_PATH}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=456
    )
    print(f"  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows\n")

    train_all(X_train, X_test, y_train, y_test)

    print(f"\nAll models saved to {MODELS_DIR}/")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
