"""
Evaluate all saved models on the held-out test set.

Usage:
    python scripts/evaluate_model.py

Reads:  experiments/feature_engineering.csv  (or reruns feature engineering
        from preprocessing_data/processed_texts_final.csv if not found)
        models/*.joblib
Prints: MAE, RMSE, R² for each model on the test split
"""

import os
import pandas as pd
import numpy as np
import textstat
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "experiments", "feature_engineering.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "preprocessing_data", "processed_texts_final.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLUMNS = [
    "word_count", "average_word_length", "percent_complex_words",
    "percent_difficult_words", "sentence_count", "syllables_per_word",
    "polysyllable_count", "flesch_reading_ease", "automated_readability_index",
    "dale_chall_readability_score", "difficult_words", "linsear_write_formula",
    "spache_readability", "reading_time",
]
TARGET_COLUMN = "fk_estimated_uk_age_num"

MODEL_NAMES = [
    "linear_regression",
    "ridge_regression",
    "lasso_regression",
    "decision_tree_regressor",
    "random_forest_regressor",
    "xgb_model",
    "stacking_regressor",
]


def load_features():
    if os.path.isfile(FEATURES_PATH):
        print(f"Loading features from {FEATURES_PATH}")
        return pd.read_csv(FEATURES_PATH)

    print(f"Feature file not found — computing from {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
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


def main():
    df = load_features()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=456)

    print(f"\nEvaluating on {len(X_test)} test samples\n")
    print(f"{'Model':<30} {'MAE':>6} {'RMSE':>6} {'R²':>6}")
    print("-" * 52)

    results = []
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if not os.path.isfile(path):
            print(f"  {name}: model file not found, skipping")
            continue
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        print(f"  {name:<28} {mae:>6.2f} {rmse:>6.2f} {r2:>6.2f}")
        results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    print()
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df["R2"].idxmax()]
    print(f"Best model by R²: {best['model']}  (R²={best['R2']:.2f}, MAE={best['MAE']:.2f})")


if __name__ == "__main__":
    main()
