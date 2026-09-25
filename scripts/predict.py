"""
Predict the reading age of one or more texts using the trained Ridge Regression model.

Usage:
    # Single text (passed as argument)
    python scripts/predict.py "It was a dark and stormy night."

    # CSV file (must have a column named 'text')
    python scripts/predict.py --csv path/to/texts.csv

Output:
    Predicted UK reading age and a suitability note for each text.
"""

import sys
import os
import argparse
import pandas as pd
import textstat
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ridge_regression.joblib")

FEATURE_COLUMNS = [
    "word_count", "average_word_length", "percent_complex_words",
    "percent_difficult_words", "sentence_count", "syllables_per_word",
    "polysyllable_count", "flesch_reading_ease", "automated_readability_index",
    "dale_chall_readability_score", "difficult_words", "linsear_write_formula",
    "spache_readability", "reading_time",
]


def extract_features(text):
    words = text.split()
    word_count = max(len(words), 1)
    complex_count = sum(1 for w in words if textstat.syllable_count(w) >= 3)
    difficult_count = textstat.difficult_words(text)
    return {
        "word_count": word_count,
        "average_word_length": sum(len(w) for w in words) / word_count,
        "complex_word_count": complex_count,
        "percent_complex_words": complex_count / word_count,
        "difficult_word_count": difficult_count,
        "percent_difficult_words": difficult_count / word_count,
        "sentence_count": textstat.sentence_count(text),
        "syllables_per_word": textstat.syllable_count(text) / word_count,
        "polysyllable_count": complex_count,
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "difficult_words": difficult_count,
        "linsear_write_formula": textstat.linsear_write_formula(text),
        "spache_readability": textstat.spache_readability(text),
        "reading_time": textstat.reading_time(text),
    }


def predict(texts):
    model = joblib.load(MODEL_PATH)
    features = pd.DataFrame([extract_features(t) for t in texts])[FEATURE_COLUMNS]
    predictions = model.predict(features)
    return predictions


def format_result(text, predicted_age):
    age = round(predicted_age, 1)
    snippet = text[:60].replace("\n", " ") + ("..." if len(text) > 60 else "")
    return f'  Text: "{snippet}"\n  Predicted reading age: {age}\n'


def main():
    parser = argparse.ArgumentParser(description="Predict reading age of text")
    parser.add_argument("text", nargs="?", help="Text string to evaluate")
    parser.add_argument("--csv", help="Path to CSV file with a 'text' column")
    args = parser.parse_args()

    if not os.path.isfile(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Run: python scripts/data_prep_and_training.py")
        sys.exit(1)

    if args.csv:
        df = pd.read_csv(args.csv)
        if "text" not in df.columns:
            print("CSV must contain a column named 'text'")
            sys.exit(1)
        texts = df["text"].tolist()
        predictions = predict(texts)
        df["predicted_reading_age"] = [round(p, 1) for p in predictions]
        out_path = args.csv.replace(".csv", "_predictions.csv")
        df.to_csv(out_path, index=False)
        print(f"Predictions saved to {out_path}")
        print(df[["text", "predicted_reading_age"]].to_string(index=False))

    elif args.text:
        pred = predict([args.text])[0]
        print(format_result(args.text, pred))

    else:
        # Interactive mode
        print("Reading Age Predictor  (type 'quit' to exit)\n")
        while True:
            text = input("Enter text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if text:
                pred = predict([text])[0]
                print(format_result(text, pred))


if __name__ == "__main__":
    main()
