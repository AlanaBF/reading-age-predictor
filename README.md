# Reading Age Predictor

A machine learning pipeline that predicts the UK reading age of a text from its linguistic and structural features. Seven regression models are trained, compared, and validated against a gold standard dataset of public domain book excerpts spanning reading ages 6–17.

## Project Structure

```
├── preprocessing_data/          # Notebooks for cleaning, scoring, and organising data
│   ├── 01_preprocess_raw_texts.ipynb
│   ├── 02_clean_and_organise_dataset.ipynb
│   ├── 03_explore_and_validate_readability.ipynb
│   ├── processed_texts_final.csv
│   ├── raw_texts_all_levels_with_smog_and_flesch_kincaid_scores.csv
│   └── analysis_outputs/
│       ├── duplicates_review.csv
│       ├── outliers/
│       └── plots/
├── experiments/                 # Feature engineering, model training, evaluation
│   ├── linear_regression/
│   ├── regression_decision_tree/
│   ├── regression_random_forest/
│   ├── ensemble_methods/
│   ├── scratchpad/
│   ├── feature_engineering.ipynb
│   ├── feature_engineering.csv
│   ├── evaluation_and_inference.ipynb
│   └── all_model_results.csv
├── gold_standard/               # Validation against public domain book excerpts
│   ├── book_excerpts.csv        # Source texts (Beatrix Potter → Jane Austen)
│   ├── GoldStandardPrep.ipynb   # Builds gold_standard.csv from book_excerpts.csv
│   ├── evaluate_models_on_gold_standard_data.ipynb
│   ├── simplify_text.ipynb      # GPT simplification pipeline (requires OPENAI_API_KEY)
│   └── model_metrics.csv
├── scripts/                     # Production scripts
│   ├── data_prep_and_training.py
│   ├── evaluate_model.py
│   └── predict.py
├── models/                      # Saved trained model files (.joblib)
├── supporting_documents/
├── requirements.txt
└── README.md
```

## Quickstart

1. Create a folder for the project
2. Clone the repo into this folder
3. Set up your virtual environment and install requirements:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Run the preprocessing notebooks in this order:
   - `01_preprocess_raw_texts.ipynb`
          > NOTE: Not needed unless you have raw text in txt files - this data is already saved in raw_texts_all_levels_with_smog_and_flesch_kincaid_scores.csv so you can move straight notebook 2.
   - `02_clean_and_organise_dataset.ipynb`
   - `03_explore_and_validate_readability.ipynb`
   Then proceed to:
   - `experiments/feature_engineering.ipynb`
   - Each model’s `model_training.ipynb`
   When ready to evaluate and  infer move to
   - `evaluation_and_inference.ipynb`
   Note the scratchpad is for code off cuts that I found useful as I developed and improved my model training but I want to keep my model training notebooks clean and so only the final code is in these.

5. Use scripts in `/scripts` for automated training, evaluation, or prediction.

## Data

All datasets used are open source or fully anonymized to comply with ethical and GDPR standards.

- Data in the following folders (`data/`) are gitignored because the data is included in the relevant csvs.

- **Raw Data:**
  - Located in `data/raw_data/` (A1, A2, B1, B2, C1, C2 subfolders)
  - Untouched source texts for each CEFR level
- **Grouped Data by Reading Band:**
  - Located in `data/grouped_by_reading_band/`
  - Texts grouped by derived reading bands (e.g., 6-7, 8-12, 13-14, etc.)

## Feature Engineering

The model is trained using the following features extracted from each text sample.

These features were selected after exploratory analysis and align with established readability metrics.

- word_count: Total number of words

- average_word_length: Mean length of words

- percent_complex_words: Proportion of words with 3+ syllables

- percent_difficult_words: Proportion of words considered ‘difficult’ (per Textstat)

- sentence_count: Number of sentences

- syllables_per_word: Average syllables per word

- polysyllable_count: Number of words with 3+ syllables

- flesch_reading_ease: Flesch Reading Ease score

- automated_readability_index: ARI readability metric

- dale_chall_readability_score: Dale-Chall readability metric

- difficult_words: Count of ‘difficult’ words

- linsear_write_formula: Linsear Write readability metric

- spache_readability: Spache readability metric

- reading_time: Estimated reading time in seconds

The target variable (y) is the estimated UK reading age (numeric), derived using the Flesch-Kincaid formula.

## Scripts

Production scripts using the best-performing model (Ridge Regression).

- **scripts/data_prep_and_training.py** — end-to-end pipeline: feature engineering, trains all 7 models, saves to `models/`
- **scripts/evaluate_model.py** — evaluates all saved models on the held-out test set, prints MAE / RMSE / R²
- **scripts/predict.py** — predicts reading age for a single text string, a CSV file, or interactively

```bash
# Single text
python scripts/predict.py "It was a dark and stormy night."

# CSV (must have a 'text' column)
python scripts/predict.py --csv my_texts.csv

# Interactive mode
python scripts/predict.py
```

## Models

- Saved trained model files are in `models/` (e.g., `model.joblib`)

## Requirements

All code dependencies are managed via a shared requirements.txt for portability and reproducibility. This approach ensures consistency across development environments.

## Gold Standard Validation

The `gold_standard/` folder validates model performance against eight public domain book excerpts
with known reading levels (Beatrix Potter age 6 through to Jane Austen age 17).

1. Run `GoldStandardPrep.ipynb` to generate `gold_standard.csv`
2. Run `evaluate_models_on_gold_standard_data.ipynb` to score all models
3. Optionally run `simplify_text.ipynb` (requires `OPENAI_API_KEY`) to generate GPT-simplified versions

## Acknowledgements & References

- [NHS Readability Tool](https://readability.ncldata.dev/)
- [CEFR Levelled English Texts (Adam Montgomerie, GitHub)](https://github.com/AMontgomerie/CEFR-English-Level-Predictor)
- [CEFR Levelled English Texts (Kaggle)](https://www.kaggle.com/datasets/amontgomerie/cefr-levelled-english-texts)
- Public domain texts sourced via [Project Gutenberg](https://www.gutenberg.org/)
