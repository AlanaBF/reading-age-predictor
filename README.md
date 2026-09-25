# Machine Learning Module 2 Project

This project, developed as part of the Module 2 ML & AI portfolio assessment, aims to build and evaluate machine learning models to estimate the UK reading age required for understanding public sector letters. This helps ensure communications are accessible, regulatory compliant, and appropriate for children and young people in public sector contexts.

## Project Structure

```
├── .venv/
├── data/
│   ├── raw_data/                # Untouched CEFR-level source texts (A1, A2, ...)
│   ├── processed_data/          # Model-ready CSVs: train.csv, test.csv, features.csv
│   └── grouped_by_reading_band/ # Texts grouped by derived reading bands (6-7, 8-12, ...)
├── preprocessing_data/          # Notebooks for cleaning, scoring, and organizing data
│   ├── 01_preprocess_raw_texts.ipynb
│   ├── 02_clean_and_organise_dataset.ipynb
│   ├── 03_explore_and_validate_readability.ipynb
│   ├── processed_texts_final.csv
│   ├── raw_texts_all_levels_with_smog_and_flesch_kincaid_scores.csv
│   └── analysis_outputs/
│       ├── duplicates_review.csv
│       └── outliers/
│       └── plots/               # Any useful diagrams
├── experiments/                 # Notebooks for feature engineering, modeling, evaluation
│   └── linear_regression/
│       └── lasso_model_training.ipynb
│       └── linear_model_training.ipynb
│       └── ridge_model_training.ipynb
│   └── regression_decision_tree/
│       └── decision_tree_model_training.ipynb
│   └── regression_random_forest/
│       └── random_forest_model_training.ipynb
│   └── scratchpad/
│       └── model_exploration.ipynb
│   └── evaluation_and_inference.ipynb
│   └── all_model_results.csv
│   └── feature_engineering.csv
│   └── feature_engineering.ipynb
├── scripts/                     # Final pipeline and prediction scripts
│   ├── data_prep_and_training.py
│   ├── evaluate_model.py
│   └── predict.py
├── models/                      # Saved trained model files
│   └── decision_tree_regressor.joblib
│   └── random_forest_regressor.joblib
│   └── linear_regression.joblib
│   └── lasso_regression.joblib
│   └── ridge_regression.joblib
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
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

These will be the production ready scripts using py files not jupyter notebooks. They will use the best performing model

- **scripts/data_prep_and_training.py**: End-to-end pipeline for data prep, feature extraction, and model training
- **scripts/evaluate_model.py**: Evaluate model performance on the test set
- **scripts/predict.py**: Make predictions on new data using the trained model

## Models

- Saved trained model files are in `models/` (e.g., `model.joblib`)

## Requirements

All code dependencies are managed via a shared requirements.txt for portability and reproducibility. This approach ensures consistency across development environments.

## Notes

- Preprocessing and analysis are separated from experiments/modeling for clarity.
- See each notebook for step-by-step documentation and outputs.

## Acknowledgements & References

Project inspired by NHS Readability and CEFR public datasets.

- [NHS Readability Tool](https://readability.ncldata.dev/)
- [CEFR Levelled English Texts (Adam Montgomerie, GitHub)](https://github.com/AMontgomerie/CEFR-English-Level-Predictor)
- [CEFR Levelled English Texts (Kaggle)](https://www.kaggle.com/datasets/amontgomerie/cefr-levelled-english-texts)
