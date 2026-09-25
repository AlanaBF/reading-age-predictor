import pandas as pd
import joblib
import textstat

def extract_features(text):
    features = {}
    features['word_count'] = len(text.split())
    features['average_word_length'] = sum(len(word) for word in text.split()) / len(text.split())
    features['complex_word_count'] = sum(1 for word in text.split() if textstat.syllable_count(word) >= 3)
    features['percent_complex_words'] = features['complex_word_count'] / features['word_count']
    features['difficult_word_count'] = textstat.difficult_words(text)
    features['percent_difficult_words'] = features['difficult_word_count'] / features['word_count']
    features['sentence_count'] = textstat.sentence_count(text)
    features['syllables_per_word'] = textstat.syllable_count(text) / len(text.split())
    features['polysyllable_count'] = features['complex_word_count']
    features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    features['automated_readability_index'] = textstat.automated_readability_index(text)
    features['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)
    features['difficult_words'] = features['difficult_word_count']
    features['linsear_write_formula'] = textstat.linsear_write_formula(text)
    features['spache_readability'] = textstat.spache_readability(text)
    features['reading_time'] = textstat.reading_time(text)
    return features

feature_columns = [
    'word_count', 'average_word_length', 'percent_complex_words', 'percent_difficult_words', 'sentence_count', 'syllables_per_word',
    'polysyllable_count', 'flesch_reading_ease', 'automated_readability_index',
    'dale_chall_readability_score', 'difficult_words', 'linsear_write_formula',
    'spache_readability', 'reading_time'
]

# Load model
model = joblib.load('models/ridge_regression.joblib')

# Load your CSV
df = pd.read_csv('gold_standard.csv')

# Note: Headers include:
# letter_id, original_text, fk_score_orig, fk_age_orig, target_age, meets_target (Y/N), gpt_35_text, fk_age_35, meets_target_35, gpt_4_text, fk_age_4, meets_target_4, model_pred_age, model_match_35 (Y/N), model_match_4 (Y/N), cosine_sim_35, cosine_sim_4


# For each text type (original, gpt_35, gpt_4), predict reading age
for col in ['original_text', 'gpt_35_text', 'gpt_4_text']:
    print(f"Processing predictions for: {col}")
    features = df[col].apply(extract_features).apply(pd.Series)
    # Ensure the columns order matches model expectation
    features = features[feature_columns]
    df[f'model_pred_age_{col}'] = model.predict(features)

# Optional: check if model prediction meets target for each version
for col in ['original_text', 'gpt_35_text', 'gpt_4_text']:
    df[f'model_meets_target_{col}'] = df.apply(lambda row: 'Y' if row[f'model_pred_age_{col}'] <= row['target_age'] else 'N', axis=1)

# Save updated dataframe to CSV
df.to_csv('gold_standard_with_predictions.csv', index=False)
print("Predictions saved to gold_standard_with_predictions.csv")
