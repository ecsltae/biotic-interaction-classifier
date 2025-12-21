# predict_with_sentences.py
import pickle
import pandas as pd

# --- Load Preprocessed Data and Sentences ---
try:
    with open("preprocessed_data2.pkl", "rb") as f:
        (X_train, y_train, X_val, y_val, X_test, y_test, test_sentences) = pickle.load(f)
    print("Preprocessed data and sentences loaded successfully.")
except FileNotFoundError:
    print("Error: 'preprocessed_data2.pkl' not found.")
    exit()

# --- Load Saved SVM Model ---
try:
    with open("svm_model.pkl", "rb") as f:
        svm_clf = pickle.load(f)
    print("SVM model loaded successfully.")
except FileNotFoundError:
    print("Error: 'svm_model.pkl' not found.")
    exit()

# --- Sample the Test Data ---
sample_size = 100
X_sample = X_test[:sample_size]
y_sample_true = y_test[:sample_size]
sample_sentences = test_sentences[:sample_size]

# --- Make Predictions on Sample Data ---
y_pred_svm = svm_clf.predict(X_sample)

# --- Create DataFrame and Save to CSV ---
sample_df = pd.DataFrame({
    'sentence': sample_sentences,
    'true_label': y_sample_true,
    'svm_prediction': y_pred_svm,
})

# Map numerical labels to 'positive' and 'negative' for readability
sample_df['true_sentiment'] = sample_df['true_label'].map({1: 'positive', 0: 'negative'})
sample_df['svm_sentiment'] = sample_df['svm_prediction'].map({1: 'positive', 0: 'negative'})

# Save the DataFrame to a CSV file
csv_filename = 'sample_sentences_with_svm_predictions3.csv'
sample_df.to_csv(csv_filename, index=False)

print(f"\nCSV file with sentences and SVM predictions saved as '{csv_filename}'.")
