import pickle
import re
import nltk
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

# Function to clean and normalize text
def preprocess_text(text):
    text = text.lower().strip()  # Lowercase & trim spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

# Load the trained SVM model and vectorizer
try:
    with open("svm_model.pkl", "rb") as f:
        svm_clf = pickle.load(f)
    print("SVM model loaded successfully.")
except FileNotFoundError:
    print("Error: 'svm_model.pkl' not found.")
    sys.exit(1)

try:
    with open("tfidf_vectorizer2.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: 'tfidf_vectorizer2.pkl' not found.")
    sys.exit(1)

print("Enter sentences to predict their sentiment. Press 'q' to exit.\n")

try:
    while True:
        # Prompt the user to input a sentence
        user_sentence = input("> ")

        # Exit if the user presses 'q' or 'Esc'
        if user_sentence.lower() == 'q':
            break

        # Preprocess the user input
        cleaned_sentence = preprocess_text(user_sentence)

        # Vectorize the preprocessed sentence
        sentence_vector = vectorizer.transform([cleaned_sentence])

        # Predict the sentiment
        prediction = svm_clf.predict(sentence_vector)
        sentiment = "positive" if prediction[0] == 1 else "negative"

        # Output the result
        print(f"Predicted sentiment: {sentiment}")

except KeyboardInterrupt:
    print("\nExiting the script. Goodbye!")
except EOFError:
    print("\nExiting the script. Goodbye!")
