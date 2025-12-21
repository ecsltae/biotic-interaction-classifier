# SVM.py
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Load Preprocessed Data ---
try:
    with open("preprocessed_data2.pkl", "rb") as f:
        (X_train, y_train, X_val, y_val, X_test, y_test, test_sentences) = pickle.load(f)
    print("Preprocessed data loaded successfully.")
except FileNotFoundError:
    print("Error: 'preprocessed_data2.pkl' not found.")
    print("Please run the data preprocessing script first.")
    exit()

# --- Train Support Vector Machine (SVM) ---
print("\nTraining Support Vector Machine model...")
svm_clf = SVC(class_weight="balanced", random_state=42, probability=True)
svm_clf.fit(X_train, y_train)
print("SVM model training complete.")

# --- Train Random Forest ---
print("\nTraining Random Forest model...")
rf_clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_clf.fit(X_train, y_train)
print("Random Forest model training complete.")

# --- Evaluate Models ---
def evaluate_model(model, X, y, model_name):
    """
    Evaluates the model on a given dataset and prints the results.
    """
    print(f"\n--- Evaluation for {model_name} on the Test Set ---")
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

# Evaluate both models on the test set
evaluate_model(svm_clf, X_test, y_test, "Support Vector Machine")
evaluate_model(rf_clf, X_test, y_test, "Random Forest")

# --- Save Models ---
with open("svm_model.pkl", "wb") as f:
    pickle.dump(svm_clf, f)
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_clf, f)

print("\nModels trained and saved successfully as 'svm_model.pkl' and 'random_forest_model.pkl'!")
