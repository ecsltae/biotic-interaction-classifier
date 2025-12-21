#logistic regression and random forest training script, verstion preprocessed_data.pkl
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed data
(X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(open("preprocessed_data.pkl", "rb"))

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train, y_train)

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_clf.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    print(f"\n Evaluation on {dataset_name}:")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(classification_report(y, y_pred))

evaluate_model(log_reg, X_test, y_test, "Test Set (Logistic Regression)")
evaluate_model(rf_clf, X_test, y_test, "Test Set (Random Forest)")

# Save models
pickle.dump(log_reg, open("logistic_regression_model.pkl", "wb"))
pickle.dump(rf_clf, open("random_forest_model.pkl", "wb"))

print("\n Models trained and saved!")



"""
OPTION 2 - BERT Model Training

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import pickle

# Load preprocessed dataset
df = pd.read_csv("training_data.csv")  # Reload raw text dataset for deep learning
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenization function
def tokenize_data(texts, labels):
    encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)
    return encodings, torch.tensor(labels.tolist())

# Convert dataset to tokenized tensors
train_texts, train_labels = df["passage"][df["label"] == 1], df["label"][df["label"] == 1]
test_texts, test_labels = df["passage"][df["label"] == 0], df["label"][df["label"] == 0]

train_encodings, train_labels = tokenize_data(train_texts, train_labels)
test_encodings, test_labels = tokenize_data(test_texts, test_labels)

# Custom PyTorch Dataset
class PassageDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = PassageDataset(train_encodings, train_labels)
test_dataset = PassageDataset(test_encodings, test_labels)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("bert_classifier")
tokenizer.save_pretrained("bert_classifier")

print("\n BERT model trained and saved!")

"""