# LUKE Entity Pair Classification with PyTorch and Hugging Face Transformers
# I need to integrate first 

import torch
from torch.utils.data import Dataset
from transformers import LukeTokenizer, LukeForEntityPairClassification, Trainer, TrainingArguments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import ast

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("training_data_with_spans.csv")
df = df.dropna(subset=["passage", "label", "sp1_span", "sp2_span"])
df["label"] = df["label"].astype(int)

# Convert string spans to tuples
df["sp1_span"] = df["sp1_span"].apply(ast.literal_eval)
df["sp2_span"] = df["sp2_span"].apply(ast.literal_eval)

# Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Load tokenizer and model
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-base", num_labels=2)
model.to(device)

# Dataset class
class LukeEntityPairDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row["passage"]
        entity_spans = [tuple(row["sp1_span"]), tuple(row["sp2_span"])]
        encodings = self.tokenizer(
            text,
            entity_spans=entity_spans,
            add_prefix_space=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        encodings = {k: v.squeeze(0) for k, v in encodings.items()}
        encodings["labels"] = torch.tensor(row["label"])
        return encodings

# Create datasets
train_dataset = LukeEntityPairDataset(train_df, tokenizer)
test_dataset = LukeEntityPairDataset(test_df, tokenizer)

# Training args
training_args = TrainingArguments(
    output_dir="./luke_entity_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./luke_entity_logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save
model.save_pretrained("luke_entity_classifier")
tokenizer.save_pretrained("luke_entity_classifier")

print("\n✅ LUKE entity pair classifier trained and saved!")
