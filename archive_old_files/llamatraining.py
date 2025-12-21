import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

# Load preprocessed dataset
df = pd.read_csv("training_data.csv")

# Ensure text and label columns have no NaN values
df = df.dropna(subset=["passage", "label"])

# Convert labels to text (for generative model)
df["label"] = df["label"].apply(lambda x: "Positive" if x == 1 else "Negative")

# Create prompts for generative model
df["prompt"] = "Passage: " + df["passage"] + " Sentiment:" #Simple prompt

# Split into training and test sets
train_texts, train_labels, _, _ = train_test_split(
    df["prompt"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].tolist()
)

# Initialize Accelerator for CPU Offloading
accelerator = Accelerator(cpu=True)

# Load tokenizer and model (Meta-Llama-3-8B)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B") #base model
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map="auto",  # Automatically distribute layers across available devices
    quantization_config=dict(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # or torch.float16
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
)



# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.label_encodings = tokenizer(labels, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()} #clone and detach
        label_item = {key: val[idx].clone().detach() for key, val in self.label_encodings.items()} #clone and detach

        input_ids = item['input_ids'].squeeze()
        label_ids = label_item['input_ids'].squeeze()

        label_ids = label_ids[:self.max_length]
        if len(label_ids) < self.max_length:
            label_ids = torch.cat([label_ids, torch.tensor([tokenizer.pad_token_id] * (self.max_length - len(label_ids)))])

        labels = torch.cat((label_ids[1:], torch.tensor([-100])))

        item['labels'] = labels
        return item

# Create dataset objects
train_dataset = TextDataset(train_texts, train_labels, tokenizer)

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # Rank of the update matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Dropout probability
    bias="none",
    task_type="CAUSAL_LM",  # Important for causal language models
    target_modules=["q_proj", "v_proj"], # Target the query and value projection layers
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model, train_dataset = accelerator.prepare(model, TextDataset(train_texts, train_labels, tokenizer, max_length=128))


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="epoch",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=False,
    learning_rate=2e-5,
    gradient_accumulation_steps=16,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model
accelerator.unwrap_model(model).save_pretrained("llama3_classifier")
tokenizer.save_pretrained("llama3_classifier")

print("\n✅ Llama 3 model trained and saved!")

# Free up GPU Memory (Example)
def free_gpu_memory():
    """Frees up GPU memory by deleting variables and clearing cache."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Call the function when needed (e.g., after training)
free_gpu_memory()