import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load the fine-tuned Llama model and tokenizer with quantization
model = AutoModelForCausalLM.from_pretrained(
    "llama3_biotic_interaction",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("llama3_biotic_interaction")

# --- Load Preprocessed Data and Sentences ---
try:
    with open("preprocessed_data2.pkl", "rb") as f:
        (X_train, y_train, X_val, y_val, X_test, y_test, test_sentences) = pickle.load(f)
    print("Preprocessed data and sentences loaded successfully.")
except FileNotFoundError:
    print("Error: 'preprocessed_data2.pkl' not found.")
    exit()
except NameError:
    import pickle
    with open("preprocessed_data2.pkl", "rb") as f:
        (X_train, y_train, X_val, y_val, X_test, y_test, test_sentences) = pickle.load(f)

# --- Sample the Test Data ---
sample_size = 100
sample_sentences = test_sentences[:sample_size]
y_sample_true = y_test[:sample_size]

# --- Make Predictions on Sample Data ---
predictions = []
for sentence in tqdm(sample_sentences, desc="Predicting"):
    prompt = f"Passage: {sentence}\nDoes this passage describe a biotic interaction between species? Answer: "
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=15)
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_text = predicted_text.replace(prompt, "").strip()
    predictions.append("Interaction" if "Interaction" in predicted_text else "No Interaction")

# --- Create DataFrame and Save to CSV ---
sample_df = pd.DataFrame({
    'sentence': sample_sentences,
    'true_label': y_sample_true,
    'predicted_interaction': predictions,
})

# Map numerical labels to 'Interaction' and 'No Interaction' for readability
sample_df['true_interaction'] = sample_df['true_label'].map({1: 'Interaction', 0: 'No Interaction'})

# Save the DataFrame to a CSV file
csv_filename = 'sample_sentences_with_biotic_interaction_predictions.csv'
sample_df.to_csv(csv_filename, index=False)
print(f"\nCSV file with sentences and biotic interaction predictions saved as '{csv_filename}'.")
