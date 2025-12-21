import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys


# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


# Load the fine-tuned Llama model and tokenizer
try:
    model = AutoModelForCausalLM.from_pretrained(
        "llama3_biotic_interaction",
        quantization_config=bnb_config,
        device_map="auto"  # Automatically offloads some layers to CPU if needed
    )
    tokenizer = AutoTokenizer.from_pretrained("llama3_biotic_interaction")
    print("Llama model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    sys.exit(1)

print("Enter sentences to check for biotic interactions. Press 'q' to exit.\n")

try:
    while True:
        # Prompt the user to input a sentence
        user_sentence = input("> ")

        # Exit if the user presses 'q'
        if user_sentence.lower() == 'q':
            break

        # Create prompt for the model
        prompt = f"Passage: {user_sentence}\nDoes this passage describe a biotic interaction between species? Answer: "

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to("cuda")

        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15)

        # Decode the output
        predicted_interaction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_interaction = predicted_interaction.replace(prompt, "").strip()

        # Output the result
        print(f"Predicted: {predicted_interaction}")

except KeyboardInterrupt:
    print("\nExiting the script. Goodbye!")
except EOFError:
    print("\nExiting the script. Goodbye!")
