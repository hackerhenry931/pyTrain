import os
import openai
from datasets import Dataset
import torch

# Set your OpenAI API key
openai.api_key = 'your-openai-api-key'

# Step 1: Load the Training Documents (plain text from all txt files in a folder)
def load_training_docs_from_folder(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                all_text += file.read() + "\n"  # Add a newline between files
    return all_text

# Step 2: Prepare Dataset for Fine-tuning (using plain text)
def prepare_dataset(text, tokenizer):
    # Tokenize the text
    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",  # Explicitly use max_length padding
        max_length=512,
        return_tensors="pt",  # Return PyTorch tensors
    )

    # Return dataset in format that Trainer expects (labels are the same as input_ids)
    return Dataset.from_dict({
        "input_ids": tokenized["input_ids"].squeeze().tolist(),
        "attention_mask": tokenized["attention_mask"].squeeze().tolist(),
        "labels": tokenized["input_ids"].squeeze().tolist(),  # In language modeling, labels are the same as input_ids
    })

# Step 3: Fine-tune GPT-3 Model (via OpenAI API)
def finetune_gpt3(training_text):
    # Use OpenAI API to fine-tune GPT-3 with your training data
    try:
        response = openai.FineTune.create(
            training_file=training_text,  # This should be your dataset in the required format
            model="davinci",  # You can use a specific GPT-3 variant like "davinci"
            n_epochs=4  # Set the number of epochs (you can adjust as needed)
        )
        return response
    except Exception as e:
        print(f"Error while fine-tuning: {e}")
        return None

# Step 4: Use Fine-tuned Model to Answer Questions via OpenAI API
def ask_question_gpt3(finetuned_model_id, question):
    # Call the GPT-3 API with the fine-tuned model
    try:
        response = openai.Completion.create(
            model=finetuned_model_id,
            prompt=question,
            max_tokens=100,  # Maximum number of tokens in the response
            temperature=0.7,  # Control the randomness of the response
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


# Main script
if __name__ == "__main__":
    # Replace these paths and variables with your own
    training_folder_path = "/Users/hnguyen/Downloads/house_music/"  # Path to your folder with .txt files
    question_to_ask = "Where did house music come from?"

    # Step 1: Load training documents from folder
    training_text = load_training_docs_from_folder(training_folder_path)

    # Step 2: Fine-tune the GPT-3 model via OpenAI API
    print("Fine-tuning the GPT-3 model...")
    fine_tuned_model_response = finetune_gpt3(training_text)
    if fine_tuned_model_response:
        finetuned_model_id = fine_tuned_model_response["fine_tuned_model"]
        print(f"Model fine-tuned successfully with model ID: {finetuned_model_id}")

        # Step 3: Ask a question to the fine-tuned model
        print("Asking a question to the fine-tuned GPT-3 model...")
        answer = ask_question_gpt3(finetuned_model_id, question_to_ask)
        print(f"Answer: {answer}")
    else:
        print("Fine-tuning failed.")
