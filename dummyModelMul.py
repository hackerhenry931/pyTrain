import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import torch

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
    # Check if the tokenizer has a valid padding token
    if tokenizer.pad_token is None:
        print("Assigning a pad_token to the tokenizer...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a new padding token

    # Tokenize the text (no need for separate inputs/targets)
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

# Step 3: Fine-tune the Model
def finetune_model(model_name, train_dataset, tokenizer, output_dir="finetuned_model"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=2,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir

# Step 4: Use Fine-tuned Model to Answer Questions
def ask_question(finetuned_model_path, tokenizer, question):
    model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)
    
    # Encode the question
    inputs = tokenizer.encode(question, return_tensors="pt")
    
    # Set pad_token_id to eos_token_id explicitly
    model.config.pad_token_id = model.config.eos_token_id
    
    # Generate the answer using the model
    outputs = model.generate(
        inputs,
        max_length=30,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,  # Top-k sampling
        top_p=0.95,  # Nucleus sampling
        attention_mask=torch.ones_like(inputs)
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Main script
if __name__ == "__main__":
    # Replace these paths and variables with your own
    training_folder_path = "/Users/hnguyen/Downloads/house_music/"  # Path to your folder with .txt files
    base_model_name = "gpt2"  # Base model name, e.g., "gpt2" or "distilgpt2"
    question_to_ask = "Where was house music born?"

    # Step 1: Load training documents from folder
    training_text = load_training_docs_from_folder(training_folder_path)

    # Step 2: Load tokenizer and prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_dataset = prepare_dataset(training_text, tokenizer)

    # Step 3: Fine-tune the model
    print("Fine-tuning the model...")
    finetuned_model_dir = finetune_model(base_model_name, train_dataset, tokenizer)
    print(f"Model fine-tuned and saved at {finetuned_model_dir}")

    # Step 4: Ask a question
    print("Asking a question to the fine-tuned model...")
    answer = ask_question(finetuned_model_dir, tokenizer, question_to_ask)
    print(f"Answer: {answer}")
