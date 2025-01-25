import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
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

    # Here we are creating a dummy context and question for each piece of text
    # For simplicity, we are using the same text as context and asking about it
    questions = ["What is the text about?"] * len(text.split('\n'))  # Dummy question for each paragraph
    contexts = text.split('\n')  # Split text into paragraphs (one per context)

    # Tokenize the data
    encodings = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
        return_token_type_ids=True,
        return_attention_mask=True,
    )

    # Return dataset in format that Trainer expects (labels are the answer spans)
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"].tolist(),
        "attention_mask": encodings["attention_mask"].tolist(),
        "token_type_ids": encodings["token_type_ids"].tolist(),
        "start_positions": torch.tensor([0] * len(contexts)),  # Dummy start positions for now
        "end_positions": torch.tensor([0] * len(contexts)),  # Dummy end positions for now
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
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
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
def ask_question(finetuned_model_path, tokenizer, question, context):
    model = AutoModelForQuestionAnswering.from_pretrained(finetuned_model_path)
    
    # Encode the question and context
    inputs = tokenizer(question, context, return_tensors="pt")
    
    # Get the answer span from the model
    outputs = model(**inputs)
    
    # Get the start and end positions of the answer
    start_position = torch.argmax(outputs.start_logits)
    end_position = torch.argmax(outputs.end_logits)
    
    # Convert the token IDs back to the string (answer)
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs.input_ids[0][start_position:end_position + 1])
    )
    
    return answer

# Main script
if __name__ == "__main__":
    # Replace these paths and variables with your own
    training_folder_path = "/Users/hnguyen/Downloads/house_music/"  # Path to your folder with .txt files
    base_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # Pre-trained BERT for question answering
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
    context = "House music originated in Chicago in the early 1980s."  # Context to use for the question
    answer = ask_question(finetuned_model_dir, tokenizer, question_to_ask, context)
    print(f"Answer: {answer}")
