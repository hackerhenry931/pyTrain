from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from transformers import Trainer, TrainingArguments

# Load the tokenizer and model
base_model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(base_model_name)
model = T5ForConditionalGeneration.from_pretrained(base_model_name)

# Example paragraphs (input and target will be the same for simplicity)
paragraphs = [
    "This is the first paragraph of the dataset.",
    "Here is the second paragraph, which will also be used for training."
]

# Tokenize the input and prepare decoder inputs
def tokenize_paragraphs(paragraphs, tokenizer):
    tokenized_data = []
    for text in paragraphs:
        # Tokenize the input and output (target)
        input_encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512)
        
        # Decoder inputs: T5 uses the same text as input and target (here for simplicity)
        target_encoding = tokenizer(text, truncation=True, padding='max_length', max_length=512)
        
        # Combine input and output (target) for sequence-to-sequence
        tokenized_data.append({
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'decoder_input_ids': target_encoding['input_ids'],  # Same as input for simplicity
            'decoder_attention_mask': target_encoding['attention_mask'],
            'labels': target_encoding['input_ids']  # Add labels for loss calculation
        })
    return tokenized_data

# Tokenize the paragraphs
tokenized_data = tokenize_paragraphs(paragraphs, tokenizer)

# Convert tokenized data into Dataset format
train_dataset = Dataset.from_dict({
    'input_ids': [item['input_ids'] for item in tokenized_data],
    'attention_mask': [item['attention_mask'] for item in tokenized_data],
    'decoder_input_ids': [item['decoder_input_ids'] for item in tokenized_data],
    'decoder_attention_mask': [item['decoder_attention_mask'] for item in tokenized_data],
    'labels': [item['labels'] for item in tokenized_data]  # Add the labels here
})

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no",  # Disable evaluation for simplicity
)

# Trainer Initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Optionally save the fine-tuned model
trainer.save_model("./finetuned_model")
