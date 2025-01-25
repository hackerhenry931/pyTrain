from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load the pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-2.7B"  # You can choose a different GPT-Neo model variant
model = GPTNeoForCausalLM.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to ask a question
def ask_question(question):
    # Encode the input question
    input_ids = tokenizer.encode(question, return_tensors="pt")
    
    # Generate a response from the model
    output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, top_k=60)
    
    # Decode the model's response
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Ask a question
question = "What is the capital of France?"
answer = ask_question(question)

# Output the answer
print("Answer:", answer)
