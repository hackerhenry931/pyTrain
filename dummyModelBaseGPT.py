from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can change this to another model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to ask a question and get an answer
def ask_question(question):
    # Encode the question
    inputs = tokenizer.encode(question, return_tensors="pt")

    # Generate a response
    outputs = model.generate(
        inputs,
        max_length=100,  # Adjust the length as needed
        num_return_sequences=1,  # We want just one answer
        no_repeat_ngram_size=2,  # Avoid repeating n-grams
        do_sample=True,  # Use sampling
        top_k=50,  # Top-k sampling
        top_p=0.95,  # Nucleus sampling (top-p)
        temperature=0.7,  # Adjust for randomness
    )

    # Decode the generated output
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Example: Asking about house music
question = "What is the capital of france that starts with a P?"
answer = ask_question(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
