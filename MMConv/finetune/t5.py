import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load pre-trained T5 model and tokenizer
model_name = 't5-base'  # You can use 't5-small', 't5-large', etc.
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the input text
input_text = "Find me a cheap restaurant nearby the north town."

# Prepend a task-specific prefix if needed (T5 uses text-to-text format)
# For example, you can specify the task as "translate English to French: " for translation tasks
# Here, you can define a prefix like "recommend restaurant: "
task_prefix = "recommend restaurant: "
formatted_input = task_prefix + input_text

# Tokenize the input
input_ids = tokenizer.encode(formatted_input, return_tensors='pt').to(device)

# Generate the answer
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)

# Decode the generated tokens to get the answer
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Answer: {answer}")
