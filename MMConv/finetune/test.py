import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
tod_bert = AutoModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1")

# Encode the input text
input_text = "[CLS] [SYS] Hello, what can I help with you today? [USR] Find me a cheap restaurant nearby the north town."
input_tokens = tokenizer.tokenize(input_text)

# Convert tokens to input IDs
input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

# Convert input IDs to tensor
story = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

# Move model and inputs to GPU if available
if torch.cuda.is_available():
    tod_bert = tod_bert.cuda()
    story = story.cuda()

# Run the model
with torch.no_grad():
    input_context = {"input_ids": story, "attention_mask": (story > 0).long()}
    hiddens = tod_bert(**input_context)[0]  # Get the hidden states

# Print the hidden states for further processing
print("Hidden States:", hiddens)

# If you need to use these hidden states for further downstream tasks,
# you can pass them to a classifier or another model component.
