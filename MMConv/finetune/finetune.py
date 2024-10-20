import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from transformers import Trainer, TrainingArguments
import dataloader_dst
# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
model = AutoModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1")  # Change num_labels as needed
args = {
    "model_type": "bert",         # Specify the model type (e.g., "bert", "gpt")
    "usr_token": "<USR>",         # Token to represent user turns
    "sys_token": "<SYS>",         # Token to represent system responses
    "example_type": "turn",       # Specify example type (e.g., "turn" or "dial")
}
# Load your JSON data
with open('train_final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

metadata = {"num_labels":0}
# Create a Dataset and DataLoader
dataset = dataloader_dst.Dataset_dst(data, tokenizer, args, metadata)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Adjust batch size as needed

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',              # Output directory
    num_train_epochs=3,                  # Number of training epochs
    per_device_train_batch_size=8,       # Batch size for training
    per_device_eval_batch_size=8,        # Batch size for evaluation
    warmup_steps=500,                     # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                   # Strength of weight decay
    logging_dir='./logs',                 # Directory for storing logs
)

# Use Trainer for training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=your_eval_dataset,  # You can also include evaluation dataset if available
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model('fine_tuned_model')
