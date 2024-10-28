import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, PeftModelForSequenceClassification

# Load the pre-trained TOD-BERT model and tokenizer
model_name = 'TODBERT/TOD-BERT-JNT-V1'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=8,           # Low-rank adaptation dimension (choose according to your needs)
    lora_alpha=32, # Scaling factor
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=[
        "bert.encoder.layer.0.attention.self.query",
        "bert.encoder.layer.0.attention.self.value",
        "bert.encoder.layer.1.attention.self.query",
        "bert.encoder.layer.1.attention.self.value",
        "bert.encoder.layer.2.attention.self.query",
        "bert.encoder.layer.2.attention.self.value",
        "bert.encoder.layer.3.attention.self.query",
        "bert.encoder.layer.3.attention.self.value",
        "bert.encoder.layer.4.attention.self.query",
        "bert.encoder.layer.4.attention.self.value",
        "bert.encoder.layer.5.attention.self.query",
        "bert.encoder.layer.5.attention.self.value",
        "bert.encoder.layer.6.attention.self.query",
        "bert.encoder.layer.6.attention.self.value",
        "bert.encoder.layer.7.attention.self.query",
        "bert.encoder.layer.7.attention.self.value",
        "bert.encoder.layer.8.attention.self.query",
        "bert.encoder.layer.8.attention.self.value",
        "bert.encoder.layer.9.attention.self.query",
        "bert.encoder.layer.9.attention.self.value",
        "bert.encoder.layer.10.attention.self.query",
        "bert.encoder.layer.10.attention.self.value",
        "bert.encoder.layer.11.attention.self.query",
        "bert.encoder.layer.11.attention.self.value"
    ],  # Example target modules in attention layers
)

# Apply LoRA to the model
lora_model = get_peft_model(model, lora_config)

from datasets import load_dataset

# Load your dataset (JSON format example)

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# Tokenize the dataset
def tokenize_function(examples):
    # Example: Tokenize both 'turn_usr' and 'turn_sys'
    print("[USR] " + str(examples['turn_usr']) + " [SYS] " + str(examples['turn_sys']))
    return tokenizer("[USR] " + examples['turn_usr'] + " [SYS] " + examples['turn_sys'], padding='max_length', truncation=True)


dataset = load_dataset('json', data_files={'train': 'train_final.json', 'validation': 'val_final.json', 'test': 'test_final.json'})

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Initialize the Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()

lora_model.save_pretrained("./fine-tuned-todbert-lora")
tokenizer.save_pretrained("./fine-tuned-todbert-lora")

from transformers import pipeline

# Load the fine-tuned LoRA model
lora_model = PeftModel.from_pretrained("./fine-tuned-todbert-lora", model)
classifier = pipeline('text-classification', model=lora_model, tokenizer=tokenizer)

# Use the classifier for predictions
text = "Your input dialogue here"
result = classifier(text)
print(result)
