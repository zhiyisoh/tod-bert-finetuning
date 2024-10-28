import json
import re
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data
dataset = load_json_data('train_final.json')

slots = [
    "music",
    "telephone",
    "venueaddress",
    "venuescore",
    "wi-fi",
    "smoking",
    "restroom",
    "parking",
    "outdoor seating",
    "reservations",
    "drinks",
    "menus",
    "venueneigh",
    "wheelchair accessible",
    "price",
    "credit cards",
    "venuename",
    "dining options"
]

for data in dataset:
    utterance = data["turn_usr"]
    if utterance == "":
        continue
    with open("C:/Users/Zhiyi/Desktop/NLC/project/tod-bert-finetuning/MMConv/distilbert/newdata/seq.in", 'a', encoding='utf-8') as f:
        f.write(utterance)
        f.write("\n")
    belief = data["belief"]
    tokens = re.findall(r'\b\w+\b', utterance) # Basic space-based tokenization
    bio_annotations = []
        # Create a new list with replacements
    modified_tokens = []
    substrings = belief.values()
    # Iterate through the tokens
    i = 0
    while i < len(tokens):
        found = False
        # Check if the substring matches starting at the current index
        for substring in substrings:
            substring_tokens = substring.split()
            substring_length = len(substring_tokens)

            # Check if the substring matches starting at the current index
            if ' '.join(tokens[i:i + substring_length]) == substring:
                key = next((k for k, v in belief.items() if v == substring), None)
                # If it matches, append 'x' for each word in the substring
                modified_tokens.extend([f"B-{key}"])
                modified_tokens.extend([f"I-{key}"] * (substring_length - 1))
                i += substring_length  # Skip over the tokens that are part of the substring
                found = True  # Mark that a substring was found
                break  # Break out of the for loop to continue with the next token

        if not found:
            # If no substring matched, append 'y' for this token
            modified_tokens.append("0")
            i += 1  # Move to the next token
    result_string = ' '.join(modified_tokens)
    with open("C:/Users/Zhiyi/Desktop/NLC/project/tod-bert-finetuning/MMConv/distilbert/newdata/seq.out", 'a', encoding='utf-8') as f:
        f.write(str(result_string))
        f.write("\n")