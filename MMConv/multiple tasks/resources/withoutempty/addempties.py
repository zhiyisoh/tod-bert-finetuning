import json
from collections import defaultdict
def add_keys_to_json(input_file, output_file, extra_keys):
    # Read the JSON content from the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)  # Load JSON data
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return
    except FileNotFoundError:
        print(f"The file {input_file} was not found.")
        return

    # Check if json_data is a list or dict and add extra keys
    if isinstance(json_data, list):
        for entry in json_data:
            if isinstance(entry, dict):  # Ensure it's a dictionary
                for key in extra_keys:
                    if key not in entry:  # Only add if the key does not exist
                        entry[key] = ""  # Add extra key with an empty string value
    elif isinstance(json_data, dict):
        for key in extra_keys:
            if key not in json_data:  # Only add if the key does not exist
                json_data[key] = ""  # Add extra key with an empty string value
    else:
        print("Unexpected JSON structure.")
        return

    for entry in json_data:
        if len(entry["turn_label"]) == 0:
            entry["turn_label"] = []

    new_json = {"dialogues": []}
    # Grouping data by 'category'
    grouped_data = defaultdict(list)
    # Iterate over the data and group by 'category'
    for item in json_data:
        category = item['dialogue_id']
        grouped_data[category].append(item)
    # Convert to a regular dictionary if needed
    grouped_data = dict(grouped_data)
    for dialogue, turns in grouped_data.items():
        for t in turns:
            del t["dialogue_id"]
        new_json["dialogues"].append({"dialogue_id" : dialogue, "turns" : turns})
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(new_json, json_file, indent=4)

# Example usage
extra_keys_to_add = [
    "system_transcript", "transcript", "system_acts"
]  # List of keys to add
add_keys_to_json('train.json', 'train_final.json', extra_keys_to_add)
add_keys_to_json('test.json', 'test_final.json', extra_keys_to_add)
add_keys_to_json('val.json', 'val_final.json', extra_keys_to_add)