import json

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

    # Write the updated JSON data to the output file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)

# Example usage
extra_keys_to_add = [
    'ID', 'turn_id', 'domains', 'turn_domain', 'turn_usr', 
    'turn_sys', 'turn_usr_delex', 'turn_sys_delex', 
    'belief_state_vec', 'db_pointer', 'dialog_history', 
    'dialog_history_delex', 'belief', 'del_belief', 
    'slot_gate', 'slot_values', 'slots', 'sys_act', 
    'usr_act', 'intent', 'turn_slot'
]  # List of keys to add
add_keys_to_json('train.json', 'train_final.json', extra_keys_to_add)
add_keys_to_json('test.json', 'test_final.json', extra_keys_to_add)
add_keys_to_json('val.json', 'val_final.json', extra_keys_to_add)