import os
import json
from collections import defaultdict

def load_ontologies(ontology_folder):
    """
    Load all ontology JSON files from the specified folder and merge them into a single ontology dictionary.
    """
    ontology = {}
    for filename in os.listdir(ontology_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(ontology_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                ontology_data = json.load(f)
                # Merge the ontology data
                for domain, items in ontology_data.items():
                    if domain not in ontology:
                        ontology[domain] = items
                    else:
                        ontology[domain].extend(items)
    return ontology

def extract_slots_from_ontology(ontology):
    """
    Extract slots for each domain from the ontology dictionary.
    """
    ontology_slots = {}
    for domain, items in ontology.items():
        slots = []
        for item in items:
            prefix = item['prefix']
            for annotation in item['annotations']:
                slot = f"{prefix}.{annotation}"
                slots.append(slot)
        ontology_slots[domain] = slots
    return ontology_slots

def load_json_files(data_folder):
    """
    Load all JSON data files from the specified folder and add domain information to each dialogue.
    Assumes that the filename (without extension) corresponds to the domain name.
    """
    data = []
    for filename in os.listdir(data_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(data_folder, filename)
            domain = filename.replace('.json', '')  # Assuming file name is domain name
            with open(file_path, 'r', encoding='utf-8') as f:
                dialogues = json.load(f)
                # Add domain information to each dialogue
                for dialogue in dialogues:
                    dialogue['domain'] = domain
                data.extend(dialogues)
    return data

def transform_data(taskmaster_data, ontology_slots):
    """
    Transform the Taskmaster data into the format required by your model.
    """
    transformed_data = []
    ID_counter = 1

    for dialogue in taskmaster_data:
        utterances = dialogue['utterances']
        belief_state = {}
        dialog_history = []
        domain = dialogue.get('domain', 'default_domain')
        slots = ontology_slots.get(domain, [])
        
        for idx, utterance in enumerate(utterances):
            speaker = utterance['speaker']
            text = utterance['text'].lower().strip()
            segments = utterance.get('segments', [])
            
            # Extract slot-value pairs from segments
            for segment in segments:
                annotations = segment.get('annotations', [])
                for annotation in annotations:
                    slot_name = annotation['name']
                    slot_value = segment['text'].lower().strip()
                    # Ensure the slot is in the ontology
                    if slot_name in slots:
                        belief_state[slot_name] = slot_value
                    else:
                        # Handle slots not in the ontology
                        continue
            
            if speaker == 'USER':
            #if speaker == 'ASSISTANT':
                turn_usr = text
                # Get the previous assistant utterance, if any
                turn_sys = ''
                if idx > 0 and utterances[idx - 1]['speaker'] == 'ASSISTANT':
                    turn_sys = utterances[idx - 1]['text'].lower().strip()
                
                # Prepare slot_gate and slot_values
                slot_gate = []
                slot_values = []
                for slot in slots:
                    if slot in belief_state:
                        slot_gate.append(1)
                        slot_values.append(belief_state[slot])
                    else:
                        slot_gate.append(0)
                        slot_values.append('none')
                
                transformed_turn = {
                    "ID": ID_counter,
                    "turn_id": idx,
                    "turn_sys": turn_sys,
                    "turn_usr": turn_usr,
                    "belief": belief_state.copy(),
                    "del_belief": {},  # Include if you have deletions
                    "slot_gate": slot_gate,
                    "slot_values": slot_values,
                    "slots": slots,
                    "domain": domain,
                    "sys_act": "",  # Add if available
                    # add any other fields if needed here
                }
                transformed_data.append(transformed_turn)
                ID_counter += 1
                
                # Update dialog history if needed
                dialog_history.append(turn_sys)
                dialog_history.append(turn_usr)
                
            elif speaker == 'ASSISTANT':
                # Update belief state or other tracking if necessary
                pass
            else:
                # Handle any unexpected speaker labels
                continue

    return transformed_data

# Paths to data and ontology folders
data_folder = 'taskmaster2_dataset/data'  # Replace with your actual data folder path
ontology_folder = 'taskmaster2_dataset/ontology'  # Replace with your actual ontology folder path

# Load ontologies and extract slots
ontology = load_ontologies(ontology_folder)
ontology_slots = extract_slots_from_ontology(ontology)

# Load dialogues
taskmaster_data = load_json_files(data_folder)

# Transform data
transformed_data = transform_data(taskmaster_data, ontology_slots)

# Save transformed data
output_file = 'FINALtransformed_data.json'  # Replace with your desired output file path
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(transformed_data, f, indent=2)
