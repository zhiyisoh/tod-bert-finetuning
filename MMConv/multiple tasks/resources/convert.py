import json
import re


def convert_txt_to_json(txt_file, output_file):
    with open(txt_file, 'r', encoding='utf-8') as file:
        # Read the contents of the file
        content = file.read()

        # Replace single quotes with double quotes for valid JSON
        # content = content.replace("'", '"')

        # Ensure keys and string values are properly formatted
        content = re.sub(r'\s*,\s*', ',', content)
        content = content[:-1]
        content = "[" + content + "]"
        # Write to a JSON file
        try:
            with open(output_file, 'w', encoding='utf-8') as json_file:
                json_file.write(content)
        # Use indent for pretty formatting
            print(f"Conversion completed. Check for the output.")
            print(content)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            # print("Content causing the issue:", content)

# Convert the text file to JSON
convert_txt_to_json('train.txt', 'train.json')
