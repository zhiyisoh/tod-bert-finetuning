meta = {"slots":{}}

import json
# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

desired_order = [
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

data = load_json_data("ontology.json")

for i in data.keys():
    name = i.lower()
    values = data[i]["options"]
    values_dict = {}
    idx = 0
    for j in values:
        values_dict.update({j : idx})
        idx += 1
    meta["slots"].update({name : values_dict})

sorted_dict = {key: meta["slots"][key] for key in desired_order if key in meta["slots"]}
meta = {"slots":sorted_dict}

with open("metadata.json", 'w') as f:
    data = json.dumps(meta)
    f.write(data)