import json
import ast
import collections
import os

from .utils_function import get_input_example


def read_langs_turn(args, dials, ds_name, max_line):
    print(("Reading from {} for read_langs_turn".format(ds_name)))
    
    data = []
    turn_sys = ""
    turn_usr = ""
    
    cnt_lin = 1
    for dial in dials:
        dialog_history = []
        for ti, turn in enumerate(dial["utterances"]):
            if turn["speaker"] == "USER":
                turn_usr = turn["text"].lower().strip()
                
                data_detail = get_input_example("turn")
                data_detail["ID"] = "{}-{}".format(ds_name, cnt_lin)
                data_detail["turn_id"] = ti % 2
                data_detail["turn_usr"] = turn_usr
                data_detail["turn_sys"] = turn_sys
                data_detail["dialog_history"] = list(dialog_history)
                
                if (not args["only_last_turn"]):
                    data.append(data_detail)
                
                dialog_history.append(turn_sys)
                dialog_history.append(turn_usr)
            elif turn["speaker"] == "ASSISTANT":
                turn_sys = turn["text"].lower().strip()
            else:
                turn_usr += " {}".format(turn["text"])
        
        if args["only_last_turn"]:
            data.append(data_detail)
        
        cnt_lin += 1
        if(max_line and cnt_lin >= max_line):
            break

    return data


def read_langs_dial(file_name, ontology, dialog_act, max_line = None, domain_act_flag=False):
    print(("Reading from {} for read_langs_dial".format(file_name)))
    
    raise NotImplementedError



def prepare_data_taskmaster(args):
    ds_name = "TaskMaster2"
    
    example_type = args["example_type"]
    max_line = args["max_line"]
    
   # Path to Taskmaster-2 data directory
    data_dir = os.path.join(args["data_path"], 'Taskmaster2/data')  # Adjust the path based on your data location
    
    # Load all dialogues from Taskmaster-2 data files
    dials_all = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                dialogues = json.load(f)
                dials_all.extend(dialogues)
    
    # Split data into train/dev/test sets (since Taskmaster-2 doesn't provide splits)
    from sklearn.model_selection import train_test_split
    train_dials, temp_dials = train_test_split(dials_all, test_size=0.2, random_state=42)
    dev_dials, test_dials = train_test_split(temp_dials, test_size=0.5, random_state=42)
    
    _example_type = "dial" if "dial" in example_type else example_type
    
    # Process the dialogues to create training examples
    pair_trn = globals()["read_langs_{}".format(_example_type)](args, train_dials, ds_name, max_line)
    pair_dev = globals()["read_langs_{}".format(_example_type)](args, dev_dials, ds_name, max_line)
    pair_tst = globals()["read_langs_{}".format(_example_type)](args, test_dials, ds_name, max_line)
    
    print("Read {} pairs train from {}".format(len(pair_trn), ds_name))
    print("Read {} pairs valid from {}".format(len(pair_dev), ds_name))
    print("Read {} pairs test  from {}".format(len(pair_tst), ds_name))
    
    meta_data = {"num_labels": 0}

    return pair_trn, pair_dev, pair_tst, meta_data

