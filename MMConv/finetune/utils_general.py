import torch
import torch.utils.data as data
import random
import math

from dataloader_dst import *

def get_loader(args, mode, tokenizer, datasets, unified_meta, shuffle=False):
    task = args["task"]
    batch_size = args["batch_size"] if mode == "train" else args["eval_batch_size"]
    
    combined_ds = []
    for ds in datasets:
        combined_ds += datasets[ds][mode]
    
    # control data ratio
    if (args["train_data_ratio"] != 1 or args["nb_shots"] != -1) and (mode == "train"): 
        original_len = len(combined_ds)
        
        if ("oos_intent" in args["dataset"]):
            nb_train_sample_per_class = int(100 * args["train_data_ratio"])
            class_count = {k: 0 for k in unified_meta["intent"]}
            random.Random(args["rand_seed"]).shuffle(combined_ds)
            pair_trn_new = []
            for d in combined_ds:
                if class_count[d["intent"]] < nb_train_sample_per_class:
                    pair_trn_new.append(d)
                    class_count[d["intent"]] += 1
            combined_ds = pair_trn_new
        else:
            if args["train_data_ratio"] != 1:
                random.Random(args["rand_seed"]).shuffle(combined_ds)
                combined_ds = combined_ds[:int(len(combined_ds)*args["train_data_ratio"])]
            else:
                random.Random(args["rand_seed"]).shuffle(combined_ds)
                combined_ds = combined_ds[:args["nb_shots"]]
        print("[INFO] Use Training Data: from {} to {}".format(original_len, len(combined_ds)))
    
    data_info = {k: [] for k in combined_ds[0].keys()}
    for d in combined_ds:
        for k in combined_ds[0].keys():
            data_info[k].append(d[k])

    dataset = globals()["Dataset_"+task](data_info, tokenizer, args, unified_meta, mode, args["max_seq_length"])
    
    bool_shuffle = (mode=="train" or shuffle)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=bool_shuffle,
                                              collate_fn=globals()["collate_fn_{}_{}".format(task, args["example_type"])])
    return data_loader


def get_unified_meta(datasets):
    unified_meta = {"others":None}
    for ds in datasets:
        for key, value in datasets[ds]["meta"].items():
            if key not in unified_meta.keys():
                unified_meta[key] = {}
            if type(value) == list:
                for v in value:
                    if v not in unified_meta[key].keys():
                        unified_meta[key][v] = len(unified_meta[key])  
            else:
                unified_meta[key] = value
                
    return unified_meta
