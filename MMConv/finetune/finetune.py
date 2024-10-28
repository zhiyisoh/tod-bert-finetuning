import json
import torch
import ast
import glob
import os
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader
from transformers import *
import dataloader_dst
import utils_general
from tqdm import tqdm
from models.multi_label_classifier import *
from models.multi_class_classifier import *
from models.BERT_DST_Picklist import *
from models.dual_encoder_ranking import *
import logging as lg
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
    
MODELS = {"bert": (BertModel,       BertTokenizer,       BertConfig),
          "todbert": (BertModel,       BertTokenizer,       BertConfig),
          "gpt2": (GPT2Model,       GPT2Tokenizer,       GPT2Config),
          "todgpt2": (GPT2Model,       GPT2Tokenizer,       GPT2Config),
          "dialogpt": (AutoModelWithLMHead, AutoTokenizer, GPT2Config),
          "albert": (AlbertModel, AlbertTokenizer, AlbertConfig),
          "roberta": (RobertaModel, RobertaTokenizer, RobertaConfig),
          "distilbert": (DistilBertModel, DistilBertTokenizer, DistilBertConfig),
          "electra": (ElectraModel, ElectraTokenizer, ElectraConfig)}

SEEDS = [10, 5, 0]
args = {
    'model_type': 'todbert',
    'my_model': 'BeliefTracker',
    'usr_token': '[USR]',
    'sys_token': '[SYS]',
    'example_type': 'turn',
    'task': 'dst',
    'task_name': 'dst',
    'batch_size': 8,
    'train_data_ratio': 0.05,
    'dataset': 'MMConv',
    'rand_seed': 111,
    'max_seq_length': 256,
    'do_train': 1,
    'output_dir': 'C:/Users/Zhiyi/Desktop/NLC/project/tod-bert-finetuning/MMConv/finetune/save',
    'nb_runs': 1,
    'fix_rand_seed': 'store_true',
    'eval_batch_size': 100,
    'epoch': 1,
    'eval_by_step': 4000,
    'model_name_or_path': 'bert-base-uncased',
    'cache_dir': '',
    'n_gpu': 1,
    'patience': 10,
    'earlystop': 'joint_acc',
    'dropout': 0.2,
    'learning_rate': 5e-05,
    'hdd_size': 400,
    'emb_size': 400,
    'grad_clip': 1,
    'teacher_forcing_ratio': 0.5,
    'load_embedding': False,
    'fix_embedding': False,
    'fix_encoder': False,
    'warmup_proportion': 0.1,
    'local_rank': -1,
    'gradient_accumulation_steps': 1,
    'weight_decay': 0.0,
    'adam_epsilon': 1e-08,
    'warmup_steps': 0,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'output_mode': 'classification',
    'max_steps': -1,
    'nb_evals': 1,
    'input_name': 'context',
    'data_path': '/export/home/dialog_datasets',
    'load_path': None,
    'add_name': '',
    'max_line': None,
    'overwrite': False,
    'logging_steps': 500,
    'save_steps': 1000,
    'save_total_limit': 1,
    'domain_act': False,
    'only_last_turn': False,
    'error_analysis': False,
    'not_save_model': False,
    'nb_shots': -1,
    'do_embeddings': False,
    'create_own_vocab': False,
    'unk_mask': True,
    'parallel_decode': True,
    'self_supervised': 'generative',
    'oracle_domain': False,
    'more_linear_mapping': False,
    'gate_supervision_for_dst': False,
    'sum_token_emb_for_value': False,
    'nb_neg_sample_rs': 0,
    'sample_negative_by_kmeans': False,
    'nb_kmeans': 1000,
    'bidirect': False,
    'rnn_type': 'gru',
    'num_rnn_layers': 1,
    'zero_init_rnn': False,
    'do_zeroshot': False,
    'oos_threshold': False,
    'ontology_version': '',
    'dstlm': False,
    'vizualization': 0
}


# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Modify this to point to your actual JSON file paths
train_file = 'train_final.json'
val_file = 'val_final.json'
test_file = 'test_final.json'

# Load the JSON data
data_trn = load_json_data(train_file)
data_dev = load_json_data(val_file)
data_tst = load_json_data(test_file)
metadata = load_json_data('metadata.json')

## Create vocab and model class
args["model_type"] = args["model_type"].lower()
model_class, tokenizer_class, config_class = MODELS[args["model_type"]]
tokenizer = tokenizer_class.from_pretrained(args["model_name_or_path"], cache_dir=args["cache_dir"])
args["model_class"] = model_class
args["tokenizer"] = tokenizer
if args["model_name_or_path"]:
    config = config_class.from_pretrained(args["model_name_or_path"], cache_dir=args["cache_dir"]) 
else:
    config = config_class()
args["config"] = config

# Create datasets for train, dev, and test
datasets = {
    "MMConv" : {
                "train": data_trn,
                "dev": data_dev,
                "test": data_tst,
                "meta": metadata
                }
}

# Create DataLoaders
batch_size = 8  # Adjust batch size as needed
train_dataloader = utils_general.get_loader(args, "train", tokenizer, datasets, metadata)
# dev_dataloader = DataLoader(datasets["dev"], batch_size=batch_size, shuffle=False)
# test_dataloader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
unified_meta = metadata
args["unified_meta"] = metadata

## Training and Testing Loop
if args["do_train"]:
    result_runs = []
    output_dir_origin = str(args["output_dir"])
    
    ## Setup logger
    lg.basicConfig(level=lg.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(args["output_dir"], "train.log"),
                        filemode='w')
    console = lg.StreamHandler()
    console.setLevel(lg.INFO)
    formatter = lg.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    lg.getLogger('').addHandler(console)

    ## training loop
    for run in range(args["nb_runs"]):
         
        ## Setup random seed and output dir
        rand_seed = SEEDS[run]
        if args["fix_rand_seed"]: 
            torch.manual_seed(rand_seed)
            args["rand_seed"] = rand_seed
        args["output_dir"] = os.path.join(output_dir_origin, "run{}".format(run)) 
        os.makedirs(args["output_dir"], exist_ok=True)
        lg.info("Running Random Seed: {}".format(rand_seed))
        
        ## Loading model
        model = globals()[args['my_model']](args)
        if torch.cuda.is_available(): model = model.cuda()
        
        ## Create Dataloader
        trn_loader = utils_general.get_loader(args, "train", tokenizer, datasets, unified_meta)
        dev_loader = utils_general.get_loader(args, "dev"  , tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
        tst_loader = utils_general.get_loader(args, "test" , tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
        print(trn_loader)
        ## Create TF Writer
        tb_writer = SummaryWriter(comment=args["output_dir"].replace("/", "-").replace(":", "-"))


        # Start training process with early stopping
        loss_best, acc_best, cnt, train_step = 1e10, -1, 0, 0
        
        try:
            for epoch in range(args["epoch"]):
                lg.info("Epoch:{}".format(epoch+1)) 
                train_loss = 0
                pbar = tqdm(trn_loader)
                for i, d in enumerate(pbar):
                    print("here")
                    model.train()
                    outputs = model(d)
                    print(outputs)
                    train_loss += outputs["loss"]
                    print(train_loss)
                    train_step += 1
                    pbar.set_description("Training Loss: {:.4f}".format(train_loss/(i+1)))

                    ## Dev Evaluation
                    if (train_step % args["eval_by_step"] == 0 and args["eval_by_step"] != -1) or \
                                                  (i == len(pbar)-1 and args["eval_by_step"] == -1):
                        model.eval()
                        dev_loss = 0
                        preds, labels = [], []
                        ppbar = tqdm(dev_loader)
                        for d in ppbar:
                            with torch.no_grad():
                                outputs = model(d)
                            #print(outputs)
                            dev_loss += outputs["loss"]
                            preds += [item for item in outputs["pred"]]
                            labels += [item for item in outputs["label"]] 

                        dev_loss = dev_loss / len(dev_loader)
                        results = model.evaluation(preds, labels)
                        dev_acc = results[args["earlystop"]] if args["earlystop"] != "loss" else dev_loss

                        ## write to tensorboard
                        tb_writer.add_scalar("train_loss", train_loss/(i+1), train_step)
                        tb_writer.add_scalar("eval_loss", dev_loss, train_step)
                        tb_writer.add_scalar("eval_{}".format(args["earlystop"]), dev_acc, train_step)

                        if (dev_loss < loss_best and args["earlystop"] == "loss") or \
                            (dev_acc > acc_best and args["earlystop"] != "loss"):
                            loss_best = dev_loss
                            acc_best = dev_acc
                            cnt = 0 # reset
                            
                            if args["not_save_model"]:
                                model_clone = BertForSequenceClassification.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
                                model_clone.load_state_dict(copy.deepcopy(model.state_dict()))
                            else:
                                output_model_file = os.path.join(args["output_dir"], "pytorch_model.bin")
                                if args["n_gpu"] == 1:
                                    torch.save(model.state_dict(), output_model_file)
                                else:
                                    torch.save(model.module.state_dict(), output_model_file)
                                lg.info("[Info] Model saved at epoch {} step {}".format(epoch, train_step))
                        else:
                            cnt += 1
                            lg.info("[Info] Early stop count: {}/{}...".format(cnt, args["patience"]))

                        if cnt > args["patience"]: 
                            lg.info("Ran out of patient, early stop...")  
                            break

                        lg.info("Trn loss {:.4f}, Dev loss {:.4f}, Dev {} {:.4f}".format(train_loss/(i+1), 
                                                                                              dev_loss,
                                                                                              args["earlystop"],
                                                                                              dev_acc))

                if cnt > args["patience"]: 
                    tb_writer.close()
                    break 
                    
        except KeyboardInterrupt:
            lg.info("[Warning] Earlystop by KeyboardInterrupt")
        
        ## Load the best model
        if args["not_save_model"]:
            model.load_state_dict(copy.deepcopy(model_clone.state_dict()))
        else:
            output_model_file = os.path.join(args["output_dir"], "pytorch_model.bin")
            os.makedirs(output_dir, exist_ok=True)
            # Start evaluating on the test set
            if torch.cuda.is_available(): 
                model.load_state_dict(torch.load(output_model_file))
            else:
                model.load_state_dict(torch.load(output_model_file, lambda storage, loc: storage))
        
        ## Run test set evaluation
        pbar = tqdm(tst_loader)
        for nb_eval in range(args["nb_evals"]):
            test_loss = 0
            preds, labels = [], []
            for d in pbar:
                with torch.no_grad():
                    outputs = model(d)
                test_loss += outputs["loss"]
                preds += [item for item in outputs["pred"]]
                labels += [item for item in outputs["label"]] 

            test_loss = test_loss / len(tst_loader)
            results = model.evaluation(preds, labels)
            result_runs.append(results)
            lg.info("[{}] Test Results: ".format(nb_eval) + str(results))
    
    ## Average results over runs
    if args["nb_runs"] > 1:
        f_out = open(os.path.join(output_dir_origin, "eval_results_multi-runs.txt"), "w")
        f_out.write("Average over {} runs and {} evals \n".format(args["nb_runs"], args["nb_evals"]))
        for key in results.keys():
            mean = np.mean([r[key] for r in result_runs])
            std  = np.std([r[key] for r in result_runs])
            f_out.write("{}: mean {} std {} \n".format(key, mean, std))
        f_out.close()

else:
    
    ## Load Model
    print("[Info] Loading model from BERT")
    model = globals()[args['my_model']](args)
    if args["load_path"]:
        print("MODEL {} LOADED".format(args["load_path"]))
        if torch.cuda.is_available(): 
            model.load_state_dict(torch.load(args["load_path"]))
        else:
            model.load_state_dict(torch.load(args["load_path"], lambda storage, loc: storage))
    else:
        print("[WARNING] No trained model is loaded...")
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("[Info] Start Evaluation on dev and test set...")
    dev_loader = utils_general.get_loader(args, "dev"  , tokenizer, datasets, unified_meta)
    tst_loader = utils_general.get_loader(args, "test" , tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
    model.eval()
    
    for d_eval in ["tst"]: #["dev", "tst"]:
        f_w = open(os.path.join(args["output_dir"], "{}_results.txt".format(d_eval)), "w")

        ## Start evaluating on the test set
        test_loss = 0
        preds, labels = [], []
        pbar = tqdm(locals()["{}_loader".format(d_eval)])
        for d in pbar:
            with torch.no_grad():
                outputs = model(d)
            test_loss += outputs["loss"]
            preds += [item for item in outputs["pred"]]
            labels += [item for item in outputs["label"]] 

        test_loss = test_loss / len(tst_loader)
        results = model.evaluation(preds, labels)
        print("{} Results: {}".format(d_eval, str(results)))
        f_w.write(str(results))
        f_w.close()

# lora_model = get_peft_model(model, lora_config)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# lora_model.to(device)

# # Define optimizer
# optimizer = AdamW(lora_model.parameters(), lr=5e-5)
# epochs = 10
# # Training loop
# for epoch in range(epochs):
#     lora_model.train()  # Set model to training mode
#     total_loss = 0
#     print(train_dataloader)
#     for batch in train_dataloader:
#         # Get the input and labels from batch and move them to the device
#         input_ids = batch["context"].to(device)
#         attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
#         labels = batch["slot_gate"].to(device)

#         # Forward pass
#         outputs = lora_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss

#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_dataloader)
#     print(f'Epoch {epoch + 1}, Loss: {avg_loss}')


# lora_model.save_pretrained('./lora_fine_tuned_dst')
# tokenizer.save_pretrained('./lora_fine_tuned_dst')
