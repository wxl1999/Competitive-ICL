import pandas as pd
from tqdm import tqdm
import json
import os
import pdb

models = ["minicpm"]
ckpt_intervals = ["60B"]
dataset_seeds = ["13", "21", "42", "87", "100"]

all_pretraining_steps = [
    list(range(30000, 350001, 20000)),
]


tasks = [
    # ===== sentiment analysis =====
    "glue-sst2",
    "financial_phrasebank",
    "tweet_eval-emotion",
    "poem_sentiment",
    
    # ===== NLI / paraphrase detection =====
    "sick",
    "glue-mrpc", 
    "snli",
    "glue-wnli",
    
    # ===== Topic Classification =====
    "trec", 
    
    # ===== Stance Classification =====
    "tweet_eval-stance_atheism", 
    "tweet_eval-stance_feminist", 
    
    # ===== Hate speech detection =====
    "tweet_eval-hate",
    "ethos-gender",
    "ethos-race",
    "ethos-national_origin",
    "ethos-religion",
]

shot_nums = ["16"]

def write_jsonl(data, file_name):
    with open(file_name, 'w') as f:
        for d in data:
            json_str = json.dumps(d)
            f.write(json_str + "\n")

def check_competitive_relation(tr_acc_list, tl_acc_list, alpha):
    
    assert len(tr_acc_list) == len(tl_acc_list)
            
    rigor_count = 0
    rigor_list = []
    for i in range(1, len(tr_acc_list)):
        if tr_acc_list[i] > tr_acc_list[i-1] + alpha and tl_acc_list[i] + alpha < tl_acc_list[i-1]:
            rigor_count += 1
            rigor_list.append(1)
        elif tr_acc_list[i] + alpha < tr_acc_list[i-1] and tl_acc_list[i] > tl_acc_list[i-1] + alpha:
            rigor_count += 1
            rigor_list.append(1)
        else:
            rigor_list.append(0)
            
    
    soft_performance_list = []
    down_up_list = []    
    
    for i in range(1, len(tr_acc_list)):
        if (tr_acc_list[i] > tr_acc_list[i-1] + alpha and tl_acc_list[i] + alpha < tl_acc_list[i-1]):
            per_down = tl_acc_list[i-1] - tl_acc_list[i]
            per_up = tr_acc_list[i] - tr_acc_list[i-1]
            
            per_delta = per_down / per_up
            soft_performance_list.append(round(per_delta, 2))
            down_up_list.append((round(per_down, 2), round(per_up, 2)))
        
        elif (tr_acc_list[i] + alpha < tr_acc_list[i-1] and tl_acc_list[i] > tl_acc_list[i-1] + alpha):
            per_down = tr_acc_list[i-1] - tr_acc_list[i]
            per_up = tl_acc_list[i] - tl_acc_list[i-1]
            
            per_delta = per_down / per_up
            soft_performance_list.append(round(per_delta, 2))
            down_up_list.append((round(per_down, 2), round(per_up, 2)))
            
        else:
            soft_performance_list.append(0)
            down_up_list.append((0, 0))
    
    return len(tr_acc_list)-1, rigor_count, rigor_list, soft_performance_list, down_up_list

for model, ckpt_interval, pretraining_steps in zip(models, ckpt_intervals, all_pretraining_steps):
    for task in tqdm(tasks):
        for shot_num in shot_nums:        
            results = []
            for dataset_seed in dataset_seeds:          
                tr_acc = []
                tl_acc = []                
                for pretraining_step in pretraining_steps:  
                    root_path = f"outputs-merge/{task}/{dataset_seed}/{model}"
                    shot_num_path = f"{root_path}/{pretraining_step}/{shot_num}/minimal"
                    tr_path = f"{shot_num_path}/random/results.csv"
                    if not os.path.exists(tr_path):
                        tr_path = f"{shot_num_path}/random-0/results.csv"
                    if not os.path.exists(tr_path):
                        tr_path = f"outputs-merge/{task}/{dataset_seed}/{model}/{pretraining_step}/{shot_num}/minimal/random-0/results.csv"
                    tl_path = f"{shot_num_path}/abstract_symbols/results.csv"
                    if not os.path.exists(tl_path):
                        tl_path = f"{shot_num_path}/abstract_symbols-0/results.csv"                     
                                     
                    tr_df = pd.read_csv(tr_path)
                    tl_df = pd.read_csv(tl_path)
                    
                    # task recognition
                    tr_acc_list = list(tr_df["accuracy"])
                    tr_score = round(sum(tr_acc_list) / len(tr_acc_list) * 100, 2)
                    tr_acc.append(tr_score)
                    
                    # task learning
                    tl_acc_list = list(tl_df["accuracy"])
                    tl_score = round(sum(tl_acc_list) / len(tl_acc_list) * 100, 2)
                    tl_acc.append(tl_score)
                    
                    # random
                    label_num = len(list(set(tr_df["gt_answer"])))
                    random_score = round(1 / label_num * 100, 2)
                
                total, competitive_num_0, competitive_list0, soft_performance, down_up_list = check_competitive_relation(tr_acc, tl_acc, 0.1)            
                
                results.append({
                    "model": model,
                    "task": task,
                    "dataset_seed": dataset_seed,
                    "shot_num": shot_num,
                    "tr_acc": tr_acc,
                    "tl_acc": tl_acc,
                    "random_score": random_score,
                    "total_num": total,
                    "competitive_num_0": competitive_num_0,
                    "competitve_list_0": competitive_list0,
                })

        save_dir = f"results/{model}/{ckpt_interval}/{task}/{shot_num}"
        os.makedirs(save_dir, exist_ok=True)
        write_jsonl(results, f"{save_dir}/results.jsonl")
    