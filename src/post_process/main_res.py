import json
import os

from collections import defaultdict, Counter
from itertools import zip_longest
import matplotlib.pyplot as plt
from itertools import accumulate

def check_competitive_relation(tr_acc_list, tl_acc_list, alpha):
    
    assert len(tr_acc_list) == len(tl_acc_list)
    if len(tr_acc_list) == 0 or len(tl_acc_list) == 0:
        return 0, 0, []
            
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
            soft_performance_list.append(per_delta)
            down_up_list.append((per_down, per_up))
        
        elif (tr_acc_list[i] + alpha < tr_acc_list[i-1] and tl_acc_list[i] > tl_acc_list[i-1] + alpha):
            per_down = tr_acc_list[i-1] - tr_acc_list[i]
            per_up = tl_acc_list[i] - tl_acc_list[i-1]
            
            per_delta = per_down / per_up
            soft_performance_list.append(per_delta,)
            down_up_list.append((per_down, per_up))
            
        else:
            soft_performance_list.append(0)
            down_up_list.append((0, 0))
    
    return len(tr_acc_list)-1, rigor_count, rigor_list, soft_performance_list, down_up_list


datasets = [
    "glue-sst2",
    "financial_phrasebank",
    "tweet_eval-emotion",
    "poem_sentiment",
    "sick",
    "snli",
    "glue-wnli",
    "glue-mrpc",
    "trec",
    "tweet_eval-stance_atheism",
    "tweet_eval-stance_feminist",
    "tweet_eval-hate",
    "ethos-gender",
    "ethos-race",
    "ethos-national_origin",
    "ethos-religion",
]


def read_jsonl(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            json_dict = json.loads(line)
            data.append(json_dict)
    return data
        

# ============== Model Scale =================

models = ["minicpm"]
ckpt_intervals = ["60B"]
shot_num = "16"
dataset_seeds = ["13", "21", "42", "87", "100"]

competitive_hard = []
competitive_soft = []
alphas = [0.01]

for alpha in alphas:
    print(alpha)
    model_avg_soft = []
    for model, ckpt_interval in zip(models, ckpt_intervals):
        
        all_tr_acc = defaultdict(list)
        all_tl_acc = defaultdict(list)
        
        for dataset in datasets:
                
            result_path = f"results/{model}/{ckpt_interval}/{dataset}/{shot_num}/results.jsonl"
            result_data = read_jsonl(result_path)
            for dataset_seed, data in zip(dataset_seeds, result_data):
                shot_num = data["shot_num"]
                tr_acc = data["tr_acc"]
                tl_acc = data["tl_acc"]
                all_tr_acc[f"{model}_{shot_num}"].append(tr_acc)
                all_tl_acc[f"{model}_{shot_num}"].append(tl_acc)
        
        tr_acc_list = all_tr_acc[f"{model}_{shot_num}"]
        tl_acc_list = all_tl_acc[f"{model}_{shot_num}"]
        column_tr_sums = [sum(x) / len(tr_acc_list) for x in list(zip(*tr_acc_list))]
        column_tl_sums = [sum(x) / len(tl_acc_list) for x in list(zip(*tl_acc_list))]
        
        all_length, competitive_num, competitve_list, soft_p_list, down_up_list = check_competitive_relation(column_tr_sums, column_tl_sums, alpha)
        
        model_avg_soft.append(round(sum(soft_p_list) / len(soft_p_list), 2))
                
        print("%-15s" % model, "%-3s" % shot_num, "%-3s" % all_length, "%-3s" % competitive_num, "%-30s" % competitve_list, "%-48s" % soft_p_list, sum(soft_p_list) / len(soft_p_list))
        
        soft_p = [round(p, 2) for p in soft_p_list]
        print("soft:", soft_p)
        
        soft_p_list.insert(0, 0)
        all_sum = sum(soft_p_list)
        
        cumulative_sums = list(accumulate(soft_p_list))
        cumulative_per_sums = [round(s / all_sum * 100, 2) for s in cumulative_sums]
        
        print("cum:", cumulative_per_sums)
    
