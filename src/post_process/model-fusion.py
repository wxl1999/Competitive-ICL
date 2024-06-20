import pandas as pd
from collections import defaultdict
import json
import pdb
from tqdm import tqdm
import numpy as np

tr_model = "minicpm"
tr_ckpt = "190000"
tl_model = "pythia-2.8b"
tl_ckpt = "82000"

random_guess = 0.36

tr_acc = 0.38
tl_acc = 0.45

tr_weight = (tr_acc - random_guess)
tl_weight = (tl_acc - random_guess)

new_tr_weight = tr_weight / (tr_weight + tl_weight)
new_tl_weight = tl_weight / (tr_weight + tl_weight)

print("tr_weight:", new_tr_weight)
print("tl_weight:", new_tl_weight)

tr_weight1 = new_tr_weight
tl_weight1 = new_tl_weight

root_path = "outputs-merge"

tasks = [
    # ===== sentiment analysis =====
    "glue-sst2",
    "financial_phrasebank",
    "tweet_eval-emotion",
    "poem_sentiment",
    
    # ===== NLI / paraphrase detection =====
    "sick",
    "glue-mrpc",
    "glue-wnli",
    "snli",
    
    # ===== Topic / Stance Classification =====
    "trec",
    "tweet_eval-stance_atheism",
    "tweet_eval-stance_feminist",
    
    # ===== Hate speech detection =====
    "tweet_eval-hate",
    "ethos-gender",
    "ethos-race",
    "ethos-national_origin",
    "ethos-religion",
]

dataset_seeds = ["13", "21", "42", "87", "100"]
shot_nums = ["16"]

def read_jsonl(path):
    data = []
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            json_object = json.loads(line)
            data.append(json_object)

    return data

tr_tl = []

for task in tqdm(tasks):
    for shot_num in shot_nums:
        logits_tr_tl = []
        logits_tr6 = []
        logits_tl6 = []
        logits_tr6_tl6 = []
        for dataset_seed in dataset_seeds:
            
            tr_path1 = f"{root_path}/{task}/{dataset_seed}/{tr_model}/{tr_ckpt}/{shot_num}/minimal/random-0/results.csv"
            tl_path1 = f"{root_path}/{task}/{dataset_seed}/{tl_model}/{tl_ckpt}/{shot_num}/minimal/abstract_symbols-0/results.csv"
            
            tr_df1 = pd.read_csv(tr_path1)
            tl_df1 = pd.read_csv(tl_path1)
            
            # construct abstract mapping
            data_path = f"data/data-{shot_num}/{task}/{task}_{shot_num}_{dataset_seed}_test.jsonl"
            tl_data = read_jsonl(data_path)
            to_tf = {
                "entailment": "True",
                "not_entailment": "False",
                "equivalent": "True",
                "not_equivalent": "False",
                "contradiction": "False",
                "neutral": "Unknown",
                "Unknown": "Unknown",
                "Yes": "True",
                "No": "False",
            }
            tl_gt = [str(to_tf[tl["output"]]) if tl["output"] in to_tf and task in ["sick", "glue-mrpc", "glue-wnli", "snli"] else str(tl["output"]) for tl in tl_data]
            
            symbol_mapping_dict1 = {}
            number_mapping_dict1 = {}
            letter_mapping_dict1 = {}
            
            symbol_mapping_dict2 = {}
            number_mapping_dict2 = {}
            letter_mapping_dict2 = {}
            
            symbol_gt1 = list(tl_df1["gt_answer"])
            
            for i, (symbol, natural_language) in enumerate(zip(symbol_gt1, tl_gt)):
                symbol_mapping_dict1[str(symbol)] = str(natural_language)
            
            tr_options_list1 = []
            tl_options_list1 = []
            
            tr_options = tr_df1.columns[6:].tolist()
            tr_options = [str(option) for option in tr_options]
            
            tl_options1 = tl_df1.columns[6:].tolist()
            tl_options1 = [str(option) for option in tl_options1]
            
            for option in tr_options:
                tr_options_list1.append({str(option): list(tr_df1[option])})
            
            for option in tl_options1:
                tl_options_list1.append({symbol_mapping_dict1[option]: list(tl_df1[option])})
            
            # tr + tl
            results_tr_tl = defaultdict(list)
            
            for tr_option, tl_option, option in zip(tr_options_list1, tl_options_list1, tr_options):
                tr_scores = tr_option[option]
                tl_scores = tl_option[option]
                for tr_score, tl_score in zip(tr_scores, tl_scores):
                    average_score = tr_score * tr_weight1 + tl_score * tl_weight1
                    results_tr_tl[option].append(average_score)
            
            logits_tr_tl_merge_results = []
            
            for i, gt_label in enumerate(tl_gt):
                current_i = {}
                for k, v in results_tr_tl.items():
                    current_i[k] = v[i]
                max_key = max(current_i, key=current_i.get)
                if max_key == gt_label or max_key == gt_label.lower():
                    logits_tr_tl_merge_results.append(1)
                else:
                    logits_tr_tl_merge_results.append(0)
            
            logits_tr_tl_merge_acc = sum(logits_tr_tl_merge_results) / len(logits_tr_tl_merge_results)
            
            logits_tr_tl.append(logits_tr_tl_merge_acc)
            
        
        task_tr_tl = sum(logits_tr_tl) / len(logits_tr_tl) * 100
        tr_tl.append(task_tr_tl)
        
all_acc_tr_tl = round(sum(tr_tl) / len(tr_tl), 2)
print("tr+tl:", all_acc_tr_tl)