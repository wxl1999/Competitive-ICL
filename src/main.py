import os
import sys
import json
import datetime
import logging
logging.getLogger("ray").setLevel(logging.ERROR)

ICL_EXPLORE_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, ICL_EXPLORE_ROOT_PATH)

from src.args import get_args
from src.evaluation.evaluation import evaluate
from src.utils.prompt_utils import (
    transfor_model2model_path,
    call_vllm_server_func,
    load_vllm_llm,
)

def main():
    
    icl_explore_kwargs = get_args()
    dataset_names = icl_explore_kwargs["dataset_names"]
    dataset_names = dataset_names.split(',')
    model_name = icl_explore_kwargs["model_name"]
    pretraining_steps = icl_explore_kwargs["pretraining_steps"]
    shot_nums = icl_explore_kwargs["shot_nums"]
    shot_nums = shot_nums.split(',')
    shot_nums = [int(shot_num) for shot_num in shot_nums]
    dataset_seeds = icl_explore_kwargs["dataset_seeds"]
    dataset_seeds = dataset_seeds.split(',')
    prompt_format = icl_explore_kwargs["prompt_format"]
    evaluation_types = icl_explore_kwargs["evaluation_types"]
    evaluation_types = evaluation_types.split(',')
    sep_symbol = icl_explore_kwargs["sep_symbol"]
    random_seed = icl_explore_kwargs["random_seed"]
    gpus = icl_explore_kwargs["gpus"]
    debug = icl_explore_kwargs["debug"]
    
    # ======================= Load Model =======================
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    model_path = transfor_model2model_path(model_name, pretraining_steps)
    print(model_path)
    llm = load_vllm_llm(
        model_path,
        revision=f"step{pretraining_steps}",
        tensor_parallel_size=len(gpus),
    )
    
    for dataset_name in dataset_names:
        for dataset_seed in dataset_seeds:
            for shot_num in shot_nums:
                for evaluation_type in evaluation_types:
        
                    # ====================== Save config ======================
                    save_folder = os.path.join(
                        ICL_EXPLORE_ROOT_PATH,
                        "outputs-merge",
                        f"{dataset_name}",
                        f"{dataset_seed}",
                        f"{model_name}",
                        f"{pretraining_steps}",
                        f"{shot_num}",
                        f"{prompt_format}",
                        f"{evaluation_type}-{random_seed}",
                    )
                    os.makedirs(save_folder, exist_ok=True)
                    
                    # ====================== Evaluate ======================
                    detailed_results_df = evaluate(model_name=model_name, pretraining_steps=pretraining_steps, dataset_name=dataset_name, dataset_seed=dataset_seed, shot_num=shot_num, prompt_format=prompt_format, evaluation_type=evaluation_type, sep_symbol=sep_symbol, random_seed=random_seed, gpus=gpus, llm=llm, debug=debug)
                    detailed_results_df.to_csv(os.path.join(save_folder, "results.csv"), index=True, header=True)
                    configs_dict = {
                        "model_name": model_name,
                        "pretraining_steps": pretraining_steps,
                        "dataset_name": dataset_name,
                        "dataset_seed": dataset_seed,
                        "shot_num": shot_num,
                        "prompt_format": prompt_format,
                        "evaluation_type": evaluation_type,
                        "sep_symbol": sep_symbol,
                        "gpus": gpus,
                    }
                    with open(os.path.join(save_folder, "configs_dict.json"), "w") as f:
                        json.dump(configs_dict, f, indent=4)
    
if __name__ == '__main__':
    main()