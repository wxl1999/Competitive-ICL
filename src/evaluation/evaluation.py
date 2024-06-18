import os
import sys
import random
import pdb
from tqdm import tqdm
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict

import logging

logging.getLogger("ray").setLevel(logging.ERROR)

ICL_EXPLORE_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ICL_EXPLORE_ROOT_PATH)

from src.dataset.base_dataset import load_dataset
from src.utils.prompt_utils import (
    transfor_model2model_path,
    call_vllm_server_func,
    load_vllm_llm,
)


def evaluate_results(results, gt_labels):
    accuracy_list = []
    for result, gt_label in zip(results, gt_labels):
        accuracy_list.append(int(result == gt_label))
    return accuracy_list


def evaluate(
    model_name,
    pretraining_steps,
    dataset_name,
    dataset_seed,
    shot_num,
    prompt_format,
    evaluation_type,
    sep_symbol,
    random_seed,
    gpus,
    llm,
    debug=None,
):
    # ====================== environment setting ======================
    np.random.seed(0)
    random.seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    # ======================= Load Dataset =======================
    DATASETS_class = load_dataset(
        dataset_name=dataset_name,
        seed=dataset_seed,
        few_shot_numbers=shot_num,
        prompt_format=prompt_format,
        evaluation_type=evaluation_type,
        sep_symbol=sep_symbol,
    )
    train_examples_list = DATASETS_class.load_train_examples()
    dev_samples_list = DATASETS_class.load_dev_examples()
    test_samples_list = DATASETS_class.load_test_examples()
    if debug:
        test_samples_list = test_samples_list[:5]
    
    if evaluation_type == "golden":
        examples_str = DATASETS_class.get_icl_examples(train_examples_list)
    elif evaluation_type == "random":
        random_examples = DATASETS_class.randomize_example(train_examples_list, random_seed)
        examples_str = DATASETS_class.get_icl_examples(random_examples)
    else:
        assert evaluation_type.startswith("abstract")
        example_mapping_dict, map_example_labels = DATASETS_class.map_example_labels(
            train_examples_list, random_seed
        )
        test_example_mapping_dict, test_samples_list = (
            DATASETS_class.map_example_labels(test_samples_list, random_seed)
        )
        examples_str = DATASETS_class.get_icl_examples(map_example_labels)
        
    test_options_list = DATASETS_class.get_options(test_samples_list)
    if evaluation_type.startswith("abstract"):
        test_options_list = [[test_example_mapping_dict[option] for option in test_option] for test_option in test_options_list]

    # ======================= Evaluate =======================
    all_input_str = []
    all_gt_labels = []

    all_eval_index = list(range(len(test_samples_list)))
    for test_sample in tqdm(test_samples_list, desc="process input str"):
        if prompt_format == "natural_language":
            input_str = DATASETS_class.format_input_examples(test_sample)
        else:
            assert prompt_format == "minimal"
            input_str = test_sample.input
        all_input_str.append(examples_str + sep_symbol + input_str)
        if evaluation_type == "golden" or evaluation_type == "random":
            label = test_sample.gt_label_name
        else:
            assert evaluation_type.startswith("abstract")
            label = test_sample.map_gt_label_name
        all_gt_labels.append(label)
    
    # ======================= call vllm func =======================
    
    llm_decode_steps = 256
    llm_temperature = 0
    
    specific_words = test_options_list[0]
    
    results, word_probabilities = call_vllm_server_func(
        all_input_str,
        llm,
        max_decode_steps=llm_decode_steps,
        temperature=llm_temperature,
        specific_words=specific_words,
    )

    parsed_results = [
        DATASETS_class.parse_answer(result)
        for result in tqdm(results, desc="parse answers")
    ]

    # pdb.set_trace()

    accuracy_list = evaluate_results(parsed_results, all_gt_labels)

    # ====================== save results ======================
    detailed_results_df = pd.DataFrame(
        list(
            zip(
                all_eval_index,
                all_input_str,
                results,
                parsed_results,
                all_gt_labels,
                accuracy_list,
            )
        ),
        columns=[
            "index_in_raw_dataset",
            "raw_prompt",
            "raw_answer",
            "parsed_answer",
            "gt_answer",
            "accuracy",
        ],
    )
    
    if specific_words is not None:
        options_probability_dict = defaultdict(list)
        for word_probability in word_probabilities:
            for word, prob in word_probability.items():
                options_probability_dict[word].append(prob)
    
        for i, specific_word in enumerate(specific_words):
            detailed_results_df.insert(
                6 + i, specific_word, options_probability_dict[specific_word]
            )

    detailed_results_df.set_index("index_in_raw_dataset", inplace=True)
    return detailed_results_df
