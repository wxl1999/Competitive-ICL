#!/bin/bash

MODEL_NAMES=("pythia-2.8b" "pythia-6.9b" "pythia-12b")
PRETRAINING_STEPS=(13000 26000 39000 52000 65000 78000 91000 104000 117000 130000 143000)
DATASET_NAMES="ade_corpus_v2-classification,ag_news,anli,art,blimp-anaphor_number_agreement,blimp-ellipsis_n_bar_2,blimp-sentential_negation_npi_licensor_present,blimp-sentential_negation_npi_scope,boolq"
DATASET_SEEDS=("13" "21" "42" "87" "100")
SHOT_NUMS="16"
PROMPT_FORMATS=("minimal")
EVALUATION_TYPES="abstract_symbols,random,golden"

for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    for PRETRAINING_STEP in "${PRETRAINING_STEPS[@]}"
    do
        for PROMPT_FORMAT in "${PROMPT_FORMATS[@]}"
        do
            python src/main.py \
                --model_name=$MODEL_NAME \
                --pretraining_steps=$PRETRAINING_STEP \
                --dataset_names=$DATASET_NAMES \
                --dataset_seeds=$DATASET_SEEDS \
                --shot_nums=$SHOT_NUMS \
                --prompt_format=$PROMPT_FORMAT \
                --evaluation_types=$EVALUATION_TYPES \
                --gpus 0 1
        done
    done
done