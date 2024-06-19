#!/bin/bash

MODEL_NAMES=("pythia-2.8b" "pythia-6.9b" "pythia-12b")
PRETRAINING_STEPS=(4000 13000 21000 30000 39000 47000 56000 65000 73000 82000 91000 99000 108000 117000 125000 134000 143000)
DATASET_NAMES="glue-sst2,financial_phrasebank,tweet_eval-emotion,poem_sentiment,trec,tweet_eval-stance_atheism,tweet_eval-stance_feminist,sick,glue-mrpc,snli,glue-wnli,tweet_eval-hate,ethos-gender,ethos-race,ethos-national_origin,ethos-religion"
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