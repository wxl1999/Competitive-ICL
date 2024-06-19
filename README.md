# Investigating the Pre-Training Dynamics of In-Context Learning: Task Recognition vs. Task Learning

This repo provides the source code & data of our paper: Investigating the Pre-Training Dynamics of In-Context Learning: Task Recognition vs. Task Learning.

## Overview 😊

+ To the best of our knowledge, this is the first time that the competitive relationship between the two abilities of ICL (\ie TR and TL) and its emergence has been investigated.
By examining the pre-training dynamics of ICL, we demonstrate a strong negative correlation between the emergence of ICL and the competition between TR and TL.

+ We conduct a fine-grained analysis of common pre-training factors (\ie model size, dataset size, and data curriculum) to understand their influence on the competition between TR and TL.

+ We propose a simple but effective method to better integrate TR and TL for ICL at inference time.
Through adaptive ensemble learning, the performance of ICL can be significantly boosted, enabling two small models to outperform a larger one with more than twice the parameters.

## Installation 🚀

transformers = 4.41.1

vllm = 0.4.2

## Reproducing Experiments 🖊️

**Model**: We use the Pythia suite (6 model sizes ranging from 410M to 12B), MiniCPM-2B, Baichuan2-7B, Amber-7B, CrystalCoder-7B, and OLMo-7B.

**Dataset**: We conduct experiments on four types of tasks: Sentiment Analysis, Topic/Stance Classification, Toxicity Detection, and Natural Language Inference/Paraphrase Detection. We follow [this](https://github.com/Alrope123/rethinking-demonstrations) to download the dataset.

For *Sentiment Analysis*, we use datasets including SST-2, financial_phrasebank, emotion, and poem_sentiment.

For *Topic/Stance Classification*, we utilize TREC, tweet_eval_atheist, and tweet_eval_feminist.

For *Toxicity Detection*, we include tweet_eval_hate, ethos_race, ethos_gender, ethos_national_origin, and ethos_religion.

For *Natural Language Inference/Paraphrase Detection*, we employ SICK, SNLI, WNLI, and MRPC.

**Shot num**: 16

**Abstract setting**: Abstract symbol

We report the results across **five random seeds** and **16 datasets**.

## Run experiments 👋

### Parameters

gpus: The number of GPUs

model_name: The name of the model (pythia-410m, pythia-1b, pythia-1.4b, pythia-2.8b, pythia-6.9b, pythia-12b, minicpm, baichuan2, amber, crystalcoder, olmo)

pretraining_steps: The step of the checkpoint

dataset_names: The name of the dataset (glue-sst2,financial_phrasebank,tweet_eval-emotion,poem_sentiment,trec,tweet_eval-stance_atheism,tweet_eval-stance_feminist,sick,glue-mrpc,snli,glue-wnli,tweet_eval-hate,ethos-gender,ethos-race,ethos-national_origin,ethos-religion)

dataset_seeds: The seed of demonstration

shot_nums: The number of demonstrations

sep_symbol: The character to separate demonstrations

evaluation_types: evaluation types (golden, random, abstrasct_symbols)

### Model path

Change your path in Line 12 in src/utils/utils.py

### Example

```
bash scripts/example.sh
```

## Contact 📮

If you have any questions for our paper or codes, please send an email to txy20010310@163.com.