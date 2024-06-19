import time
import os
from typing import List
from transformers import Conversation
from vllm import LLM, SamplingParams
import pdb
import math

import logging
logging.getLogger("ray").setLevel(logging.ERROR)

def transfor_model2model_path(model_name: str, revision: int):
    if "pythia" in model_name:
        model_path = pythia_path
    elif "baichuan2" in model_name:
        model_path = baichuan2_path
    elif "olmo" in model_name:
        model_path = olmo_path
    elif "minicpm" in model_name:
        model_path = minicpm_path
    elif "amber" in model_name:
        model_path = amber_path
    return model_path

def call_vllm_server_func(
    prompt, model, specific_words=None, max_decode_steps=128, temperature=0, stop_tokens=None
):
    """The function to call vllm with a list of input strings or token IDs."""
    
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_decode_steps, stop=stop_tokens, logprobs=5
    )
    res_completions = []
    word_probabilities = []

    if isinstance(prompt, str):
        prompt = [prompt]

    if isinstance(prompt, list):
        if all(isinstance(elem, str) for elem in prompt):
            completions = model.generate(prompts=prompt, sampling_params=sampling_params)
        elif all(isinstance(sublist, list) and all(isinstance(item, int) for item in sublist) for sublist in prompt):
            completions = model.generate(prompt_token_ids=prompt, sampling_params=sampling_params)
        
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            logprobs = output.outputs[0].logprobs
            res_completions.append(generated_text)
            
            word_probability = {}
            
            for step, probs in enumerate(logprobs):
                step_probs = logprobs[step]
                for token_ids, prob_info in step_probs.items():
                    token_prob = prob_info.logprob
                    decoded_token = prob_info.decoded_token
                    if decoded_token in specific_words:
                        word_probability[decoded_token] = math.exp(token_prob)
                    
                if 0 < len(word_probability) <= len(specific_words):
                    for word in specific_words:
                        if word not in word_probability:
                            word_probability[word] = 0
                    break
                
            if len(word_probability) == 0:
                word_probability = {word: 0 for word in specific_words}
            
            word_probabilities.append(word_probability)
    
    return res_completions, word_probabilities


def load_vllm_llm(model, tensor_parallel_size, revision=None):
    llm = LLM(model=model, revision=revision, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=0.9, trust_remote_code=True)
    return llm