import numpy as np
import openai

from src.LLM.ChatGPT import ChatGPT
from src.constants import DEFAULT_PROMPT

import string
from typing import List
from collections import Counter


def normalize_prediction(prediction: str, 
                         lowercase: bool = True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(
        str.maketrans('', '', string.punctuation))

    return prediction

def zero_one(response: str, target: str) -> float:

    reward = 0

    response_tokens = normalize_prediction(response, lowercase=True).split()
    target_tokens = normalize_prediction(target, lowercase=True).split()

    return float(set(response_tokens) == set(target_tokens))

def contains(response: str, target: str) -> float:

    response_tokens = normalize_prediction(response, lowercase=True).split()
    target_tokens = normalize_prediction(target, lowercase=True).split()

    return float(set(response_tokens).issubset(set(target_tokens)))

def ChatGPT_eval(response: str, target: str) -> float:

    evaluatorLLM = ChatGPT()
    reward = 0
    
    prompt = DEFAULT_PROMPT.RATE_SIMILARITY.replace("[Target]", target).replace("[Response]", response)

    while True:
        try:
            reward = float(evaluatorLLM.get_response(prompt = prompt, n = 1)[0])
            if reward >= 0 and reward <=1:
                return reward
        except:
            print("Try LLM evaluation again...")
            

def f1_score(response: str, target: str)  -> float:
    prediction_tokens = normalize_prediction(
        response, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(
        target, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def multi_ans_f1_score(response: str, target: List[str]) -> float:
    f1 = max([f1_score(response, t) for t in target])
    return f1

def multi_ans_zero_one(response: str, target: List[str]) -> float:
    zero_one = max([zero_one(response, t) for t in target])
    return zero_one

def multi_ans_contains(response: str, target: List[str]) -> float:
    contains = max([contains(response, t) for t in target])
    return contains




