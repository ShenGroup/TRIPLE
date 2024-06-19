'''
Create baseline methods for comparison
'''
import promptsource
import datasets
import os

# os.path.join(os.path.dirname(__file__), "..")

from ..constants import *
from ..utils.prompt_preprocess import *


def Null_Instruction(examples, test_examples):
    
    
    instruction = ""
    prompt = fit_in_prompt(PROMPT_TEMPLATE, instruction, examples, test_examples)
    
    return prompt


def Prompt_Instruction(instruction, examples, test_examples):
    
    prompt = fit_in_prompt(PROMPT_TEMPLATE, instruction, examples, test_examples)
    
    return prompt
    
    