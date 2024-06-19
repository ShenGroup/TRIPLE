
from ..constants import *


def dataset_example_reformat(examples, labels):
    
    results = []
    
    for example in examples:
        input = example['text']
        if example['label'] >= 0:
            output = labels[example['label']]
        else:
            output = ""
        results.append(fit_in_examples(EXAMPLE_TEMPLATE, input, output))
        
        
    return results




def fit_in_examples(template, input, output):
    '''
    Given a template, input, and output, fill in the template with the input and output.
    '''
    
    if isinstance(output, list):
        output = output[0]
    
    return template.replace('[Input]', input).replace('[Output]', output)





def fit_in_prompt(template, instruction, examples, test_example):
    '''
    Given a template, input, and output, fill in the template with the input and output.
    '''
    return template.replace('[Instruction]', instruction).replace('[Examples]', examples).replace('[Test_Examples]', test_example)