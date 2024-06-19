import torch
import json
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from promptsource.templates import DatasetTemplates
import datasets
from torch.utils.data import Dataset

induce_data_path = os.path.join(os.path.dirname(__file__), '../../data/instruction_induction/raw/induce/')
eval_data_path = os.path.join(os.path.dirname(__file__), '../../data/instruction_induction/raw/execute/')
bigbench_data_path = os.path.join(os.path.dirname(__file__), '../../data/bigbench-ii/')

class InductionDataset(Dataset):
    def __init__(self, task):
        self.task = task
        train_file = os.path.join(induce_data_path, f"{task}.json")
        test_file = os.path.join(eval_data_path, f"{task}.json")
        with open(train_file, 'r') as f:
            self.train_examples = json.load(f)['examples']
            # random.shuffle(self.train_examples)
        with open(test_file, 'r') as f:
            self.test_examples = json.load(f)['examples']
            # random.shuffle(self.test_examples)

    def __len__(self):
        return len(self.train_examples), len(self.test_examples)
    
    def get_len(self):
        return len(self.train_examples), len(self.test_examples)


    def __getitem__(self, idx, split='train'):
        if split == 'train':
            examples = self.train_examples
        elif split == 'test':
            examples = self.test_examples
        else:
            raise ValueError("Invalid split! Split must be 'train' or 'test'.")
        
        if idx < len(examples):
            example = examples[str(idx+1)]
        else:
            raise IndexError("Index out of range. The dataset contains {} examples, but you tried to access index {}.".format(len(examples), idx))
        
        if self.task == 'cause_and_effect':
            cause, effect = example['cause'], example['effect']
            # Pick an order randomly
            if random.random() < 0.5:
                input_ = f'Sentence 1: {cause} Sentence 2: {effect}'
            else:
                input_ = f'Sentence 1: {effect} Sentence 2: {cause}'
            output_ = cause
            return input_, output_
        elif self.task == 'common_concept':
            items = example['items']
            # Make comma separated list of items
            input_ = ', '.join(items[:-1])
            output_ = example['all_common_concepts']
            return input_, output_ 
        elif self.task == "rhymes":
            return example['input'], example['other_rhymes']
        elif "translation"  in self.task:
            return example['input'], example['possible_translations']
        elif self.task == "word_in_context":
            return example['input'], example['possible_outputs']
        
        return example['input'], example['output']
    
    def get_split(self, split='train'):
        if split == 'train':
            return self.train_examples
        elif split == 'test':
            return self.test_examples
        else:
            raise ValueError("Invalid split! Split must be 'train' or 'test'.")
        
    def sample_batch(self, batch_size, split='train'):
        if split == 'train':
            examples = self.train_examples
        elif split == 'test':
            examples = self.test_examples
        else:
            raise ValueError("Invalid split! Split must be 'train' or 'test'.")

        indices = np.random.choice(len(examples), batch_size, replace=False)
        batch = [self.__getitem__(idx, split) for idx in indices]
        return batch
    
    def sample_examples(self,num_examples):
        
        # sample from train num_examples
        examples = self.train_examples
        indices = np.random.choice(len(examples), num_examples, replace=False)
        batch = [self.__getitem__(idx, 'train') for idx in indices]
        return batch
    

class BigbenchDataset(Dataset):
    
    def __init__(self, task, eval_size: float = 0.2):
        self.task = task
        data_file = os.path.join(bigbench_data_path, f"{task}/prompt.csv")
        print(bigbench_data_path, data_file)
        dataset_df = pd.read_csv(data_file)
        # split into train and test
        inputs = dataset_df['input'].to_list()
        if "target_scores" in dataset_df.columns:
            choices = dataset_df['target_scores'].to_list()
            inputs = [f"{inp} \n  Choices: {choice}" for inp, choice in zip(inputs, choices)]
        labels = dataset_df['labels'].astype(str).to_list()
        self.train_examples, self.test_examples, self.train_labels, self.test_labels = train_test_split(inputs, labels, test_size=eval_size, random_state=42)
        


    def __len__(self):
        return len(self.train_examples), len(self.test_examples)
    
    def get_len(self):
        return len(self.train_examples), len(self.test_examples)


    def __getitem__(self, idx, split='train'):
        if split == 'train':
            assert idx < len(self.train_examples)
            input_data = self.train_examples[idx]
            label = self.train_labels[idx]
        elif split == 'test':
            assert idx < len(self.test_examples)
            input_data = self.train_examples[idx]
            label = self.train_labels[idx]
        else:
            raise ValueError("Invalid split! Split must be 'train' or 'test'.")
        

        return input_data, label
    
    def sample_batch(self, batch_size, split='train'):
        if split == 'train':
            examples = self.train_examples
        elif split == 'test':
            examples = self.test_examples
        else:
            raise ValueError("Invalid split! Split must be 'train' or 'test'.")

        indices = np.random.choice(len(examples), batch_size, replace=False)
        batch = [self.__getitem__(idx, split) for idx in indices]
        return batch
    
    def sample_examples(self,num_examples):
        
        # sample from train num_examples
        examples = self.train_examples
        indices = np.random.choice(len(examples), num_examples, replace=False)
        batch = [self.__getitem__(idx, 'train') for idx in indices]
        return batch

    
    
    
class AdvancedDataset(Dataset):
    def __init__(self, task):
        '''
        Initialize datasets that is availiable in Huggingface datasets library
        '''
        
        self.train_examples = datasets.load_dataset(task, split='train')
        self.test_examples = datasets.load_dataset(task, split='test')
        self.example_prompts = DatasetTemplates(task)
        self.prompt_names = self.example_prompts.all_template_names
        labels = self.example_prompts[self.prompt_names[0]].get_fixed_answer_choices_list()
        label_mapping = dict()
        for i in range(len(labels)):
            label_mapping[i] = labels[i]
        self.train_examples = [{'text': entry['text'], 'label': label_mapping[entry['label']]} for entry in self.train_examples]
        self.test_examples = [{'text': entry['text'], 'label': label_mapping[entry['label']]} for entry in self.test_examples]

    def __len__(self):
        return len(self.train_examples), len(self.test_examples)

    def __getitem__(self, idx, split='train'):
        if split == 'train':
            examples = self.train_examples
        elif split == 'test':
            examples = self.test_examples
        else:
            raise ValueError("Invalid split! Split must be 'train' or 'test'.")
        
        if idx < len(examples):
            example = examples[idx]
        else:
            raise IndexError("Index out of range. The dataset contains {} examples, but you tried to access index {}.".format(len(examples), idx))
        return example['text'], example['label']
    
    def get_split(self, split='train'):
        if split == 'train':
            return self.train_examples
        elif split == 'test':
            return self.test_examples
        else:
            raise ValueError("Invalid split! Split must be 'train' or 'test'.")
        
    def sample_batch(self, batch_size, split='train'):
        if split == 'train':
            examples = self.train_examples
        elif split == 'test':
            examples = self.test_examples
        else:
            raise ValueError("Invalid split! Split must be 'train' or 'test'.")

        indices = np.random.choice(len(examples), batch_size, replace=False)
        batch = [self.__getitem__(idx, split) for idx in indices]
        return batch
    
    def sample_examples(self,num_examples):
        
        # sample from train num_examples
        examples = self.train_examples
        indices = np.random.choice(len(examples), num_examples, replace=False)
        batch = [self.__getitem__(idx, 'train') for idx in indices]
        return batch
        
    
        
        

if __name__ == "__main__":
    data = InductionDataset("antonyms")
    for d in data:
        print(d)
