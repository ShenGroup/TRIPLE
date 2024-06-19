# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import bitsandbytes, accelerate
import os
from typing import Optional, List, Dict
from accelerate import Accelerator
from transformers import (pipeline, AutoTokenizer, AutoModelForCausalLM, 
                          BitsAndBytesConfig, AutoConfig, TrainingArguments)
from src.LLM.basic_LLM import BasicLLM
import torch
import asyncio
import re

hf_checkpoints_path = "your_path_to_hf_checkpoints"

prompt_temp = "<s>[INST] <<SYS>> \n  [system_prompt] \n <</SYS>>  [prompt] \n [/INST]"


class WhiteBox(BasicLLM):
    def __init__(
        self,
        model: str = "Llama-2-7B",
        # tokenizer: str = "Llama-2-7B",
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 50,
        max_batch_size: int = 8,
        quantize: bool = True,
        # max_gen_len: Optional[int] = None,
    ) -> None:
        
        
        accelerator = Accelerator()

        
        if quantize:
            quant_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            )
        else:
            quant_config = None
            
        
        
        # Check with gpu has enough memory
        # if device is None:
        #     raise ValueError("No GPU has enough memory")
        #     device = "cpu"
        # else:
        #     device = "cuda:" + str(device)
            
        
        
        if os.path.exists(hf_checkpoints_path + model):
            model_path = hf_checkpoints_path + model
            print(f"Model {model} found in {model_path}, loading model...")
        else:
            model_path = model
            print(f"Model {model} not found in {hf_checkpoints_path}, loading model from {model_path}...")
            
        # if model not in supported_llms:
        #     raise ValueError(f"Model {model} not supported. Supported models are: {supported_llms}")
        # else:
            
        
        # Check if the model is downloaded
        # if not os.path.exists(model_path):
        #     model_path = model
        
        model = AutoModelForCausalLM.from_pretrained(model_path,device_map = "auto", quantization_config=quant_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = accelerator.prepare(model)
        # self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len

    def get_gpu_with_enough_memory(self, memory_threshold: int):
        memory_infos = torch.cuda.mem_get_info()
        for res in memory_infos:
            if res > memory_threshold:
                return memory_infos.index(res)
        return None


    def get_response(self, prompt: str, n: int = 1, temperature = 1) -> str:
        instruction = prompt_temp.replace("[system_prompt]", "You are an intelligent assistant. Please answer finish the given task, answer with the output only and reply nothing else.")
        instruction = instruction.replace("[prompt]", prompt)
        # print(instruction)
        responses = []
        
        # generate n responses from the model
        embedding_id = self.tokenizer.encode(instruction, return_tensors="pt").to(self.model.device)
        chat_history_ids = self.model.generate(embedding_id, max_new_tokens=self.max_seq_len, pad_token_id=self.tokenizer.eos_token_id,
                                               temperature=temperature, top_p=self.top_p, num_beams = n, num_return_sequences=n ,do_sample=True)
        # breakpoint()
        if n > 1:
            responses = self.tokenizer.batch_decode(chat_history_ids[:, embedding_id.shape[-1]:], skip_special_tokens=True)
        else:
            response = self.tokenizer.decode(chat_history_ids[:, embedding_id.shape[-1]:][0], skip_special_tokens=True)
            responses.append(response.split(":")[-1])

        raw_responses = responses.copy()
        responses = []
        # breakpoint()
        for response in raw_responses:
            cleaned_response = re.sub(r'[^A-Za-z0-9 ]+', '', response)
            responses.append(cleaned_response)


        # breakpoint()
        
        return responses

    def reset(self):
        pass