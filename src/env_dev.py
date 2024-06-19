import numpy as np
import openai
import datasets
import asyncio
import os
import tiktoken
import pandas as pd
import random
import pdb



from typing import List
from openai import OpenAI
from sklearn.cluster import KMeans
from src.constants import *
from src.bandit import (
    UCB, 
    SequentialHalving,
    SuccessiveRejects,
    TopK,
    ContinuousRejects,
    GSE
)
from src.utils.dataloader import InductionDataset, BigbenchDataset
from src.LLM import ChatGPT, WhiteBox
from src.utils.evaluate import (
    zero_one, 
    ChatGPT_eval,
    f1_score,
    multi_ans_f1_score,
    multi_ans_zero_one
)
from src.utils.prompt_preprocess import fit_in_prompt, fit_in_examples
from src.utils.utils import set_random_seeds, get_embedding, parse_tagged_text
from tqdm import tqdm
EXAMPLE_TEMPLATE = "Input: [Input] \nOutput: [Output] \n\n"

openai.api_key = DEFAULT_OPENAI.API_KEY

class Env:

    def __init__(
            self,
            bandit_choice: str = 'UCB',
            T: int = 0,
            few_shot: bool = False,
            LLM_choice: str = 'ChatGPT',
            reward_method_train: str = 'ChatGPT',
            reward_method_eval: str = '0/1',
            use_examples: bool = True,
            num_examples: int = 3,
            num_prompts_examples: int = 5, 
            # num_examples_per_prompt: int = 5,
            prompt_for_using_examples: str = DEFAULT_PROMPT.GEN_USING_EXAMPLES,
            examples: list = list(), 
            use_rephrases: bool = False, 
            num_prompts_rephrases: int = 5, 
            prompt_for_using_rephrases: str = DEFAULT_PROMPT.GEN_USING_REPHRASES,
            prompt_to_be_rephrased: str = None,
            task: str = "antonyms",
            random_seed: int = 0,
            load_from_file: bool = False,
            file_path: str = None,
            ) -> None:

        
        # assert bandit_choice in ["Cluster", "Gradient", "FuncApprox", None], "Input a valid advanced method"
        assert bandit_choice in ["UCB", "SequentialHalving", "SuccessiveRejects", "TopK", "ContinuousRejects", "GSE", "Cluster", "Gradient", "FuncApprox"], "Input a valid advanced method"
        # Basic initialization
        self.LLM_choice = LLM_choice
        self.LLM = self.init_LLM()
        self.T = T
        self.few_shot = few_shot
        self.use_examples = use_examples
        self.num_prompts_examples = num_prompts_examples
        self.prompt_for_using_examples = prompt_for_using_examples
        self.bandit_choice = bandit_choice

        if task in BASIC_TASKS + INDUCTION_TASKS:
            self.dataset = InductionDataset(task)
        elif task in BIGBENCH_TASKS:
            self.dataset = BigbenchDataset(task)
        else:
            raise ValueError("Invalid task! Task must be one of the following: {}".format(INDUCTION_TASKS + BIGBENCH_TASKS))
        
        if use_examples and len(examples) == 0:
            self.examples = self.dataset.sample_examples(num_examples=num_examples)
        else:
            self.examples = examples
        self.use_rephrases = use_rephrases
        self.num_prompts_rephrases = num_prompts_rephrases
        self.prompt_for_using_rephrases = prompt_for_using_rephrases
        self.prompt_to_be_rephrased = prompt_to_be_rephrased
        self.reward_train = self.init_reward(reward_method_train)
        self.reward_eval = self.init_reward(reward_method_eval)
        
        # if load_from_file, then directly load all the baseline prompts,
        # Otherwise, initialize the baseline prompts, the perform the embedding and clustering steps
        if load_from_file:
            # breakpoint()
            assert os.path.isfile(file_path), "Input a valid file path"
            print(f"Loading prompts from {file_path}...")
            self.prompts_df = pd.read_csv(file_path)
            self.candidate_prompts = self.prompts_df["prompts"].to_list()
            print(f"Loaded {len(self.candidate_prompts)} prompts from {file_path}.")
            self.num_prompts = len(self.candidate_prompts)
            self.num_clusters = len(self.prompts_df["clusters"].unique())
        else:
            self.num_prompts = 0
            self.num_prompts += num_prompts_examples if use_examples else 0
            self.num_prompts += num_prompts_rephrases if use_rephrases else 0
            self.num_clusters = np.round(np.sqrt(self.num_prompts))
       
        self.t = 0
        
        set_random_seeds(random_seed)
        print(f"Random seed set to {random_seed}")
       
        
        # Will only be used for 
        self.wrong_exmaples = dict()
        
    async def async_init(self):
        
        
        if self.LLM_choice == "ChatGPT":
            self.candidate_prompts = await self.generate_candidate_prompt_async()
        else:
            self.candidate_prompts = self.generate_candidate_prompt()
        self.prompts_df = pd.DataFrame({"prompts": self.candidate_prompts})
        self.__generate_embeddings()
        if self.bandit_choice == "Cluster":
            # Cluster enforce equal size/ budget proportional to the number of prompts
            self.__cluster_prompts()
    
        if self.bandit_choice not in ["Cluster", "Gradient", "FuncApprox"]:
            self.prompt_bandit = self.init_bandit(
                bandit_choice=self.bandit_choice,
                num_arms=self.num_prompts,
                budget=self.T
            )
            
            self.phase1_budget = 0
            self.current_phase = 2
        elif self.bandit_choice == "Cluster":
            self.phase1_budget = self.__clustering_buget()
            self.cluster_bandit = self.init_bandit(
                bandit_choice="SequentialHalving",
                num_arms=self.num_clusters,
                budget=self.phase1_budget * 2
            )
            self.prompt_bandit = self.init_bandit(
                bandit_choice="SequentialHalving",
                num_arms=self.num_prompts,
                budget=self.T
            )
            self.phase1_sample_mean = np.zeros(self.num_prompts)
            self.phase1_num_pulls = np.zeros(self.num_prompts) 
            self.current_phase = 1
            self.prompt_bandit_choice = "SequentialHalving"
        elif self.bandit_choice == "Gradient":
            self.prompt_bandit = self.init_bandit(
                bandit_choice="ContinuousRejects",
                num_arms=self.num_prompts,
                budget=self.T
            )
            self.prompt_bandit.target_arms = 2
            self.phase1_budget = 0
            self.current_phase = 2
        


    ########## Initialization functions ##########
    def init_LLM(self):
        if self.LLM_choice == 'ChatGPT':
            return ChatGPT(model= DEFAULT_OPENAI.RESPONSE_MODEL)
        elif self.LLM_choice == 'WhiteBox':
            return WhiteBox(model= DEFAULT_LLAMA.CKPT_DIR, tokenizer= DEFAULT_LLAMA.TOKENIZER_PATH)
        else:
            raise Exception("Cannot find requested LLM model.")

    def init_bandit(self,
                    bandit_choice: str = 'UCB',
                    num_arms: int = 0,
                    budget: int = 0,
                    ) -> object:
        if bandit_choice == 'UCB':
            return UCB(num_arms = num_arms)
        elif bandit_choice == "SequentialHalving":
            assert self.T > 0, "Input a positive horizon T"
            return SequentialHalving(num_arms= num_arms, T = budget)
        elif bandit_choice == "SuccessiveRejects":
            assert self.T > 0, "Input a positive horizon T"
            return SuccessiveRejects(num_arms= num_arms, T = budget)
        elif bandit_choice == "TopK":
            return TopK(num_arms= num_arms, T=budget)
        elif bandit_choice == "ContinuousRejects":
            return ContinuousRejects(num_arms= num_arms, T=budget)
        elif bandit_choice == "GSE":
            return GSE(num_arms= num_arms, T=budget, contexts=self.prompts_df["embedding"].apply(eval).to_list())
        else:
            raise Exception("Cannot find requested bandit algorithm.")
    
    def init_reward(self, method: str):
        if method == '0/1':
            return zero_one
        elif method == 'ChatGPT':
            return ChatGPT_eval
        elif method == "f1":
            return f1_score
        elif method == "multi_ans_f1":
            return multi_ans_f1_score
        elif method == "multi_ans_0/1":
            return multi_ans_zero_one
        else:
            raise ValueError(f'reward method {method} not found, available ones: 0/1, ChatGPT')
        
    def __generate_embeddings(self) -> None:
        
        assert self.prompts_df is not None, "Input a valid set of prompts"
        embedding_client = OpenAI()
        encoding = tiktoken.get_encoding(DEFAULT_EMBEDDING.ENCODING)
        self.prompts_df["n_tokens"] = self.prompts_df["prompts"].apply(lambda x: len(encoding.encode(x)))
        self.prompts_df["embedding"] = self.prompts_df["prompts"].apply(lambda x: get_embedding(x,embedding_client, model=DEFAULT_EMBEDDING.MODEL))
        if self.LLM_choice =="ChatGPT":
            # self.prompts_df["embedding"] = self.prompts_df["embedding"].apply(lambda x: x[0])
            self.reconnect_LLM()
        
    def __cluster_prompts(self) -> None:
        kmean = KMeans(n_clusters=self.num_clusters, random_state=0)
        clusters = kmean.fit_predict(self.prompts_df["embedding"].to_list())
        self.prompts_df["clusters"] = clusters
        
    def __clustering_buget(self) -> int:
        max_budget = int(self.num_clusters * 5)
        cluster_distribution = self.prompts_df["clusters"].value_counts(normalize=True)
        uniform_distribution = np.ones(self.num_clusters) / self.num_clusters
        kl_divergence = np.sum(cluster_distribution * np.log(cluster_distribution / uniform_distribution))
        phase1_budget = int(max_budget * (1 - kl_divergence))
        print("Phase 1 budget: ", phase1_budget)
        return int(max_budget * (1 - kl_divergence))
    
    
    ####### Syncronized functions for WhiteBox #######
    
    def __generate_prompts_from_examples(self, 
                                         examples: List[str] = [],
                                         n: int = 10) -> list:
        candidate_prompts = []
        assert n >0, "Input a positive number for how many prompts are to be generated from examples"
        assert len(examples) > 0, "Input a valid set of examples"
        assert '[Examples]' in self.prompt_for_using_examples, "Input a prompt with [Examples] to indicate where to fill in examples"
        if isinstance(self.examples[0][1], list):
            examples_to_use = [EXAMPLE_TEMPLATE.replace("[Input]", input).replace("[Output]", output[0]) for input, output in examples]
        else:
            examples_to_use = [EXAMPLE_TEMPLATE.replace("[Input]", input).replace("[Output]", output) for input, output in examples]
        examples_to_use = "\n".join(examples_to_use)
        prompt_to_use = self.prompt_for_using_examples.replace("[Examples]", f"{examples_to_use}")
        candidate_prompts = self.LLM.get_response(prompt_to_use, n = self.num_prompts_examples, temperature=1.5)
        
        return candidate_prompts
    
    def __rephrase_prompt(self, 
                          example_prompt: str = "Solve the following task:",
                          n: int = 3) -> list:
        
        assert n >0, "Input a positive number for how many prompts are to be generated from rephrasing"
        assert example_prompt is not None, "Input the prompt to be rephrased."
        assert "[Prompt]" in self.prompt_for_using_rephrases, "Input a prompt with [Prompt] to indicate where to fill in the original prompt"
        prompt_to_use = self.prompt_for_using_rephrases.replace("[Prompt]", self.prompt_to_be_rephrased)
        candidate_promts = self.LLM.get_response(prompt_to_use, n = self.num_prompts_rephrases)
        
        return candidate_promts
    
    def generate_candidate_prompt(self) -> list:
        candidate_prompts = []
        if self.use_examples:
           candidate_prompts += self.__generate_prompts_from_examples(self.examples, self.num_prompts_examples)
        if self.use_rephrases:
           candidate_prompts += self.__rephrase_prompt(self.prompt_to_be_rephrased, self.num_prompts_rephrases)
          
        candidate_prompts = list(set(candidate_prompts))
        self.num_prompts = len(candidate_prompts)
        print(candidate_prompts)

        return candidate_prompts
    
    def step(self):
        # - choose a query (utils)
        # - choose a prompt (bandit)
        # - get response (LLM)
        # - evaluate (utils)
        # - update bandits (bandit)
        
        if self.bandit_choice == "Cluster":
            # Filter out clusters first
            query, target = self.dataset.sample_batch(batch_size = 1, split = "train")[0]
            if self.current_phase == 1:
                arm = self.cluster_bandit.choose_action()
                prompt_id = random.choice(self.prompts_df[self.prompts_df["clusters"] == arm].index)
                instruction = self.candidate_prompts[prompt_id]
            else:
                arm = self.prompt_bandit.choose_action()
                prompt_id = self.active_prompts[arm]
                instruction = self.candidate_prompts[prompt_id]
        else:
            query, target = self.dataset.sample_batch(batch_size = 1, split = "train")[0]
            arm = self.prompt_bandit.choose_action()
            instruction = self.candidate_prompts[arm]
        if self.use_examples:
            examples = [fit_in_examples(EXAMPLE_TEMPLATE, input, output) for input, output in self.examples]
        else:
            examples = []
        task_example = fit_in_examples(EXAMPLE_TEMPLATE, query, "{output}")
        if self.few_shot:
            prompt = fit_in_prompt(PROMPT_TEMPLATE, instruction, "".join(examples), task_example)
        else:
            prompt = fit_in_prompt(PROMPT_TEMPLATE, instruction, "", task_example)
            # Old version of the response
        response = self.LLM.get_response(prompt = "Provide only one answer and NOTHING else.\n" + prompt, n=1)[0] 

            # Filter the unwanted responses and only keep anything after Output:
        # response = response.split(":")[-1]  
        reward = self.reward_train(response, target)
        self.t += 1
        if self.t == self.phase1_budget and self.bandit_choice == "Cluster":
            self.current_phase = 2
            active_clusters = self.cluster_bandit.active_arms
            self.active_prompts = self.prompts_df[self.prompts_df["clusters"].isin(active_clusters)].index.to_numpy()
            active_mean = self.phase1_sample_mean[self.active_prompts]
            num_pulls = self.phase1_num_pulls[self.active_prompts]
            # for i,prompt_id in enumerate(self.active_prompts):
            #     cluster_id = self.prompts_df.loc[prompt_id, "clusters"]
            #     active_mean[i] = self.cluster_bandit.sample_mean[cluster_id]
            # remove rejected clusters
            self.prompt_bandit = self.init_bandit(
                bandit_choice=self.prompt_bandit_choice,
                num_arms = self.active_prompts.shape[0],
                budget=self.T - self.phase1_budget
            )
            self.prompt_bandit.sample_mean = active_mean
            print(f"Phase 1 ended, remaining budget: {self.T - self.phase1_budget}, remaining prompts: {self.active_prompts} \n")
        if self.bandit_choice == "Cluster" and self.current_phase == 1:
            self.cluster_bandit.update(arm, reward)
            self.phase1_sample_mean[arm] = (self.phase1_sample_mean[arm] * self.phase1_num_pulls[arm] + reward) / (self.phase1_num_pulls[arm] + 1)
            self.phase1_num_pulls[arm] += 1
        else:
            self.prompt_bandit.update(arm, reward)

        return arm, query, target, response, reward
    
    
    def evaluation(self, num_eval_samples: int = 100):
        # Evaluate the bandit algorithm by running through testing dataset
        # num_correct = 0
        num_available_samples = self.dataset.__len__()[1]
        # assert num_eval_samples <= num_available_samples, "Not enough samples existing in the evaluation set."
        num_eval_samples = min(num_eval_samples, num_available_samples)
        eval_set = self.dataset.sample_batch(batch_size=num_eval_samples, split="test")
        # chosen_arm = self.bandit.choose_action()
        bandit_best_arm, bandit_best_prompt = self.best_prompt()
        arm_eva_rewards = np.zeros(self.num_prompts)
        for j in range(self.num_prompts):
            progress_bar = tqdm(range(num_eval_samples))
            prompt = self.candidate_prompts[j]
            print(f"Evaluating arm: {j, prompt}")
            for i in progress_bar:
                query, target = eval_set[i]
                # Old version of the response
                response = self.LLM.get_response(prompt = "Provide only one answer and NOTHING else." + prompt + " " + query, n=1)
                response = response[0]
                reward = self.reward_eval(response, target)
                arm_eva_rewards[j] += reward
                progress_bar.set_description(f"Evaluating arm: {j}, Evaluation reward: {arm_eva_rewards[j]/(i+1):.2f}")
        
        true_best_arm = np.argmax(arm_eva_rewards/num_eval_samples)
                
            # num_correct += reward
            # self.bandit.update(arm, reward)
            
        # print(f"bandit best arm: {bandit_best_arm}, evaluation best arm {true_best_arm}")
        print(f"bandit best prompt: {bandit_best_prompt}, evaluation best prompt: {self.candidate_prompts[true_best_arm]}")
        print(f"Training reward: {arm_eva_rewards[bandit_best_arm]/num_eval_samples}, evaluation reward: {arm_eva_rewards[true_best_arm]/num_eval_samples}")
        
        return arm_eva_rewards/num_eval_samples
    
    
    
    
    ###### Asyncronized functions for ChatGPT ######
    
    async def __generate_prompts_from_examples_async(self, 
                                         examples: List[str] = [],
                                         n: int = 10) -> list:
        candidate_prompts = []
        assert n >0, "Input a positive number for how many prompts are to be generated from examples"
        assert len(examples) > 0, "Input a valid set of examples"
        assert '[Examples]' in self.prompt_for_using_examples, "Input a prompt with [Examples] to indicate where to fill in examples"
        if isinstance(self.examples[0][1], list):
            examples_to_use = [EXAMPLE_TEMPLATE.replace("[Input]", input).replace("[Output]", output[0]) for input, output in examples]
        else:
            examples_to_use = [EXAMPLE_TEMPLATE.replace("[Input]", input).replace("[Output]", output) for input, output in examples]
        examples_to_use = "\n".join(examples_to_use)
        prompt_to_use = self.prompt_for_using_examples.replace("[Examples]", f"{examples_to_use}")
        candidate_prompts = await self.LLM.get_response(prompt_to_use, n = self.num_prompts_examples, temperature=1.5)
        
        return candidate_prompts
    
    async def __rephrase_prompt_async(self, 
                          example_prompt: str = "Solve the following task:",
                          n: int = 3) -> list:
        
        assert n >0, "Input a positive number for how many prompts are to be generated from rephrasing"
        assert example_prompt is not None, "Input the prompt to be rephrased."
        assert "[Prompt]" in self.prompt_for_using_rephrases, "Input a prompt with [Prompt] to indicate where to fill in the original prompt"
        prompt_to_use = self.prompt_for_using_rephrases.replace("[Prompt]", self.prompt_to_be_rephrased)
        candidate_promts = await self.LLM.get_response(prompt_to_use, n = self.num_prompts_rephrases)
        
        return candidate_promts
    
    async def generate_candidate_prompt_async(self) -> list:
        candidate_prompts = []
        if self.use_examples:
           candidate_prompts += await self.__generate_prompts_from_examples_async(self.examples, self.num_prompts_examples)
        if self.use_rephrases:
           candidate_prompts += await self.__rephrase_prompt_async(self.prompt_to_be_rephrased, self.num_prompts_rephrases)
          
        candidate_prompts = list(set(candidate_prompts))
        self.num_prompts = len(candidate_prompts)
        print(candidate_prompts)

        return candidate_prompts

    async def async_step(self):
        # - choose a query (utils)
        # - choose a prompt (bandit)
        # - get response (LLM)
        # - evaluate (utils)
        # - update bandits (bandit)
        
        if self.bandit_choice == "Cluster":
              # Filter out clusters first
            query, target = self.dataset.sample_batch(batch_size = 1, split = "train")[0]
            if self.current_phase == 1:
                arm = self.cluster_bandit.choose_action()
                prompt_id = random.choice(self.prompts_df[self.prompts_df["clusters"] == arm].index)
                instruction = self.candidate_prompts[prompt_id]
            else:
                arm = self.prompt_bandit.choose_action()
                prompt_id = self.active_prompts[arm]
                instruction = self.candidate_prompts[prompt_id]
        else:
            query, target = self.dataset.sample_batch(batch_size = 1, split = "train")[0]
            arm = self.prompt_bandit.choose_action()
            # breakpoint()
            instruction = self.candidate_prompts[arm]
        if self.use_examples:
            examples = [fit_in_examples(EXAMPLE_TEMPLATE, input, output) for input, output in self.examples]
        else:
            examples = []
        task_example = fit_in_examples(EXAMPLE_TEMPLATE, query, "[Output]")
        if self.few_shot:
            prompt = fit_in_prompt(PROMPT_TEMPLATE, instruction, "".join(examples), task_example)
        else:
            prompt = fit_in_prompt(PROMPT_TEMPLATE, instruction, "", task_example)
            # Old version of the response
        response = await self.LLM.get_response(prompt = "Provide only one answer and NOTHING else." + prompt, n=1)
        response = response[0]
            

            # breakpoint()
            # response = self.LLM.get_response(prompt = prompt, n=1)[0]
        reward = self.reward_train(response, target)
        self.t += 1
        # If it is gradient, then we need to collect wrong examples for each prompt
        if self.bandit_choice == "Gradient" and reward < 0.5:
            wrong_example = fit_in_examples(EXAMPLE_TEMPLATE, query, target)
            wrong_example += ("Your Respnse: " + response + "\n")
            if arm in self.wrong_exmaples:    
                self.wrong_exmaples[arm].append(wrong_example)
            else:
                self.wrong_exmaples[arm] = [wrong_example]
            
        if self.t == self.phase1_budget and self.bandit_choice == "Cluster":
            self.current_phase = 2
            active_clusters = self.cluster_bandit.active_arms
            self.active_prompts = self.prompts_df[self.prompts_df["clusters"].isin(active_clusters)].index.to_numpy()
            active_mean = np.zeros(self.active_prompts.shape[0])
            for i, prompt_id in enumerate(self.active_prompts):
                cluster_id = self.prompts_df.loc[prompt_id, "clusters"]
                active_mean[i] = self.cluster_bandit.sample_mean[cluster_id]
            # remove rejected clusters
            self.prompt_bandit = self.init_bandit(
                bandit_choice=self.prompt_bandit_choice,
                num_arms=len(self.active_prompts),
                budget=self.T - self.phase1_budget
            )
            self.prompt_bandit.sample_mean = active_mean
            print(f"Phase 1 ended, remaining budget: {self.T - self.phase1_budget}, remaining prompts: {self.active_prompts} \n")
        if self.bandit_choice == "Cluster" and self.current_phase == 1:
            self.cluster_bandit.update(arm, reward)
        else:
            self.prompt_bandit.update(arm, reward)

        return arm, query, target, response, reward
    
    async def evaluation_async(self, num_eval_samples: int = 100):
        # Evaluate the bandit algorithm by running through testing dataset
        # num_correct = 0
        num_available_samples = self.dataset.__len__()[1]
        # assert num_eval_samples <= num_available_samples, "Not enough samples existing in the evaluation set."
        num_eval_samples = min(num_eval_samples, num_available_samples)
        eval_set = self.dataset.sample_batch(batch_size=num_eval_samples, split="test")
        # chosen_arm = self.bandit.choose_action()
        bandit_best_arm, bandit_best_prompt = self.best_prompt()
        arm_eva_rewards = np.zeros(self.num_prompts)
        for j in range(self.num_prompts):
            progress_bar = tqdm(range(num_eval_samples))
            prompt = self.candidate_prompts[j]
            print(f"Evaluating arm: {j, prompt}")
            for i in progress_bar:
                query, target = eval_set[i]
                # Old version of the response
                response = await self.LLM.get_response(prompt = "Provide only one answer and NOTHING else." + prompt + " " + query, n=1)
                response = response[0]
                reward = self.reward_eval(response, target)
                arm_eva_rewards[j] += reward
                progress_bar.set_description(f"Evaluating arm: {j}, Evaluation reward: {arm_eva_rewards[j]/(i+1):.2f}")
        
        true_best_arm = np.argmax(arm_eva_rewards/num_eval_samples)
                
            # num_correct += reward
            # self.bandit.update(arm, reward)
            
        # print(f"bandit best arm: {bandit_best_arm}, evaluation best arm {true_best_arm}")
        print(f"bandit best prompt: {bandit_best_prompt}, evaluation best prompt: {self.candidate_prompts[true_best_arm]}")
        print(f"Training reward: {arm_eva_rewards[bandit_best_arm]/num_eval_samples}, evaluation reward: {arm_eva_rewards[true_best_arm]/num_eval_samples}")
        
        return arm_eva_rewards/num_eval_samples
    
    async def gradient_update(self):
        
        # First "compute" gradient from the best arms
        # pdb.set_trace()
        candidate_prompts = []
        candidate_idxs = np.argsort(self.prompt_bandit.get_sample_mean())[::-1][:2]
        for idx in candidate_idxs:
            prompt = self.candidate_prompts[idx]
            if idx in self.wrong_exmaples.keys():
                wrong_examples = self.wrong_exmaples[idx]
                gradient_prompt = GRADIENT_TEMPLATE.replace("{prompt}", prompt).replace("{error_string}", "\n".join(wrong_examples)).replace("{num_feedbacks}", "2")
                gradient_response = await self.LLM.get_response(prompt=gradient_prompt, n = 1, temperature = 1)
                gradient_response = gradient_response[0]
                gradients = parse_tagged_text(gradient_response, "<START>", "<END>")
                print("For prompt: ", prompt, "we found gradients: ", gradients)
                new_prompt_instruction = GENERATION_TEMPLATE.replace("{prompt}", prompt).replace("{error_str}", "\n".join(wrong_examples)).replace("{gradient}", "\n".join(gradients)).replace("{num_prompts}", "5")
                # print(new_prompt_instruction)
                new_prompt_response = await self.LLM.get_response(prompt = new_prompt_instruction, n = 1, temperature=1)
                new_prompt_response = new_prompt_response[0]
                new_prompts = parse_tagged_text(new_prompt_response, "<START>", "<END>")
                # Filter the real prompts from the response
                candidate_prompts = candidate_prompts + new_prompts
                
        print("Generated new prompts: ", candidate_prompts)
        self.prompts_df = pd.DataFrame({"prompts": candidate_prompts})
        self.__generate_embeddings()
        self.T = 5 * len(candidate_prompts)
        self.candidate_prompts = candidate_prompts
        self.prompt_bandit = self.init_bandit(
                bandit_choice="ContinuousRejects",
                num_arms=len(candidate_prompts),
                budget=self.T
            )
        # self.__cluster_prompts()
        
        
                
        
    
    def best_prompt(self) -> str:
        best_arm = self.prompt_bandit.best_arm()
        if self.bandit_choice == "Cluster" and self.current_phase == 2:
            best_arm = self.active_prompts[best_arm]
        return best_arm, self.candidate_prompts[best_arm]

    def reset(self):
        self.candidate_prompts = self.generate_candidate_prompt()
        self.prompt_bandit.reset()
        self.LLM.reset()
        if self.bandit_choice == "Cluster":
            self.cluster_bandit.reset()
            self.current_phase = 1
        
    def get_bandit_sample_mean(self):
        
        if self.bandit_choice == "Cluster" and self.current_phase == 1:
            print("Warning: the bandit is still in the cluster phase, the sample mean is not accurate.")
            return self.cluster_bandit.get_sample_mean()
        
        return self.prompt_bandit.get_sample_mean()
    
    def reconnect_LLM(self):
        self.LLM = self.init_LLM()
    
    
    
        

    
