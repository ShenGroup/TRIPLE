import pandas as pd
import wandb
import json
import numpy as np
import re
import time
import fire
import os
import asyncio
import tiktoken
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm
from openai import OpenAI
from src.utils.utils import set_random_seeds, get_embedding
from src.env import Env
from src.utils.dataloader import InductionDataset as Dataset
from src.constants import *
from sklearn.manifold import TSNE



def main(task: str = "informal_to_formal",
         budget: int = 100,
         bandit_algorithm: str = "UCB",
         config_path: str = "./config.json",
         storage_path: str = "/your/path/to/embeddings/",
         random_seed: int = 42,
         llm: str = "WhiteBox",
         ) -> None:
    # Load config from file
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Load examples from file
    if os.path.isfile(f"{storage_path}{task}_prompts.csv"):
        prompts_df = pd.read_csv(f"{storage_path}{task}_prompts.csv")
        prompts = prompts_df["prompts"].to_list()
        prompt_scores = prompts_df["scores"].to_numpy()
        best_prompt = prompts[np.argmax(prompt_scores)]
        best_score = np.max(prompt_scores)
        num_prompts = len(prompts)
        total_budget = num_prompts * budget
    if task in ["rhymes", "word_in_context", "common_concept"] or "translation" in task:
                config["reward_method_train"] = "multi_ans_f1"
                config["reward_method_eval"] = "multi_ans_f1"
                
    if bandit_algorithm == "Cluster":
        config["prompt_bandit_choice"] = "SequentialHalving"
        config["cluster_bandit_choice"] = "SequentialHalving"
        config["with_cluster"] = True
    else:
        config["prompt_bandit_choice"] = bandit_algorithm
        config["cluster_bandit_choice"] = "SequentialHalving"
        config["with_cluster"] = False
    config["random_seed"] = random_seed
    config["LLM_choice"] = llm
        
    env = Env(
            bandit_choice=bandit_algorithm,
            T=total_budget,
            reward_method_train=config["reward_method_train"],
            reward_method_eval=config["reward_method_eval"],
            LLM_choice = llm,
            use_examples=False,
            num_examples=0,
            num_prompts_examples=0,
            task=task,
            load_from_file=True,
            file_path=f"{storage_path}{task}_prompts.csv",
            random_seed=random_seed,
            )
    asyncio.run(env.async_init())
    
    config["task"] = task
    config["T"] = total_budget
    dataset = storage_path.split("/")[-4]
    # Initialize wandb    
    run = wandb.init(project=f"{dataset}_banditonly", config=config, name=f"task_{task}_bandit_{bandit_algorithm}_budget_{budget}_seed_{random_seed}")
          
          
    # run.log({"best_prompt": best_prompt,"best_arm": np.argmax(prompt_scores), "best_score": best_score})
    progress_bar = tqdm(range(total_budget), desc=f"Bandit best arm: {env.best_prompt()[0]}, Training reward: 0.00")
    running_average_reward = 0  # Variable to store the running average efficiency

    for i in progress_bar:
        if config["LLM_choice"] == "ChatGPT":
            _, _, _, _, reward = asyncio.run(env.async_step())
        else:
            _, _, _, _, reward = env.step()
        running_average_reward = (running_average_reward * 0.8 + 0.2  * reward)
        progress_bar.set_description(f"Bandit best arm: {env.best_prompt()[0]}, Training reward: {running_average_reward:.2f}")
        # Log the running average efficiency
        run.log({"running_average_reward": running_average_reward})

        if (i + 1) % 10 == 0:
            run.log({"current_best_arm": env.best_prompt()[0]})
    
    
    
    bandit_best_arm = env.best_prompt()[0]
    bandit_best_prompt = prompts[bandit_best_arm]
    bandit_best_score = prompt_scores[bandit_best_arm]
    found_good = ((best_score - bandit_best_score) < 0.1 * best_score)
    # run.log({"bandit_best_prompt": bandit_best_prompt,"bandit_best_arm": bandit_best_arm, "bandit_best_score": bandit_best_score})
    # run.log({"difference": best_score - bandit_best_score, "found_best": best_prompt == bandit_best_prompt})
    table = wandb.Table(columns=["task","bandit_algorithm", "budget", "random_seed", "best_prompt", "best_arm", "best_score", "bandit_best_prompt", "bandit_best_arm", "bandit_best_score", "difference", "found_good"],
                        data=[[task, bandit_algorithm, budget, random_seed, best_prompt, np.argmax(prompt_scores), best_score, bandit_best_prompt, bandit_best_arm, bandit_best_score, best_score - bandit_best_score, found_good]])
    run.log({"Experiment_results": table})
    df = pd.DataFrame(columns=["task","bandit_algorithm", "budget", "random_seed", "best_prompt", "best_arm", "best_score", "bandit_best_prompt", "bandit_best_arm", "bandit_best_score", "difference", "found_good"],
                      data=[[task, bandit_algorithm, budget, random_seed, best_prompt, np.argmax(prompt_scores), best_score, bandit_best_prompt, bandit_best_arm, bandit_best_score, best_score - bandit_best_score, found_good]])
    if os.path.isfile(f"./your/storage/path"):
        # df.to_csv(f"./results/useful_tasks_bandit_results_budget_{budget}.csv", mode='a', header=False, index=False)
        df.to_csv(f"./your/storage/path", mode='a', header=False, index=False)
    else:
        df.to_csv(f"./your/storage/path", index=False)
    if (best_score-bandit_best_score)/best_score > 0.15:
        wandb.alert(title="Bandit found a bad prompt", text=f"Bandit {bandit_algorithm} found a bad prompt for task {task} under budget {budget} with a difference of {(best_score-bandit_best_score)/best_score}")
    run.finish()

if __name__ == "__main__":
    # Parse command line arguments using fire
    fire.Fire(main)
