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

from sklearn.cluster import KMeans
from tqdm import tqdm
from openai import OpenAI
from src.utils.utils import set_random_seeds, get_embedding
from src.env import Env
from src.utils.dataloader import InductionDataset, BigbenchDataset
from src.constants import DEFAULT_OPENAI, INDUCTION_TASKS, BIGBENCH_TASKS
from sklearn.manifold import TSNE
from wandb import AlertLevel



def main(task: str = "informal_to_formal",
         num_prompts: int = 10,
         llm_choice: str = "ChatGPT",
         config_path: str = "./config.json",
         num_clusters: int = 3) -> None:
    # Load config from file
    with open(config_path, "r") as f:
        config = json.load(f)
        
        
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"
    max_tokens = 8000
        
    # Load examples from file
        

    print(f"Task: {task}")
    start_time = time.time()
        
    config["task"] = task
    config["bandit_choice"] = "UCB"
    config["use_examples"] = True
    config["use_rephrases"] = False
    
    # Get the number of samples in the training dataset
    if task in INDUCTION_TASKS:
        dataset = InductionDataset(task)
        task_set= "instruction_induction"
    elif task in BIGBENCH_TASKS:
        dataset = BigbenchDataset(task)
        task_set= "bigbench-ii"
    else:
        print("Invalid task!")
    train_sample_size, eval_sample_size = dataset.get_len()
    num_prompts_examples = num_prompts
    if task in BIGBENCH_TASKS:
        num_examples = min(int(train_sample_size*0.5), 10)
    else:
        num_examples = min(int(train_sample_size*0.5), 100)

    if llm_choice =="WhiteBox":
        num_examples = 5
    num_samples = 10 * num_prompts_examples# Replace with the appropriate method to get the number of samples
    
    config["T"] = num_samples  # Set T equal to the number of samples
    if task in ["rhymes", "word_in_context", "common_concept","object_counting"] or "translation" in task:
        config["reward_method_train"] = "multi_ans_f1"
        config["reward_method_eval"] = "multi_ans_f1"
    
    storage_path = f"./data/llama2-13b/embeddings/"
    markers = ['o', 's', '^', 'D', 'p', '*', 'h', 'v']
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    # Initialize wandb run with a custom name
    run = wandb.init(project="prompt_bandit_gen", config=config, name=f"task_{task}_baseline")
    env = Env(
            bandit_choice="UCB",
            T=150,
            advanced_method=None,
            reward_method_train=config["reward_method_train"],
            reward_method_eval=config["reward_method_eval"],
            LLM_choice = llm_choice,
            use_examples=True,
            num_examples=5,
            num_prompts_examples=num_prompts,
            task=task,
            load_from_file=False,
            random_seed=42,
            )
    asyncio.run(env.async_init())
    
    
    candidate_prompts = env.candidate_prompts
    prompts_df = pd.DataFrame(candidate_prompts, columns=["prompts"])
    embedding_client = OpenAI()
    encoding = tiktoken.get_encoding(embedding_encoding)
    prompts_df["n_tokens"] = prompts_df["prompts"].apply(lambda x: len(encoding.encode(x)))
    prompts_df["embedding"] = prompts_df["prompts"].apply(lambda x: get_embedding(x,embedding_client, model=embedding_model))
    embeddings = prompts_df["embedding"].to_list()
    
    kmean = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmean.fit_predict(embeddings)
    prompts_df["clusters"] = clusters
    
    matrix = prompts_df["embedding"].to_list()
    matrix = np.array(matrix)
        # matrix.shape
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)    
    
    x = [x for x,y in vis_dims]
    y = [y for x,y in vis_dims]
    
    if llm_choice == "ChatGPT":
        env.reconnect_LLM()
    
    if llm_choice == "ChatGPT":
        eva_reward = asyncio.run(env.evaluation_async(num_eval_samples=100))
    else:
        eva_reward = env.evaluation(num_eval_samples=100)
    prompt_scores = eva_reward
    prompts_df["scores"] = prompt_scores.tolist()
    score_std = np.std(prompt_scores)
    relative_std = score_std / np.mean(prompt_scores)
    fig = plt.figure(figsize=(10, 10))
    plt.bar(np.arange(len(prompt_scores)), prompt_scores, align='center', alpha=0.5)
    plt.xlabel('Prompts')
    plt.ylabel('Eval Scores')
    plt.title(f'Evaluation Scores for {task}')
    run.log({f"Eval reward on task {task}": wandb.Image(plt)})
    

    # Check whether the std is large or small
    if not os.path.exists(f"./your/storage/path"):
        os.makedirs(f"./your/storage/path")

    
    if relative_std < 0.3:
        # Too large or too small variances are not good for our algorithm
      
        print("Low variance prompts don't exist! Creating...")
        prompts_df.to_csv(f"./your/storage/path")
        plt.savefig(f"./your/storage/path")
        plt.close()
    else:
        print("High variance prompts don't exist! Creating...")
        prompts_df.to_csv(f"./your/storage/path")
        plt.savefig(f"./your/storage/path")
        plt.close()
        fig = plt.figure(figsize=(10, 10))
        for cluster in np.unique(clusters):
            idxs = np.array(clusters) == cluster
            plt.scatter(np.array(x)[idxs], np.array(y)[idxs], c=colors[cluster % len(colors)], s=np.array(prompt_scores)[idxs] * 500,
                alpha=0.5, marker=markers[cluster % len(markers)], label=f'Cluster {cluster}')

        # Highlight the best prompt
        # You can change the marker and color to make it stand out
        best_prompt_idx = np.argmax(prompt_scores)
        plt.scatter(x[best_prompt_idx], y[best_prompt_idx], c='red', s=prompt_scores[best_prompt_idx] * 500, 
                    alpha=1, marker=markers[clusters[best_prompt_idx] % len(markers)], edgecolors='black',)

        plt.colorbar(label='Clusters')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(title=f'Scatter Plot for {task}')
            
        run.log({f"Cluster plot on task {task} with importance": wandb.Image(plt)})
        plt.savefig(f"./your/storage/path")
        run.alert(title=f"Found a useful sample for task {task}!", text=f"The found relative variance is {relative_std}, the prompt scores are {prompt_scores}", level = AlertLevel.INFO)
        plt.close()

    
        # Convert prompt_scores to a list
    prompts_df["scores"] = prompt_scores.tolist()
    run.finish()
            # print(f"Task: {task}, Time: {time.time() - start_time:.2f}, Reward: {eva_reward:.2f}")


if __name__ == "__main__":
    # Parse command line arguments using fire
    fire.Fire(main)
