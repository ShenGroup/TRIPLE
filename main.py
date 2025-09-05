import pandas as pd
import wandb
import json
import numpy as np
import re
import fire
import asyncio



from promptsource.templates import DatasetTemplates
from tqdm import tqdm
from src.env import Env
from src.utils.dataloader import InductionDataset as Dataset


def main(examples: list = list(), 
         prompt_example:  str = "Which animal is larger", 
         config_path: str = None) -> None:

    with open(config_path, "r") as f:
        config = json.load(f)
    run = wandb.init(project="prompt_bandit_test", config=config)
    if not prompt_example:
        prompt_examples = DatasetTemplates(config["task"])
        prompt_names = prompt_examples.all_template_names
        rand_idx = np.random.randint(len(prompt_names))
        template = prompt_examples[prompt_names[rand_idx]].jinja
        prompt_example = re.sub(r'{{.*?}}', '', template)
    env = Env(
        bandit_choice= config["bandit_choice"],
        T = config["T"],
        LLM_choice = config["LLM_choice"],
        reward_method_train = config["reward_method_train"],
        reward_method_eval = config["reward_method_eval"],
        use_examples = config["use_examples"],
        method= config["method"],
        num_examples= config["num_examples"],
        num_prompts_examples= config["num_prompts_examples"],
        num_examples_per_prompt= config["num_examples_per_prompt"],
        examples= examples,
        use_rephrases= config["use_rephrases"],
        num_prompts_rephrases= config["num_prompts_rephrases"],
        prompt_to_be_rephrased= prompt_example,
        task = config["task"]
    )


    progress_bar = tqdm(range(config["T"]))
    for i in progress_bar:
        _, _, _, _, reward = asyncio.run(env.step())
        eva_reward = 0
        log_info = {"training_reward": reward}

        run.log(log_info)
        progress_bar.set_description(f"Bandit best arm: {env.best_prompt()[0]}, Training reward: {reward:.2f}")

    eva_reward =  asyncio.run(env.evaluation(num_eval_samples = config["num_eval_samples"]))
    run.log({"evaluation_reward": eva_reward}) 
    
            

    run.finish()


if __name__ == "__main__":

    # with open("config.json", "r") as f:
    #     config = json.load(f)
    fire.Fire(main)
    # main(prompt_example=None,
    #      config=config)

