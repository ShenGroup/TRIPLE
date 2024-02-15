#!/bin/bash
# You need to collect the embedded prompts to run this code, for the prompt collection, please run data_collection.sh
tasks=("movie_recommendation")

for task in "${tasks[@]}"
do
    for budget in 5 # Specify the budget you what to use here
    do
        for random_seed in 2 4 6 8 10 # Specify the random seed you want to use here
        do
            for algorithm in  "GSE" # You can choose from TopK, UCB, SequentialHalving, ContinuousRejects, Cluster and GSE
            do
                python bandit_exp.py --task "$task" --budget "$budget" --bandit_algorithm "$algorithm" --random_seed "$random_seed" --storage_path "Your/Path/to/prompts" --llm "ChatGPT"
            done
        done
    done
done

echo "All tasks have been completed."
