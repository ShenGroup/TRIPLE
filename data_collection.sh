#!/bin/bash
# Script for data collection
tasks=("rhymes" "cause_and_effect" "larger_animal")

for task in "${tasks[@]}"
do
    python collect_baselines.py --task "$task" --num_prompts 30 --num_clusters 6 --llm_choice WhiteBox
done
