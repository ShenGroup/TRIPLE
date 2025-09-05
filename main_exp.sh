#!/bin/bash
# "antonyms" "cause_and_effect" "common_concept" "informal_to_formal" "larger_animal" "taxonomy_animal" "negation" "Cluster" "TopK" "UCB" "SequentialHalving" "ContinuousRejects"  "rhymes" "orthography_starts_with" "taxonomy_animal" "antonyms"
# tasks=( "cause_and_effect" "common_concept"  "larger_animal"  "orthography_starts_with" "rhymes"  "sentence_similarity" "snarks" "sports_understanding")
# tasks=("cause_and_effect" "disambiguation_qa" "common_concept" "gender_inclusive_sentences_german" "larger_animal" "hyperbaton" "word_sorting" "snarks" "movie_recommendation"  "question_selection" "object_counting" "orthography_starts_with"  "ryhmes")
tasks=("snarks")
# "rhymes" "ruin_names" "rhymes" "sentence_similarity" "snarks" "sports_understanding" "timedial" "word_sorting" "cause_and_effect" "disambiguation_qa" "hyperbaton" "larger_animal"  "movie_recommendation"
# "TopK" "UCB" "SequentialHalving" "ContinuousRejects" "orthography_starts_with" "rhymes" 
for task in "${tasks[@]}"
do
    for budget in 5
    do
        for random_seed in 2 4 6 8
        do
            for algorithm in  "GSE" "NeuralUCB" "TopK" "Cluster"
            do
                python bandit_exp.py --task "$task" --budget "$budget" --bandit_algorithm "$algorithm" --random_seed "$random_seed" --storage_path "./data/useful_tasks/embeddings/100prompts/" --llm "ChatGPT"
            done
        done
        # Wait for all algorithms to complete for the current task and random seed
        # for pid in "${pids[@]}"
        # do
        #     wait "$pid"
        # done
    done
done

echo "All tasks have been completed."
