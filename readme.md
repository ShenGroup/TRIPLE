# Efficient Prompt Optimization Through the Lens of Best Arm Identification (NeurIPS 2024)

This repository contains the official implementation of our NeurIPS 2024 paper, **"[Efficient Prompt Optimization Through the Lens of Best Arm Identification](https://arxiv.org/pdf/2402.09723)"**. We introduce **TRIPLE**, a principled and cost-effective framework for automatic prompt engineering.

---

## 🎯 Introduction

Optimizing prompts to elicit the best possible responses from Large Language Models (LLMs) is a critical but often expensive task. Traditional methods rely on extensive trial-and-error, best-of-n sampling, or human-in-the-loop tuning, all of which consume significant computational resources and API costs.

This project tackles the problem of **budget-limited prompt optimization**. We establish a novel connection between this challenge and the **Best Arm Identification (BAI)** problem from the Multi-Armed Bandit (MAB) literature. Our framework, **TRIPLE**, reframes prompt selection as a statistical process of identifying the single best "arm" (prompt) from a set of candidates within a fixed budget of LLM evaluations.

This approach allows us to leverage powerful, theoretically-grounded BAI algorithms to create a more efficient and automated prompt selection pipeline.

---

## 🔎 The TRIPLE Framework

The **TRIPLE** framework is built on a simple yet powerful analogy:

* **Arms**: Each candidate prompt in the search space is an "arm" of a multi-armed bandit.
* **Arm Pulls**: Evaluating a prompt by sending it to an LLM and receiving a response is equivalent to "pulling" an arm.
* **Reward**: The quality or performance score of the LLM's response serves as the "reward" for that arm pull.
* **Budget**: The maximum number of allowed LLM evaluations is the fixed budget, $T$.

The goal is to maximize the probability of identifying the best prompt (the arm with the highest mean reward) without exceeding the budget. We adapt and implement two state-of-the-art BAI algorithms for this task.

---

## ✨ Key Algorithms and Features

This repository implements the core algorithms and extensions presented in our paper:

### Core BAI Algorithms
* **TRIPLE-SH (Sequential Halving)**: This algorithm evenly divides the budget across rounds. In each round, it allocates budget to a set of surviving prompts and eliminates the worst-performing half, effectively focusing resources on the most promising candidates over time.
* **TRIPLE-CR (Continuously Reject)**: A more adaptive elimination strategy. It continuously samples from the set of active prompts and uses a rejection criterion derived from the Large Deviation Principle to discard underperforming prompts as soon as there is sufficient statistical confidence.

### Scalability Enhancements for Large Prompt Pools
To handle scenarios with thousands of candidate prompts, we introduce two embedding-based enhancements that enable effective information sharing:

* **TRIPLE-CLST (Clustering)**: Before selection, prompts are converted to embeddings and clustered. The BAI algorithm then operates on these clusters, sharing evaluation information among semantically similar prompts to accelerate learning.
* **TRIPLE-GSE (Gaussian Process with Squared Exponential Kernel)**: This approach models the relationship between prompt embeddings and their scores using a Gaussian Process. It uses a kernel function to capture similarities between prompts, allowing for more efficient and informed exploration of the vast prompt space.

### Extension to Few-Shot Example Selection
The TRIPLE framework is extended to tackle the complex combinatorial problem of selecting the optimal set of few-shot examples for in-context learning, framing it as a **Combinatorial MAB (CMAB)** problem.

---

## 📊 Experimental Results

Our experiments demonstrate that TRIPLE significantly outperforms existing baselines in identifying high-performing prompts across various tasks while adhering to a strict budget.

### Experimental results

![Performance with 30 candidate prompts and 150 budget](./exp_results/P_30_N_150.png)

![Performance with 150 candidate prompts and only 100 budget](./exp_results/P_150_N_100.png)



---

## 🚀 Getting Started

To generate prompts, generate the embeddings and perform the clustering process, simply run

```
bash data_collection.sh
```

After collecting the prompts, run this script to compare different BAI algorithm's performance:

```
bash main_exp.sh
```

---

If you find our work useful, please consider citing our paper:
```
@inproceedings{shi2024efficient,
  title={Efficient Prompt Optimization Through the Lens of Best Arm Identification},
  author={Shi, Chengshuai and Yang, Kun and Chen, Zihan and Li, Jundong and Yang, Jing and Shen, Cong},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
  year={2024},
  url={[https://arxiv.org/abs/2402.09723](https://arxiv.org/abs/2402.09723)}
}
```

---

## Acknowledgements
We have referenced the following repos for code development:

[APE](https://github.com/keirp/automatic_prompt_engineer)
[InstructZero](https://github.com/Lichang-Chen/InstructZero)
[APO](https://github.com/microsoft/LMOps/tree/main/prompt_optimization)