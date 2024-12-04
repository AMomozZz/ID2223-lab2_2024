# ID2223 lab2 2024

## Foundation LLMs

Inference will be on CPUs, so big models will be slow. We selected `Llama-3.2-1B-Instruct`, which is a small model and inference fast on CPUs. It's also suitable for fine-tuning.

Initially, we experimented with several models, including `Llama-3.2-3B-Instruct`, `Mistral-Small-Instruct-2409` and `Phi-3.5-mini-instruct`. After training each model for 60 steps, we observed that Phi-3.5-mini-instruct had the lowest evaluation loss.

However, further research, including insights from [this tweet](https://x.com/AIatMeta/status/1839018085329809831) posted by [AI at Meta](https://x.com/AIatMeta), showed that Llama 3.2 performs better than Phi-3.5-mini on most benchmarks. So we decided to continue using Llama 3.2 as the base model for further fine-tuning.

## Dataset

We used `orpo-dpo-mix-40k`[2], which is a dataset specialized in ORPO.

We used [`FineTome-100k`](https://huggingface.co/datasets/mlabonne/FineTome-100k) as the evaluation dataset, which is a subset of [`The-Tome`](https://huggingface.co/datasets/arcee-ai/The-Tome). `The Tome` is a curated dataset designed for training large language models with a focus on instruction following.

## Model-centric approach

<!-- e.g., tune hyperparameters, change the fine-tuning model architecture, etc. -->

TODO: tune hyperparameters

Original post presents a supervised fine-tuning(SFT) architecture based on a series of models. However, it will also generate undesirable answers[1].

<img src="report/reject.png" alt="drawing" width="400"/>

Therefore, we added a preference alignment stage to widden the gap between the preferred and rejected outputs. Traditionally, the two stages are separate and needs Reinforcement Learning with Human Feedback (RLHF) or Direct Preference Optimization (DPO). Inspired by ORPO[1] [5], we adopted the ORPO method, which elegantly combines these two stages into one and showed clear improvements compared with previous approaches, as demonstrated in the literature.

<img src="report/metrics.png" alt="drawing" width="400"/>

## Evaluation

It's hard to determine how to compare two large language models. Traditional evaluation metrics based on the similarity between outputs and reference answers (e.g., ROUGE, BLEU) seems ineffective.

The LLM-as-a-judge[3] approach seems to be a good option. Due to cost of proprietary models like chatgpt, we used an open-source 3.8B LM judge: Flow Judge[4] for LLM system evaluations.

We compared our model with the base `Llama-3.2-1B-Instruct` model.

TODO: results

## Reference

[1] Hong, J., Lee, N., & Thorne, J. (2024, March 12). ORPO: Monolithic Preference Optimization without Reference Model. arXiv.org. <https://arxiv.org/abs/2403.07691>

[2] mlabonne/orpo-dpo-mix-40k · Datasets at Hugging Face. (2001, June 2). <https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k>

[3] Using LLM-as-a-judge 🧑‍⚖️ for an automated and versatile evaluation - Hugging Face Open-Source AI Cookbook. (n.d.). <https://huggingface.co/learn/cookbook/en/llm_judge>

[4] Flowaicom. (n.d.). GitHub - flowaicom/flow-judge: Code for evaluating with Flow-Judge-v0.1 - an open-source, lightweight (3.8B) language model optimized for LLM system evaluations. Crafted for accuracy, speed, and customization. GitHub. <https://github.com/flowaicom/flow-judge>

[5] Fine-tune Llama 3 with ORPO. (n.d.). <https://huggingface.co/blog/mlabonne/orpo-llama-3>

## Appendix

We attempted to use Flow Judge on a Windows system, but encountered an issue because Flow Judge relies on `os.setsid`, a function that is only supported on Linux systems. To work around this, we decided to continue our attempts within a Jupyter Notebook environment in WSL (Windows Subsystem for Linux). However, in this setup, we found that Flow Judge was unable to connect to the Llamafile server running on the host machine,which is started by function in Flow Judge, preventing the evaluation and testing from proceeding as expected.
