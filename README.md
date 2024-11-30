# Foundation LLMs
Try out fine-tuning a couple of different open-source foundation LLMs to get one that works best with your UI for inference.

Inference will be on CPUs, so big models will be slow. We selected 3 small foundation models that have good performance on CPU:
1. Mistral-Small-Instruct-2409,     
2. Phi-3.5-mini-instruct,           
3. Llama-3.2-1B-Instruct-bnb-4bit,

After experiments with 3 models, we found Phi-3.5-mini-instruct perofrms better on the target dataset `mlabonne/FineTome-100k`. Detailed metrics are under `./foundation_model_metrics`.

# Data-centric approach
identify new data sources that enable you to train a better model that one provided in the blog post.

# Model-centric approach
e.g., tune hyperparameters, change the fine-tuning model architecture, etc.

# Fine-tuning frameworks
You are free to use other fine-tuning frameworks, such as Axolotl of HF FineTuning.