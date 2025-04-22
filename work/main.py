import json
from custom_dataset import CustomDataset
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

HF_TOKEN = "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"
INITIAL_BATCH_SIZE = 8
MIN_BATCH_SIZE = 1

models = ["google/gemma-2b", "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct", "meta-llama/Llama-3.2-1B",
          "meta-llama/Llama-3.2-3B", "Qwen/Qwen2-0.5B", "Qwen/Qwen2-1.5B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-7B", "01-ai/Yi-1.5-6B"]

login(token=HF_TOKEN)

print("COMPUTE DATASET STATISTICS")
statistics = []
for model in models:
    custom_dataset = CustomDataset(
        annotated_cells=set(), tokenizer_name=model
    )

    statistics.append({
        "model": model,
        "max_context": custom_dataset._get_model_max_length(model),
        "total_prompts": 84907,
        "out_of_context_prompts": custom_dataset.total_out_of_context,
        "annotable_prompts": 84907 - custom_dataset.total_out_of_context
    })

    print({
        "model": model,
        "max_context": custom_dataset._get_model_max_length(model),
        "total_prompts": 84907,
        "out_of_context_prompts": custom_dataset.total_out_of_context,
        "annotable_prompts": 84907 - custom_dataset.total_out_of_context
    })

with open('data.json', 'w') as f:
    json.dump(statistics, f)
