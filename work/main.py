import os
import json
from time import time
from generate import LLM
from tqdm_loggable.auto import tqdm
from database import Database
from export import Export
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO)

HF_TOKEN = os.getenv("HF_TOKEN")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
model_name = os.getenv("MODEL_NAME")
tokenizer_name = os.getenv("TOKENIZER_NAME")

login(token=HF_TOKEN)

try:
    AutoModel.from_pretrained(model_name)
    AutoTokenizer.from_pretrained(tokenizer_name)
    print(
        f"Successfully loaded model: {model_name} and tokenizer: {tokenizer_name}")
except Exception as e:
    raise ValueError(f"Invalid model or tokenizer name: {e}")

db = Database()


def get_annotated_cells() -> set[str]:
    print("Load Annotated Cells")
    annotated_cells_set = set[str]()
    annotated_cells = db.get_all_documents()
    for cell in tqdm(annotated_cells):
        annotated_cells_set.add(f"{cell.table}_{cell.row}_{cell.column}")

    return annotated_cells_set


cell_set_annotated = get_annotated_cells()

custom_dataset = CustomDataset(
    annotated_cells=cell_set_annotated, tokenizer_name=tokenizer_name)
llm = LLM(model_name=model_name, tokenizer_name=tokenizer_name)


mammotab_dataloader = DataLoader(custom_dataset, batch_size=BATCH_SIZE)
for batch in tqdm(mammotab_dataloader):
    prompts = batch["prompt"]
    responses = batch["response"]
    rows = batch["row"].tolist()
    columns = batch["column"].tolist()
    cells = batch["cell"]
    tables = batch["table"]
    time_elapsed = 0
    try:
        start = time()
        llm_responses = llm.generate(prompts)
        end = time()
        time_elapsed = end - start
    except Exception as e:
        for row, column, cell, table in zip(prompts, responses, rows, columns, cells, tables, llm_responses):
            db.save_missings(cell=cell, table=table, row=row, column=column)
        continue
    for prompt, response, row, column, cell, table, llm_response in zip(prompts, responses, rows, columns, cells, tables, llm_responses):
        db.save_response(model_name=model_name, prompt=prompt, cell=cell, table=table, row=row, column=column,
                         correct_response=response, model_response=llm_response, correct=response == llm_response, avg_time=time_elapsed/BATCH_SIZE)


export = Export(db=db)
stats_export = export.compute_stats()

with open(f'./{model_name.split("/")[-1]}.json', 'w', encoding='utf-8') as f:
    json.dump(stats_export, f)
