import os
from generate import LLM
from tqdm import tqdm
from database import Database
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

model_name = os.getenv("MODEL_NAME")
tokenizer_name = os.getenv("TOKENIZER_NAME")

try:
    AutoModel.from_pretrained(model_name)
    AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Successfully loaded model: {model_name} and tokenizer: {tokenizer_name}")
except Exception as e:
    raise ValueError(f"Invalid model or tokenizer name: {e}")

batch_size = 8

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


mammotab_dataloader = DataLoader(custom_dataset, batch_size=batch_size)
for batch in tqdm(mammotab_dataloader):
    prompts = batch["prompt"]
    responses = batch["response"]
    rows = batch["row"].tolist()
    columns = batch["column"].tolist()
    cells = batch["cell"]
    tables = batch["table"]
    try:
        llm_responses = llm.generate(prompts)
    except Exception as e:
        for row, column, cell, table in zip(prompts, responses, rows, columns, cells, tables, llm_responses):
            db.save_missings(cell=cell, table=table, row=row, column=column)
        continue
    for prompt, response, row, column, cell, table, llm_response in zip(prompts, responses, rows, columns, cells, tables, llm_responses):
        db.save_response(model_name=model_name, prompt=prompt, cell=cell, table=table, row=row, column=column,
                         correct_response=response, model_response=llm_response, correct=response == llm_response)
