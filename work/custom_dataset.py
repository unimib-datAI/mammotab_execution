from transformers import AutoTokenizer
from torch.utils.data import Dataset
from tqdm_loggable.auto import tqdm
import torch
import json
import logging

logging.basicConfig(level=logging.INFO)


class CustomDataset(Dataset):

    def __init__(self, annotated_cells: set[str], tokenizer_name: str, file_path: str = "./mammotab_sample.jsonl"):
        super().__init__()
        self.file_path = file_path
        self.annotated_cells = annotated_cells
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, padding_side="left")
        self.data = self.load_data()

    def load_data(self):
        data = []
        with open(self.file_path) as json_file:
            for line in tqdm(json_file, total=84185):
                current_data = json.loads(line)
                table = current_data["table"]
                row = current_data["row"]
                column = current_data["column"]
                if len(self.tokenizer(text=current_data["prompt"])["input_ids"]) > self.tokenizer.model_max_length:
                    continue
                if f"{table}_{row}_{column}" not in self.annotated_cells:
                    data.append({
                        "prompt": current_data["prompt"],
                        "table": table,
                        "row": row,
                        "column": column,
                        "cell": current_data["cell"],
                        "response": current_data["response"]["response_string"]
                    })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):  # Convert tensor index to a list or integer
            idx = idx.item() if idx.numel() == 1 else idx.tolist()

        if isinstance(idx, list):  # If idx is a list of indices
            return [self.data[i] for i in idx]

        # Single index case
        if idx < 0 or idx >= len(self.data):  # Out-of-bounds check
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self.data)}")

        return self.data[idx]
