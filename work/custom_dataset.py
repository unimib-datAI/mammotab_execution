from transformers import AutoTokenizer
from torch.utils.data import Dataset
from tqdm_loggable.auto import tqdm
import torch
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    def __init__(
        self,
        annotated_cells: set[str],
        tokenizer_name: str,
        file_path: str = "./mammotab_sample.jsonl",
        max_workers: int = 4,
    ):
        super().__init__()
        self.file_path = file_path
        self.annotated_cells = annotated_cells
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, padding_side="left", use_fast=True
        )
        self.data = self._load_data_parallel(max_workers)

    def _process_line(self, line):
        """Process a single line from the JSONL file"""
        try:
            current_data = json.loads(line)
            table = current_data["table"]
            row = current_data["row"]
            column = current_data["column"]

            # Skip if already annotated
            if f"{table}_{row}_{column}" in self.annotated_cells:
                logger.debug(f"Skipping annotated cell: {table}_{row}_{column}")
                return None

            # Check token length
            input_ids = self.tokenizer(
                current_data["prompt"],
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]

            if len(input_ids) > self.tokenizer.model_max_length:
                logger.warning(
                    f"Skipping line due to token length ({len(input_ids)}) > max ({self.tokenizer.model_max_length})"
                )
                return None

            return {
                "prompt": current_data["prompt"],
                "table": table,
                "row": row,
                "column": column,
                "cell": current_data["cell"],
                "response": current_data["response"]["response_string"],
            }
        except Exception as e:
            logger.warning(f"Error processing line: {e}")
            return None

    def _load_data_parallel(self, max_workers: int) -> list:
        """Load and process data in parallel"""
        data = []

        with open(self.file_path, "r") as json_file:
            # First count lines for accurate progress bar
            total_lines = sum(1 for _ in json_file)
            print(f"Total lines in file: {total_lines}")
            json_file.seek(0)  # Reset file pointer

            # Process in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                process_fn = partial(self._process_line)
                results = list(
                    tqdm(
                        executor.map(process_fn, json_file),
                        total=total_lines,
                        desc="Loading dataset",
                    )
                )

        # Filter out None results
        data = [item for item in results if item is not None]
        logger.info(f"Loaded {len(data)} valid samples")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            return [self.data[i] for i in idx]

        if isinstance(idx, slice):
            return self.data[idx]

        return self.data[idx]
