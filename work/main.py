import os
import argparse
from time import time
from generate import LLM
from tqdm_loggable.auto import tqdm
from database import Database
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
import logging
from dotenv import load_dotenv

load_dotenv()
MIN_BATCH_SIZE = 1

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


parser = argparse.ArgumentParser(
    description="Run the main script with arguments.")
parser.add_argument("--model_name", type=str,
                    help="Name of the Hugging Face model")
parser.add_argument("--batch_size", type=str, help="Batch size for processing")
parser.add_argument("--hf_token", type=str, help="Hugging Face token")
parser.add_argument("--input_file", type=str, help="Path to the input file")

args = parser.parse_args()

model_name = args.model_name or os.getenv("MODEL_NAME")
tokenizer_name = model_name
HF_TOKEN = args.hf_token or os.getenv("HF_TOKEN")
INITIAL_BATCH_SIZE = (
    int(args.batch_size) if args.batch_size else int(
        os.getenv("BATCH_SIZE", "8"))
)
CHUNK_FILE = args.input_file or os.getenv(
    "CHUNK_FILE", "./mammotab_sample.jsonl")
db = Database(model_name=model_name)

login(token=HF_TOKEN)

try:
    AutoModel.from_pretrained(model_name, trust_remote_code=True)
    AutoTokenizer.from_pretrained(tokenizer_name)
    print(
        f"Successfully loaded model: {model_name} and tokenizer: {tokenizer_name}")
except Exception as e:
    raise ValueError(f"Invalid model or tokenizer name: {e}")


custom_dataset = CustomDataset(
    tokenizer_name=tokenizer_name,
    file_path=CHUNK_FILE,
)
llm = LLM(model_name=model_name, tokenizer_name=tokenizer_name)


def process_batch(batch, current_batch_size):
    try:
        # First validate the batch structure
        required_keys = ["prompt", "response",
                         "row", "column", "cell", "table"]
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Missing required key in batch: {key}")
            if len(batch[key]) == 0:
                raise ValueError(f"Empty array for key: {key}")

        prompts = batch["prompt"]
        responses = batch["response"]
        rows = batch["row"].tolist()
        columns = batch["column"].tolist()
        cells = batch["cell"]
        tables = batch["table"]

        logging.info(f"Processing batch of size {current_batch_size}")
        logging.debug(
            f"First prompt: {prompts[0] if len(prompts) > 0 else 'N/A'}")

        start = time()
        chunk_size = int(os.getenv("CHUNK_SIZE", 64))
        llm_responses = llm.generate(prompts, chunk_size=chunk_size)
        end = time()
        time_elapsed = end - start

        # Validate LLM responses match expected count
        if len(llm_responses) != len(prompts):
            raise ValueError(
                f"LLM responses count mismatch. Expected {len(prompts)}, got {len(llm_responses)}"
            )

        # If successful, save results
        for i in range(len(prompts)):
            db.save_response(chunk_file=CHUNK_FILE,
                             model=model_name,
                             prompt=prompts[i],
                             cell=cells[i],
                             table=tables[i],
                             row=rows[i],
                             column=columns[i],
                             correct_response=responses[i],
                             model_response=llm_responses[i],
                             correct=responses[i] == llm_responses[i],
                             avg_time=time_elapsed / current_batch_size,
                             )
        return True

    except Exception as e:
        logging.error(
            f"Batch processing failed with error: {str(e)}", exc_info=True)
        # Save the failed items as missing
        for i in range(len(batch.get("prompt", []))):
            try:
                db.save_missings(
                    cell=batch["cell"][i],
                    table=batch["table"][i],
                    row=batch["row"][i],
                    column=batch["column"][i],
                )
            except Exception as db_error:
                logging.error(f"Failed to save missing item: {db_error}")
        return False


def process_with_retry(batch_items, initial_batch_size):
    current_batch_size = initial_batch_size
    remaining_items = batch_items

    while remaining_items and current_batch_size >= MIN_BATCH_SIZE:
        try:
            # Create a smaller batch
            current_batch = {
                key: value[:current_batch_size]
                for key, value in remaining_items.items()
            }

            # Validate we have items to process
            if len(current_batch.get("prompt", [])) == 0:
                logging.info("No more items to process in this batch")
                break

            success = process_batch(current_batch, current_batch_size)

            if success:
                # Remove processed items and reset batch size for next try
                remaining_items = {
                    key: value[current_batch_size:]
                    for key, value in remaining_items.items()
                }
                current_batch_size = initial_batch_size  # Reset to original size
            else:
                # Reduce batch size for next attempt
                new_batch_size = max(MIN_BATCH_SIZE, current_batch_size // 2)
                if new_batch_size == current_batch_size:
                    break  # Prevent infinite loop when we can't reduce further
                current_batch_size = new_batch_size
                logging.info(f"Reducing batch size to {current_batch_size}")

        except Exception as e:
            logging.error(f"Unexpected error in retry loop: {e}")
            break

    if remaining_items and len(remaining_items.get("prompt", [])) > 0:
        logging.error(
            f"Failed to process {len(remaining_items['prompt'])} items")
        logging.debug(f"First failed prompt: {remaining_items['prompt'][0]}")


# Create a single DataLoader with the maximum batch size
dataloader = DataLoader(custom_dataset, batch_size=INITIAL_BATCH_SIZE)

for batch in tqdm(dataloader):
    process_with_retry(batch, INITIAL_BATCH_SIZE)
