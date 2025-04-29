import os
from pathlib import Path
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_ROOT = Path.home() / "mammotab_data"


class Database:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.saving_path = model_name.split("/")[-1] + ".jsonl"

    def save_missings(self, cell: str, table: str, row: int, column: int) -> None:
        """Save a single missing cell record to filesystem"""
        try:
            path = Path(self.saving_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {"cell": cell, "table": table, "row": row, "column": column}, f
                )
        except Exception as e:
            logger.warning(
                f"Failed to save missing cell {table}_{row}_{column}: {e}")

    def save_response(self, chunk_file: str, **kwargs) -> None:
        """Save a single inference result to filesystem"""
        try:
            # Create directory if it doesn't exist (safer than just file creation)
            output_dir = Path("chunks_results")
            output_dir.mkdir(exist_ok=True)

            current_file = chunk_file.split("/")[1]
            file_path = output_dir / current_file

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(kwargs, ensure_ascii=False) + "\n")

        except IOError as e:
            logger.error(f"Failed to save response to {file_path}: {e}")
        except json.JSONEncodeError as e:
            logger.error(f"Failed to serialize data to JSON: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving response: {e}")

    def get_all_documents(self, model_name: str) -> List[Dict]:
        """Get all documents for a specific model from filesystem"""
        results = []
        chunk_resilts_path = Path("chunks_results")
        for jsonl_file in chunk_resilts_path.glob("*.jsonl"):

            if not jsonl_file.exists():
                logger.warning(
                    f"Result file not found for model {model_name} at {jsonl_file.name}"
                )
                return results
            try:
                with jsonl_file.open('r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            line = line.strip()
                            if not line:
                                continue

                            data = json.loads(line)
                            results.append(
                                {
                                    "table": data.get("table"),
                                    "row": data.get("row"),
                                    "column": data.get("column"),
                                    "correct": data.get("correct"),
                                    "model": data.get(
                                        "model"
                                    ),  # Keep model name for potential filtering
                                    "avg_time": data.get(
                                        "avg_time", 0.0
                                    ),  # Default avg_time if missing
                                }
                            )
                        except json.JSONDecodeError as json_err:
                            logger.warning(
                                f"Skipping invalid JSON line in {jsonl_file.name}: {json_err}. Line: '{line[:100]}...'"
                            )
                        except KeyError as key_err:
                            logger.warning(
                                f"Skipping line with missing key {key_err} in {jsonl_file.name}. Line: '{line[:100]}...'"
                            )
                        except (
                            Exception
                        ) as line_err:  # Catch other potential errors per line
                            logger.warning(
                                f"Error processing line in {jsonl_file.name}: {line_err}. Line: '{line[:100]}...'"
                            )

            except Exception as e:
                logger.error(
                    f"Failed to read or process file {jsonl_file.name}: {e}")

        print("PROCESSED:", len(results))
        return results

    def get_stats_by_model(self, model_name: str) -> Dict[str, float]:
        """Calculate statistics for a specific model"""
        total = 0
        correct = 0
        total_time = 0.0

        for doc in self.get_all_documents(model_name):
            total += 1
            if doc["correct"]:
                correct += 1
            total_time += doc["avg_time"]

        return {
            "accuracy": correct / total if total > 0 else 0,
            "avg_time": total_time / total if total > 0 else 0,
            "total": total,
        }

    def get_table_stats(self, table_name: str) -> Dict[str, float]:
        """Calculate statistics for a specific table"""
        total = 0
        correct = 0
        total_time = 0.0

        table_dir = DATA_ROOT / "cea" / table_name
        if not table_dir.exists():
            return {"accuracy": 0, "avg_time": 0, "total": 0}

        for file_path in table_dir.glob("*.json"):
            with open(file_path) as f:
                data = json.load(f)
                total += 1
                if data["correct"]:
                    correct += 1
                total_time += data["avg_time"]

        return {
            "accuracy": correct / total if total > 0 else 0,
            "avg_time": total_time / total if total > 0 else 0,
            "total": total,
        }
