import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_ROOT = Path.home() / "mammotab_data"


class Database:
    """Filesystem-based database operations for inference pipeline"""

    def __init__(self):
        # Create data directories if they don't exist
        (DATA_ROOT / "cea").mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / "missings").mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized filesystem storage at {DATA_ROOT}")

    def _get_cea_path(self, table: str, row: int, column: int) -> Path:
        """Get filesystem path for CEA record"""
        return DATA_ROOT / "cea" / table / f"{row}_{column}.json"

    def _get_missing_path(self, table: str, row: int, column: int) -> Path:
        """Get filesystem path for missing record"""
        return DATA_ROOT / "missings" / table / f"{row}_{column}.json"

    def save_missings(self, cell: str, table: str, row: int, column: int) -> None:
        """Save a single missing cell record to filesystem"""
        try:
            path = self._get_missing_path(table, row, column)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {"cell": cell, "table": table, "row": row, "column": column}, f
                )
        except Exception as e:
            logger.warning(f"Failed to save missing cell {table}_{row}_{column}: {e}")

    def bulk_save_missings(self, records: List[Dict[str, Any]]) -> int:
        """Bulk save missing cells to filesystem"""
        success_count = 0
        for record in records:
            try:
                self.save_missings(
                    cell=record["cell"],
                    table=record["table"],
                    row=record["row"],
                    column=record["column"],
                )
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to save missing record: {e}")
        return success_count

    def save_response(self, **kwargs) -> None:
        """Save a single inference result to filesystem"""
        try:
            path = self._get_cea_path(kwargs["table"], kwargs["row"], kwargs["column"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(kwargs, f)
        except Exception as e:
            logger.error(f"Failed to save response: {e}")

    def bulk_save_responses(self, records: List[Dict[str, Any]]) -> int:
        """Bulk save inference results to filesystem"""
        success_count = 0
        for record in records:
            try:
                self.save_response(**record)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to save response: {e}")
        return success_count

    def get_all_documents(self, model_name: str) -> List[Dict]:
        """Get all documents for a specific model from filesystem"""
        results = []
        for table_dir in (DATA_ROOT / "cea").iterdir():
            if not table_dir.is_dir():
                continue
            for file_path in table_dir.glob("*.json"):
                with open(file_path) as f:
                    data = json.load(f)
                    if data.get("model") == model_name:
                        results.append(
                            {
                                "table": data["table"],
                                "row": data["row"],
                                "column": data["column"],
                                "correct": data["correct"],
                                "model": data["model"],
                                "avg_time": data["avg_time"],
                            }
                        )
        return results

    def get_all_documents_full(self) -> List[Dict]:
        """Get all documents with full details from filesystem"""
        results = []
        for table_dir in (DATA_ROOT / "cea").iterdir():
            if not table_dir.is_dir():
                continue
            for file_path in table_dir.glob("*.json"):
                with open(file_path) as f:
                    data = json.load(f)
                    results.append(
                        {
                            "table": data["table"],
                            "row": data["row"],
                            "column": data["column"],
                            "correct": data["correct"],
                            "model": data["model"],
                            "avg_time": data["avg_time"],
                            "cell": data["cell"],
                            "model_response": data["model_response"],
                            "correct_response": data["correct_response"],
                        }
                    )
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
