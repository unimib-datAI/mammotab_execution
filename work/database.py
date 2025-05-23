from mongoengine import (
    connect,
    Document,
    StringField,
    IntField,
    BooleanField,
    FloatField,
)
from typing import List, Dict, Any
import logging
from pymongo import WriteConcern
from mongoengine.queryset.queryset import QuerySet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Cea(Document):
    """Optimized database model for storing inference results"""

    meta = {
        "collection": "cea_results",
        "indexes": [
            "model",
            # "table",  # Frequently filtered by table
            # {
            #    "fields": ["model", "table", "row", "column"],
            #    "unique": True,
            # },  # Compound index for unique cell reference
            "correct",  # For statistics queries
            {"fields": ["table", "correct"]},  # For table-specific accuracy
            {"fields": ["model", "correct"]},  # For model-specific accuracy
        ],
        "write_concern": WriteConcern(
            w=1, j=False
        ),  # Balance between safety and performance
    }

    model = StringField(required=True)
    prompt = StringField(required=True)
    cell = StringField(required=True)
    model_response = StringField(required=True)
    correct_response = StringField(required=True)
    table = StringField(required=True)
    row = IntField(required=True)
    column = IntField(required=True)
    correct = BooleanField(required=True)
    avg_time = FloatField(required=True, default=0.0)


class Missings(Document):
    """Optimized database model for tracking failed cells"""

    meta = {
        "collection": "missing_cells",
        "indexes": [
            {"fields": ["table", "row", "column"], "unique": True}  # Prevent duplicates
        ],
        "write_concern": WriteConcern(w=1, j=False),
    }

    cell = StringField(required=True)
    table = StringField(required=True)
    row = IntField(required=True)
    column = IntField(required=True)


class Database:
    """High-performance database operations for inference pipeline"""

    def __init__(self):
        # Connection with optimized settings
        connect(
            db="mammotab",
            host="mongo",
            port=27017,
            username="root",
            password="mammotab_execution",
            authentication_source="admin",
            connectTimeoutMS=30000,  # 30 second timeout
            socketTimeoutMS=None,  # No timeout for operations
            maxPoolSize=100,  # Larger connection pool
            minPoolSize=10,  # Minimum connections to maintain
            retryWrites=True,  # Enable retryable writes
        )

        # Create indexes if they don't exist
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Ensure all indexes are created (idempotent operation)"""
        Cea.ensure_indexes()
        Missings.ensure_indexes()
        logger.info("Database indexes verified/created")

    def save_missings(self, cell: str, table: str, row: int, column: int) -> None:
        """Save a single missing cell record"""
        try:
            Missings(cell=cell, table=table, row=row, column=column).save()
        except Exception as e:
            logger.warning(f"Failed to save missing cell {table}_{row}_{column}: {e}")

    def bulk_save_missings(self, records: List[Dict[str, Any]]) -> int:
        """Bulk save missing cells with optimized performance"""
        if not records:
            return 0

        try:
            # Use bulk_insert with ordered=False for best performance
            result = Missings._get_collection().insert_many(
                [dict(record) for record in records],
                ordered=False,  # Continue on errors
            )
            return len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Bulk save missings partially failed: {e}")
            # Fallback to individual saves
            success_count = 0
            for record in records:
                try:
                    self.save_missings(**record)
                    success_count += 1
                except Exception:
                    continue
            return success_count

    def save_response(self, **kwargs) -> None:
        """Save a single inference result"""
        try:
            Cea(**kwargs).save()
        except Exception as e:
            logger.error(f"Failed to save response: {e}")

    def bulk_save_responses(self, records: List[Dict[str, Any]]) -> int:
        """Bulk save inference results with optimized performance"""
        if not records:
            return 0

        try:
            # Use bulk_write for best performance
            operations = [Cea(**record).to_mongo() for record in records]
            result = Cea._get_collection().bulk_write(
                [InsertOne(op) for op in operations],
                ordered=False,  # Continue on errors
            )
            return result.inserted_count
        except Exception as e:
            logger.error(f"Bulk save responses partially failed: {e}")
            # Fallback to individual saves
            success_count = 0
            for record in records:
                try:
                    self.save_response(**record)
                    success_count += 1
                except Exception:
                    continue
            return success_count

    def get_all_documents(self, model_name: str) -> QuerySet:
        """Optimized query for all documents with projection"""
        return (
            Cea.objects.filter(model=model_name)
            .only(
                "table",
                "row",
                "column",
                "correct",
                "model",
                "avg_time",
                "correct_response",  # Only fetch needed fields
            )
            .timeout(False)
        )  # No timeout for large queries

    def get_all_documents_full(self) -> QuerySet:
        """Optimized query for all documents with projection"""
        return Cea.objects.only(
            "table",
            "row",
            "column",
            "correct",
            "model",
            "avg_time",
            "cell",
            "model_response",
            "correct_response",
        ).timeout(False)  # No timeout for large queries

    def get_stats_by_model(self, model_name: str) -> Dict[str, float]:
        """Get statistics for a specific model"""
        pipeline = [
            {"$match": {"model": model_name}},
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "correct": {"$sum": {"$cond": ["$correct", 1, 0]}},
                    "avg_time": {"$avg": "$avg_time"},
                }
            },
        ]

        result = list(Cea._get_collection().aggregate(pipeline))
        if result:
            return {
                "accuracy": result[0]["correct"] / result[0]["total"],
                "avg_time": result[0]["avg_time"],
                "total": result[0]["total"],
            }
        return {"accuracy": 0, "avg_time": 0, "total": 0}

    def get_table_stats(self, table_name: str) -> Dict[str, float]:
        """Get statistics for a specific table"""
        pipeline = [
            {"$match": {"table": table_name}},
            {
                "$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "correct": {"$sum": {"$cond": ["$correct", 1, 0]}},
                    "avg_time": {"$avg": "$avg_time"},
                }
            },
        ]

        result = list(Cea._get_collection().aggregate(pipeline))
        if result:
            return {
                "accuracy": result[0]["correct"] / result[0]["total"],
                "avg_time": result[0]["avg_time"],
                "total": result[0]["total"],
            }
        return {"accuracy": 0, "avg_time": 0, "total": 0}
