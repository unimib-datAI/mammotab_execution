from database import Database
import math
import torch
import json
import csv
import platform
import os

model_name = os.getenv("MODEL_NAME")
test_locally = os.getenv("TEST_LOCALLY", "False").lower() == "true"


class Export:
    def __init__(self, db: Database):
        self.TOTAL_CELLS = 84907
        self.db = db
        self.stats = self.load_stats()

    def load_stats(self):
        stats_dict = {}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "general_stats_per_table.json")
        with open(file_path, "r") as file:
            for data in json.load(file):
                stats_dict[data["table"]] = data["stats"]

        return stats_dict

    def truncate(self, number, digits) -> float:
        # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
        nbDecimals = len(str(number).split(".")[1])
        if nbDecimals <= digits:
            return number
        stepper = 10.0**digits
        return math.trunc(stepper * number) / stepper

    def compute_stats(self):
        all_documents: list[Cea] = self.db.get_all_documents(model_name=model_name)

        # First, calculate accuracy per table
        table_statistics = {}
        for document in all_documents:
            table = document.table
            if table not in table_statistics:
                table_statistics[table] = {"total": 0, "correct": 0}

            table_statistics[table]["total"] += 1
            if document.correct:
                table_statistics[table]["correct"] += 1

        # Calculate accuracy per table and identify tables above threshold
        tables_above_threshold = set()
        for table, stats in table_statistics.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
                if accuracy >= 0.2:  # 20% threshold
                    tables_above_threshold.add(table)

        needed_stats = [
            "nils",
            "col_tag",
            "row_tag",
            "acro_added",
            "typos_added",
            "alias_added",
            "table_generic_types",
            "table_specific_types",
            "count_single_domain",
            "count_multi_domain",
            "small_per_cols",
            "medium_per_cols",
            "large_per_cols",
            "small_per_rows",
            "medium_per_rows",
            "large_per_rows",
        ]

        table_level_stats = [
            "col_tag",
            "row_tag",
            "count_single_domain",
            "count_multi_domain",
            "table_generic_types",
            "table_specific_types",
            "acro_added",
            "typos_added",
            "alias_added",
            "small_per_cols",
            "medium_per_cols",
            "large_per_cols",
            "small_per_rows",
            "medium_per_rows",
            "large_per_rows",
        ]

        counted_tables = {stat: set() for stat in table_level_stats}
        cell_set = set()
        total_correct = 0
        total_computed = 0
        total_time = 0.0
        model_stats = {
            "nils": 0,
            "acro_added": 0,
            "typos_added": 0,
            "alias_added": 0,
            "table_generic_types": 0,
            "table_specific_types": 0,
            "count_single_domain": 0,
            "count_multi_domain": 0,
            "small_per_cols": 0,
            "medium_per_cols": 0,
            "large_per_cols": 0,
            "small_per_rows": 0,
            "medium_per_rows": 0,
            "large_per_rows": 0,
        }

        model_stats_total = {
            "nils": 14856,
            "entities_found": 71500,
            "entities_not_found": 23528,
            "table_generic_types": 96,
            "table_specific_types": 774,
            "acro_added": 3518,
            "typos_added": 12135,
            "alias_added": 7117,
            "count_single_domain": 435,
            "count_multi_domain": 435,
            "small_per_cols": 152,
            "medium_per_cols": 465,
            "large_per_cols": 253,
            "small_per_rows": 394,
            "medium_per_rows": 232,
            "large_per_rows": 244,
            "total_tables": 870,
            "total_rows": 37820,
            "total_cols": 5252,
            "total_cells": 133137,
        }

        mapping_dict = {
            "nils": "nils",
            "acro_added": "acronyms",
            "typos_added": "typos",
            "alias_added": "alias",
            "table_generic_types": "generic_types",
            "table_specific_types": "specific_types",
            "count_single_domain": "single_domain",
            "count_multi_domain": "multi_domain",
            "small_per_cols": "small_per_cols",
            "medium_per_cols": "medium_per_cols",
            "large_per_cols": "large_per_cols",
            "small_per_rows": "small_per_rows",
            "medium_per_rows": "medium_per_rows",
            "large_per_rows": "large_per_rows",
        }

        acro_typo_alias = {}

        for document in all_documents:
            table = document.table
            row = document.row
            column = document.column
            if f"{table}_{row}_{column}" not in cell_set:
                total_computed += 1
                if document.avg_time is not None:
                    total_time += document.avg_time
                    cell_set.add(f"{table}_{row}_{column}")
                    if document.correct:
                        total_correct += 1

                    # Process cell-level stats (always)
                    for stat in self.stats[table]:
                        if stat in needed_stats and stat not in table_level_stats:
                            stat_value = self.stats[table][stat]
                            if isinstance(stat_value, str):
                                try:
                                    stat_value = float(stat_value)
                                except ValueError:
                                    continue

                            if document.correct and stat_value > 0:
                                model_stats[stat] += 1

                    # Process table-level stats (only for tables above threshold)
                    if table in tables_above_threshold:
                        for stat in self.stats[table]:
                            if stat in needed_stats and stat in table_level_stats:
                                # Handle col_tag/row_tag size categories
                                if stat == "col_tag" or stat == "row_tag":
                                    tag_value = self.stats[table][stat]
                                    size_categories = {
                                        "col_tag": [
                                            "small_per_cols",
                                            "medium_per_cols",
                                            "large_per_cols",
                                        ],
                                        "row_tag": [
                                            "small_per_rows",
                                            "medium_per_rows",
                                            "large_per_rows",
                                        ],
                                    }

                                    if tag_value in size_categories[stat]:
                                        if (
                                            table not in counted_tables[tag_value]
                                            and document.correct
                                        ):
                                            model_stats[tag_value] += 1
                                            counted_tables[tag_value].add(table)
                                # Handle other table-level stats
                                else:
                                    stat_value = self.stats[table][stat]
                                    if isinstance(stat_value, str):
                                        try:
                                            stat_value = float(stat_value)
                                        except ValueError:
                                            continue

                                    if (
                                        stat
                                        in [
                                            "acro_added",
                                            "typos_added",
                                            "alias_added",
                                        ]
                                        and document.correct
                                    ):
                                        if table not in acro_typo_alias:
                                            acro_typo_alias[table] = {}
                                        if stat not in acro_typo_alias[table]:
                                            acro_typo_alias[table][stat] = 1
                                        else:
                                            acro_typo_alias[table][stat] += 1
                                        if acro_typo_alias[table][stat] < stat_value:
                                            model_stats[stat] += 1
                                            counted_tables[stat].add(table)
                                    else:
                                        if (
                                            document.correct
                                            and stat_value > 0
                                            and table not in counted_tables[stat]
                                        ):
                                            model_stats[stat] += 1
                                            counted_tables[stat].add(table)
        final_stats = {
            key: (
                (
                    f"{math.trunc(model_stats[key] / model_stats_total[key] * 100 * 10) / 10}%"
                )
                if model_stats_total[key] != 0
                else None
            )
            for key in model_stats
        }

        final_stats = {
            mapping_dict.get(key, key): value for key, value in final_stats.items()
        }

        model_stats = {
            mapping_dict.get(key, key): value for key, value in model_stats.items()
        }

        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cuda": torch.cuda.get_device_name()
            if torch.cuda.is_available() and not test_locally
            else "CPU",
            "model_name": model_name,
            "ne_cells": self.TOTAL_CELLS,
            "total_time": self.truncate(total_time, 3),
            "accuracy": self.truncate(total_correct / self.TOTAL_CELLS, 3),
            "total_correct": total_correct,
            "stats": model_stats,
            "final_stats": final_stats,
        }

    def _update_table_level_stat(
        self, stat, table, is_correct, counted_tables, model_stats
    ):
        stat_value = self.stats[table][stat]

        # Convert string values to numeric if needed
        if isinstance(stat_value, str):
            try:
                stat_value = float(stat_value)
            except ValueError:
                return  # Skip non-numeric values

        # Only count each table once for this stat if document is correct
        if is_correct and stat_value > 0 and table not in counted_tables[stat]:
            model_stats[stat] += 1
            counted_tables[stat].add(table)
