from database import Cea, Database
import math
import torch
import json
import csv
import platform
import os

model_name = os.getenv("MODEL_NAME")
NIL_RESPONSE = "<NIL [DESCRIPTION] Not in list [TYPE] None>"


class Export:
    def __init__(self, db: Database):
        self.db = db
        self.stats = self.load_stats()
        self.stats_schema = self._detect_stats_schema()
        default_total_cells = 84185 if self.stats_schema == "legacy" else 84907
        self.TOTAL_CELLS = int(os.getenv("TOTAL_CELLS", default_total_cells))

    def load_stats(self):
        stats_dict = {}
        with open("./general_stats_per_table.json", "r") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                "general_stats_per_table.json must contain a list of table records"
            )

        for index, table_data in enumerate(data):
            if not isinstance(table_data, dict):
                raise ValueError(
                    "Invalid table record at index "
                    f"{index}: expected an object"
                )
            table = table_data.get("table")
            table_stats = table_data.get("stats")
            if not isinstance(table, str) or not isinstance(table_stats, dict):
                raise ValueError(
                    "Invalid table record at index "
                    f"{index}: expected string 'table' and object 'stats'"
                )
            if table in stats_dict:
                raise ValueError(
                    f"Duplicate table ID in general_stats_per_table.json: {table}"
                )
            stats_dict[table] = table_stats

        return stats_dict

    def _detect_stats_schema(self) -> str:
        """Identify the historical stats_needed schema by its available fields."""
        available_fields = {
            field
            for table_stats in self.stats.values()
            for field in table_stats
        }
        enriched_fields = {
            "col_tag",
            "row_tag",
            "table_generic_types",
            "table_specific_types",
        }
        return "enriched" if enriched_fields.issubset(available_fields) else "legacy"

    @staticmethod
    def _numeric_stat_value(value):
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def truncate(self, number, digits) -> float:
        # Improve accuracy with floating point operations, to avoid truncate(16.4, 2) = 16.39 or truncate(-1.13, 2) = -1.12
        number_string = str(number)
        if "." not in number_string:
            return number
        nbDecimals = len(number_string.split(".", 1)[1])
        if nbDecimals <= digits:
            return number
        stepper = 10.0**digits
        return math.trunc(stepper * number) / stepper

    def _validate_stats_for_documents(self, documents):
        """Refuse to export results when metadata and results use different tables."""
        result_tables = {document.table for document in documents}
        missing_tables = sorted(result_tables.difference(self.stats))
        extra_tables = sorted(set(self.stats).difference(result_tables))
        if missing_tables or extra_tables:
            examples = ", ".join(missing_tables[:5])
            if len(missing_tables) > 5:
                examples += ", ..."
            if extra_tables:
                extra_examples = ", ".join(extra_tables[:5])
                if len(extra_tables) > 5:
                    extra_examples += ", ..."
                if examples:
                    examples += "; "
                examples += f"statistics-only tables: {extra_examples}"
            raise ValueError(
                "Dataset/statistics mismatch: "
                f"{len(missing_tables)} table(s) missing from statistics and "
                f"{len(extra_tables)} table(s) missing from inference results "
                f"(examples: {examples}). Refusing to export incomplete "
                "challenge statistics. Use the metadata generated for the "
                "same mammotab_sample.jsonl release."
            )

    @staticmethod
    def _is_nil_response(response) -> bool:
        """Return True for the canonical NIL response, ignoring outer whitespace."""
        return isinstance(response, str) and response.strip() == NIL_RESPONSE

    @staticmethod
    def _format_percentage(numerator, denominator):
        if denominator == 0:
            return None
        return f"{math.trunc(numerator / denominator * 100 * 10) / 10}%"

    def _compute_legacy_stats(self, documents):
        """Compute metrics for the historical stats_needed schema.

        The legacy file has per-table counters but no table-size or table-type
        fields. Metrics are therefore calculated only for fields available in
        that file, with denominators based on the evaluated unique cells.
        """
        stats_keys = []
        seen_stats_keys = set()
        for table_stats in self.stats.values():
            for stat in table_stats:
                if stat not in seen_stats_keys and stat != "nils":
                    stats_keys.append(stat)
                    seen_stats_keys.add(stat)

        model_stats = {stat: 0 for stat in stats_keys}
        model_stats_total = {stat: 0 for stat in stats_keys}
        cell_set = set()
        total_correct = 0
        total_computed = 0
        total_time = 0.0
        correct_nils = 0
        total_nils = 0

        for document in documents:
            table = document.table
            cell_key = (table, document.row, document.column)
            if cell_key in cell_set:
                continue
            cell_set.add(cell_key)
            total_computed += 1

            if document.avg_time is not None:
                total_time += document.avg_time
            if document.correct:
                total_correct += 1

            if self._is_nil_response(document.correct_response):
                total_nils += 1
                if document.correct:
                    correct_nils += 1

            for stat, raw_value in self.stats[table].items():
                if stat == "nils":
                    continue
                stat_value = self._numeric_stat_value(raw_value)
                if stat_value is None or stat_value <= 0:
                    continue
                model_stats_total[stat] += 1
                if document.correct:
                    model_stats[stat] += 1

        final_stats = {
            stat: self._format_percentage(model_stats[stat], model_stats_total[stat])
            for stat in stats_keys
        }
        model_stats["nils"] = correct_nils
        final_stats["nils"] = self._format_percentage(correct_nils, total_nils)

        accuracy = (
            self.truncate(total_correct / self.TOTAL_CELLS, 3)
            if self.TOTAL_CELLS
            else None
        )
        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cuda": torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else "CPU",
            "model_name": model_name,
            "ne_cells": self.TOTAL_CELLS,
            "total_cells": self.TOTAL_CELLS,
            "too_long": max(self.TOTAL_CELLS - total_computed, 0),
            "total_time": self.truncate(total_time, 3),
            "accuracy": accuracy,
            "total_correct": total_correct,
            "stats": model_stats,
            "final_stats": final_stats,
        }

    def compute_stats(self):
        all_documents: list[Cea] = self.db.get_all_documents(model_name=model_name)
        self._validate_stats_for_documents(all_documents)

        if self.stats_schema == "legacy":
            return self._compute_legacy_stats(all_documents)

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
        nils = 0
        total_nils = 0
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

                    if self._is_nil_response(document.correct_response):
                        total_nils += 1
                        if document.correct:
                            nils += 1

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

        model_stats["nils"] = nils
        final_stats["nils"] = self._format_percentage(nils, total_nils)

        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cuda": torch.cuda.get_device_name()
            if torch.cuda.is_available()
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
