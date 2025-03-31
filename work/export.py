from database import Cea, Database
import math
import torch
import json
import platform
import os

model_name = os.getenv("MODEL_NAME")


class Export:

    def __init__(self, db: Database):
        self.TOTAL_CELLS = 84185
        self.db = db
        self.stats = self.load_stats()

    def load_stats(self):
        stats_dict = {}
        with open("./stats_needed.json", "r") as file:
            for data in json.load(file):
                stats_dict[data["table"]] = data["stats"]

        return stats_dict

    def compute_stats(self):
        all_documents: list[Cea] = self.db.get_all_documents()

        cell_set = set()
        total_correct = 0
        total_computed = 0
        total_time = 0
        model_stats = {"tot_linked_cell": 0, "entities_found": 0, "entities_not_found": 0, "types_found": 0, "types_not_found": 0, "tot_cells": 0, "nils": 0, "count_with_header": 0, "count_with_caption": 0, "acro_added": 0,
                       "typos_added": 0, "approx_added": 0, "alias_added": 0, "generic_types": 0, "specific_types": 0, "filtered_types": 0, "found_perfect_types": 0, "tot_cols_with_types": 0, "count_single_domain": 0, "count_multi_domain": 0}
        model_stats_total = {"tot_linked_cell": 0, "entities_found": 0, "entities_not_found": 0, "types_found": 0, "types_not_found": 0, "tot_cells": 0, "nils": 0, "count_with_header": 0, "count_with_caption": 0, "acro_added": 0,
                             "typos_added": 0, "approx_added": 0, "alias_added": 0, "generic_types": 0, "specific_types": 0, "filtered_types": 0, "found_perfect_types": 0, "tot_cols_with_types": 0, "count_single_domain": 0, "count_multi_domain": 0}

        for document in all_documents:
            table = document.table
            row = document.row
            column = document.column
            if f"{table}_{row}_{column}" not in cell_set:
                total_computed += 1
                total_time += document.avg_time
                cell_set.add(f"{table}_{row}_{column}")
                if document.correct:
                    total_correct += 1
                for stat in self.stats[table]:
                    if document.correct:
                        if self.stats[table][stat] > 0:
                            model_stats[stat] += 1
                    if self.stats[table][stat] > 0:
                        model_stats_total[stat] += 1
        final_stats = {key: ((f"{math.trunc(model_stats[key] / model_stats_total[key] * 100 * 10)/10}%")
                             if model_stats_total[key] != 0 else None) for key in model_stats}

        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cuda": torch.cuda.get_device_name(),
            "model_name": model_name,
            "total_cells": self.TOTAL_CELLS,
            "total_time": total_time,
            "accuracy": total_correct/self.TOTAL_CELLS,
            "too_long": self.TOTAL_CELLS - total_computed,
            "stats": model_stats,
            "final_stats": final_stats
        }
