from database import Cea, Database
import math
import torch
import json
import platform
import os

model_name = os.getenv("MODEL_NAME")


class Export:

    def __init__(self, db: Database):
        self.TOTAL_CELLS = 84907
        self.db = db
        self.stats = self.load_stats()

    def load_stats(self):
        stats_dict = {}
        with open("./general_stats_per_table.json", "r") as file:
            for data in json.load(file):
                stats_dict[data["table"]] = data["stats"]

        return stats_dict

    def compute_stats(self):
        all_documents: list[Cea] = self.db.get_all_documents()

        needed_stats = ["nils", "acro_added", "typos_added", "approx_added", "alias_added", "generic_types", "specific_types", "count_single_domain", "count_multi_domain"]
        cell_set = set()
        total_correct = 0
        total_computed = 0
        total_time = 0.0
        model_stats = {"nils": 0, "acro_added": 0,
                       "typos_added": 0, "approx_added": 0, "alias_added": 0, "generic_types": 0, "specific_types": 0, "count_single_domain": 0, "count_multi_domain": 0}
        model_stats_total = {"nils": 0, "acro_added": 0,
                             "typos_added": 0, "approx_added": 0, "alias_added": 0, "generic_types": 0, "specific_types": 0, "count_single_domain": 0, "count_multi_domain": 0}
        mapping_dict = {
            "nils": "nils", "acro_added": "acronyms",
            "typos_added": "typos", "approx_added": "approx", "alias_added": "alias", "generic_types": "generic_types", "specific_types": "specific_types", "count_single_domain": "single_domain", "count_multi_domain": "multi_domain"
        }

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
                    for stat in self.stats[table]:
                        if stat in needed_stats:
                            if document.correct:
                                if self.stats[table][stat] > 0:
                                    model_stats[stat] += 1
                            if self.stats[table][stat] > 0:
                                model_stats_total[stat] += 1
        
        final_stats = {key: ((f"{math.trunc(model_stats[key] / model_stats_total[key] * 100 * 10)/10}%")
                            if model_stats_total[key] != 0 else None) for key in model_stats}
        # Rename final_stats
        final_stats = {mapping_dict.get(key, key): value 
                        for key, value in final_stats.items()}

        # Rename model_stats
        model_stats = {mapping_dict.get(key, key): value 
                        for key, value in model_stats.items()}

        return {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cuda": torch.cuda.get_device_name(),
            "model_name": model_name,
            "total_cells": self.TOTAL_CELLS,
            "total_time": total_time,
            "accuracy": total_correct/self.TOTAL_CELLS,
            "stats": model_stats,
            "final_stats": final_stats
        }
