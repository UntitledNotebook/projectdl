import os
import json
import numpy as np
from pathlib import Path
import amulet

combined_palette = set()
combined_counts = {}

for subfolder in Path("data").iterdir():
    if subfolder.is_dir():
        palette_file = subfolder / "snbt_palette_map.json"
        counts_file = subfolder / "snbt_counts.json"

        if palette_file.exists():
            with open(palette_file, 'r') as f:
                palette_data = json.load(f)
                combined_palette.update(palette_data.values())

        if counts_file.exists():
            with open(counts_file, 'r') as f:
                counts_data = json.load(f)
                for key, value in counts_data.items():
                    combined_counts[key] = combined_counts.get(key, 0) + value


air_snbt_str = amulet.Block("universal_minecraft", "air").snbt_blockstate
combined_palette.discard(air_snbt_str)
sorted_other_snbt = sorted(list(combined_palette))
snbt_to_id_map = {air_snbt_str: 0}
snbt_to_id_map.update({snbt: i + 1 for i, snbt in enumerate(sorted_other_snbt)})
id_to_snbt_map = {i: snbt for snbt, i in snbt_to_id_map.items()}

dataset = []

for subfolder in Path("data").iterdir():
    if subfolder.is_dir():
        npy_file = subfolder / "processed_block_ids.npy"
        sorted_other_snbt = {}
        with open(subfolder / "snbt_palette_map.json", 'r') as f:
            sorted_other_snbt = json.load(f)
        if npy_file.exists():
            block_id_array = np.load(npy_file)
            id_remap = {i: snbt_to_id_map.get(snbt, 0) for i, snbt in enumerate(sorted_other_snbt)}
            remapped_array = np.vectorize(lambda x: id_remap.get(x, 0))(block_id_array)
            dataset.append(remapped_array)

with open(Path("data") / "snbt_palette_map.json", 'w') as f:
    json.dump(id_to_snbt_map, f, indent=2, sort_keys=True)
with open(Path("data") / "snbt_counts.json", 'w') as f:
    json.dump(combined_counts, f, indent=2, sort_keys=True)
np.save(Path("data") / "processed_block_ids.npy", np.concatenate(dataset, axis=0))
