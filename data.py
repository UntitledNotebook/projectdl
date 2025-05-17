import os
import json
import numpy as np
from pathlib import Path
import amulet
from collections import defaultdict

USE_PRE = False

if USE_PRE:
    id_to_snbt = json.load(open("data/id_to_snbt.json", 'r'))
    id_to_snbt = {int(k): v for k, v in id_to_snbt.items()}
    snbt_to_id = {v: int(k) for k, v in id_to_snbt.items()}
else:
    combined_counts = {}

    for subfolder in Path("data").iterdir():
        if subfolder.is_dir():
            counts_file = subfolder / "snbt_counts.json"
            if counts_file.exists():
                with open(counts_file, 'r') as f:
                    counts_data = json.load(f)
                    for key, value in counts_data.items():
                        combined_counts[key] = combined_counts.get(key, 0) + value

    with open(Path("data") / "snbt_counts.json", 'w') as f:
        json.dump(combined_counts, f, indent=2, sort_keys=True)

    THRESHOLD = int(input("Enter the threshold for combined counts: "))

    combined_counts = {k: v for k, v in combined_counts.items() if v > THRESHOLD}
    air_snbt_str = amulet.Block("universal_minecraft", "air").snbt_blockstate
    combined_palette = set(combined_counts.keys())
    combined_palette.discard(air_snbt_str)
    sorted_other_snbt = sorted(list(combined_palette))
    snbt_to_id = {air_snbt_str: 0}
    snbt_to_id.update({snbt: i + 1 for i, snbt in enumerate(sorted_other_snbt)})
    id_to_snbt = {i: snbt for snbt, i in snbt_to_id.items()}

dataset = defaultdict(list)

import multiprocessing

def process_subfolder(subfolder_path):
    npy_file = subfolder_path / "block_ids.npy"
    metadata_file = subfolder_path / "meta.json"
    id_to_snbt_file = subfolder_path / "id_to_snbt.json"
    
    if not (npy_file.exists() and metadata_file.exists() and id_to_snbt_file.exists()):
        return None

    with open(id_to_snbt_file, 'r') as f:
        subfolder_id_to_snbt = json.load(f)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        length = metadata.get("region_chunk_radius", 0) * 16
        height = metadata.get("region_height", 0)
    
    block_id_array = np.load(npy_file)
    legal_local_ids = set([int(i) for i in subfolder_id_to_snbt.keys()\
                           if subfolder_id_to_snbt[i] in snbt_to_id.keys()])
    legal_id_remap = {i: snbt_to_id[subfolder_id_to_snbt[str(i)]] for i in legal_local_ids}
    remapped_array = []
    for i in range(block_id_array.shape[0]):
        if np.all(np.isin(block_id_array[i], list(legal_local_ids))):
            remapped_array.append(np.vectorize(lambda x: legal_id_remap.get(x, 0), otypes=[np.int32])(block_id_array[i]))
    remapped_array = np.stack(remapped_array, axis=0)
    counts = np.bincount(remapped_array.flatten(), minlength=len(snbt_to_id))
    return (length, height), remapped_array, counts

if __name__ == '__main__':
    subfolders = [subfolder for subfolder in Path("data").iterdir() if subfolder.is_dir()]

    with multiprocessing.Pool(processes=6) as pool:
        results = pool.map(process_subfolder, subfolders)
    combined_counts = np.zeros(len(snbt_to_id), dtype=np.int32)
    for result in results:
        if result:
            (length, height), remapped_array, counts = result
            dataset[(length, height)].append(remapped_array)
            combined_counts += counts
    combined_counts = {id_to_snbt[i]: int(count) for i, count in enumerate(combined_counts)}
    with open(Path("data") / "id_to_snbt.json", 'w') as f:
        json.dump(id_to_snbt, f, indent=2, sort_keys=True)
    with open(Path("data") / "snbt_counts.json", 'w') as f:
        json.dump(combined_counts, f, indent=2, sort_keys=True)

    dataset_info = {}

    for (length, height), array_list in dataset.items():
        combined_array = np.concatenate(array_list, axis=0)
        np.save(Path("data") / f"block_ids_{length}_{height}.npy", combined_array)
        dataset_info[str((length, height, length))] = {
            "shape": combined_array.shape,
            "dtype": str(combined_array.dtype),
            "size": combined_array.nbytes
        }

    with open(Path("data") / "dataset_info.json", 'w') as f:
        json.dump(dataset_info, f, indent=2, sort_keys=True)