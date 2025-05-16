import amulet
import numpy as np
import json
from pathlib import Path
from amulet.api import Block
import argparse
import logging

logging.basicConfig(level=logging.INFO)

argparser = argparse.ArgumentParser(description="Visualize output in minecraft.")
argparser.add_argument("--cx", type=int, default=0, help="Base X coordinate of the chunk")
argparser.add_argument("--y", type=int, default=0, help="Base Y coordinate of the region")
argparser.add_argument("--cz", type=int, default=0, help="Base Z coordinate of the chunk")
argparser.add_argument("--idx_to_snbt", type=str, default="idx_to_snbt.json", help="Path to the id to snbt map")
argparser.add_argument("--model_output", type=str, default="sampled_mapped_indices.npy", help="Path to the model output")
argparser.add_argument("--world_path", type=str, default="world", help="Path to the world")
args = argparser.parse_args()

MODEL_OUTPUT = Path(args.model_output)
WORLD_PATH = Path(args.world_path)

output = np.load(MODEL_OUTPUT)
logging.info(f"Output shape: {output.shape}")

idx_to_snbt = {int(k): v for k, v in json.load(open(args.idx_to_snbt)).items()}

level = amulet.load_level(WORLD_PATH)   
chunk = level.get_chunk(args.cx, args.cz, "minecraft:overworld")

for x_offset in range(16):
    for y_offset in range(16):
        for z_offset in range(16):
            block_id = int(output[x_offset, y_offset, z_offset])
            snbt = idx_to_snbt[block_id]
            chunk.set_block(x_offset, y_offset + args.y,
                            z_offset, Block.from_snbt_blockstate(snbt))

logging.info("Finished setting blocks")
chunk.changed = True
level.save()
level.close()
