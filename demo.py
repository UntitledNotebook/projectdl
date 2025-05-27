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
argparser.add_argument("--true_samples", type=str, default="true_samples.npy", help="Path to the true samples")
argparser.add_argument("--inpaint_samples", type=str, default="sampled_mapped_indices.npy", help="Path to the model output")
argparser.add_argument("--world_path", type=str, default="world", help="Path to the world")
args = argparser.parse_args()

true_samples = np.load(args.true_samples)
inpaint_samples = np.load(args.inpaint_samples)
x_len, y_len, z_len = true_samples.shape
cx_len = x_len // 16
cz_len = z_len // 16

idx_to_snbt = {int(k): v for k, v in json.load(open(args.idx_to_snbt)).items()}

level = amulet.load_level(args.world_path)

for cx in range(args.cx, args.cx + cx_len):
    for cz in range(args.cz, args.cz + cz_len):
        chunk = level.get_chunk(cx, cz, "minecraft:overworld")
        for x_offset in range(16):
            for y_offset in range(y_len):
                for z_offset in range(16):
                    block_id = int(true_samples[(cx - args.cx) * 16 + x_offset, y_offset, (cz - args.cz) * 16 + z_offset])
                    snbt = idx_to_snbt[block_id]
                    chunk.set_block(x_offset, y_offset + args.y,
                                    z_offset, Block.from_snbt_blockstate(snbt))
logging.info("Finished setting blocks")
chunk.changed = True
level.save()
level.close()
