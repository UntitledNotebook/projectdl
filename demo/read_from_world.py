import amulet
import argparse
import numpy as np
from pathlib import Path
import json

argparser = argparse.ArgumentParser(description="Read blocks from a Minecraft world.")
argparser.add_argument("--cx", type=int, default=0, help="Base X coordinate of the chunk")
argparser.add_argument("--y", type=int, default=0, help="Base Y coordinate of the region")
argparser.add_argument("--cz", type=int, default=0, help="Base Z coordinate of the chunk")
argparser.add_argument("--world_path", type=str, default="world", help="Path to the world")
argparser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
args = argparser.parse_args()

level = amulet.load_level(args.world_path)

DATA_PATH = Path(args.data_path)
shape = np.load(DATA_PATH / "block_id.npy")[0].shape
snbt_to_id = json.load(open(DATA_PATH / "snbt_to_id.json"))

raw = np.zeros(shape, dtype=np.int32)

len_x, len_y, len_z = shape
cx_len = len_x // 16
cz_len = len_z // 16

for cx in range(args.cx, args.cx + cx_len):
    for cz in range(args.cz, args.cz + cz_len):
        chunk = level.get_chunk(cx, cz, "minecraft:overworld")
        for x_offset in range(16):
            for y_offset in range(len_y):
                for z_offset in range(16):
                    block = chunk.get_block(x_offset, y_offset + args.y, z_offset)
                    snbt = block.snbt_blockstate
                    block_id = snbt_to_id.get(snbt, 0)  # Default to 0 if not found
                    raw[(cx - args.cx) * 16 + x_offset, y_offset, (cz - args.cz) * 16 + z_offset] = block_id

level.close()
np.save(DATA_PATH / "raw.npy", raw)