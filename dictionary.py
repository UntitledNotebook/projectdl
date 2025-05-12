import amulet
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from amulet.api.block import Block
from amulet.api.errors import ChunkLoadError, ChunkDoesNotExist
import logging
from tqdm import tqdm
from pathlib import Path
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_surface_height(chunk, x_offset, z_offset, max_y=319, min_y=-64):
    """
    Find the surface height for a given x,z offset in the chunk by scanning downward.
    Returns the y-coordinate of the first non-air block or min_y if none found.
    """
    for y in range(max_y, min_y - 1, -1):
        block = chunk.blocks[x_offset, y, z_offset]
        universal_block = chunk.block_palette[block]
        if universal_block.base_name != "air":
            return y
    return min_y

def process_chunk(cx, cz, level, block_snbt_set, lock, dimension="minecraft:overworld"):
    """
    Process a single chunk, adding SNBT blockstates to the shared set.
    """
    local_snbt_set = set()
    try:
        chunk = level.get_chunk(cx, cz, dimension)
        # Iterate over x,z offsets within the chunk (16x16)
        for x_offset in range(16):
            for z_offset in range(16):
                # Find surface height
                y_surface = find_surface_height(chunk, x_offset, z_offset)
                # Define 16x16x16 region (y_surface - 8 to y_surface + 7)
                y_start = y_surface - 8
                y_end = y_surface + 8  # Exclusive
                # Extract blocks in the 16x16x16 region
                for y in range(y_start, y_end):
                    if y < -64 or y > 319:  # Skip out-of-bounds y
                        continue
                    block_id = chunk.blocks[x_offset, y, z_offset]
                    universal_block = chunk.block_palette[block_id]
                    if isinstance(universal_block, Block):
                        snbt = universal_block.snbt_blockstate
                        local_snbt_set.add(snbt)
    except (ChunkLoadError, ChunkDoesNotExist) as e:
        logging.warning(f"Failed to load chunk ({cx}, {cz}): {e}")
    
    # Safely add to shared set
    with lock:
        block_snbt_set.update(local_snbt_set)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Collect unique SNBT blockstate strings from a Minecraft world.")
    parser.add_argument("world_folder", type=str, help="Path to the Minecraft world folder")
    parser.add_argument("chunk_radius", type=int, help="Chunk radius to process (non-negative integer)")
    args = parser.parse_args()

    world_folder = Path(args.world_folder)
    chunk_radius = args.chunk_radius

    if not world_folder.exists():
        logging.error(f"World folder {world_folder} does not exist.")
        return
    if chunk_radius < 0:
        logging.error("Chunk radius must be non-negative.")
        return

    # Load the world
    try:
        level = amulet.load_level(str(world_folder))
    except Exception as e:
        logging.error(f"Failed to load world: {e}")
        return

    # Generate chunk coordinates
    chunk_coords = [(cx, cz) for cx in range(-chunk_radius, chunk_radius + 1)
                    for cz in range(-chunk_radius, chunk_radius + 1)]
    total_chunks = len(chunk_coords)

    logging.info(f"Processing {total_chunks} chunks in radius {chunk_radius}...")

    # Initialize shared set and lock for thread safety
    block_snbt_set = set()
    lock = threading.Lock()

    # Process chunks in parallel
    with ThreadPoolExecutor() as executor:
        # Use tqdm for progress bar
        futures = [executor.submit(process_chunk, cx, cz, level, block_snbt_set, lock)
                   for cx, cz in chunk_coords]
        for _ in tqdm(futures, total=total_chunks, desc="Processing chunks"):
            _.result()  # Wait for all futures to complete

    # Close the world
    level.close()

    # Save the set to a JSON file
    output_file = world_folder / "block_snbt_vocabulary.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(sorted(list(block_snbt_set)), f, indent=2)
        logging.info(f"Saved {len(block_snbt_set)} unique SNBT strings to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save output: {e}")

if __name__ == "__main__":
    main()