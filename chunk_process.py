import amulet
import argparse
import logging
import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Initialize the argument parser instance.
arg_parser = argparse.ArgumentParser(description="Process Minecraft world data to extract 16x16x16 block ID samples near the surface.")
arg_parser.add_argument("--world-folder", type=str, required=True, help="Path to the Minecraft world folder")
arg_parser.add_argument("--output-dir", type=str, required=True, help="Directory to save processed data")
arg_parser.add_argument("--chunk-radius", type=int, default=5, help="Chunk radius around world center (0,0) to process (default: 5)")
arg_parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4, help=f"Number of worker threads (default: {os.cpu_count() or 4})")

def collect_chunk_metadata_and_snbt(world_obj, cx, cz):
    """
    Determines the surface Y level for a chunk and collects unique SNBT strings
    from the 16x16x16 region at that surface.
    Returns (cx, cz, set_of_snbt_strings, base_y).
    """
    min_y_dim, max_y_dim = 0, 128
    local_unique_snbt = set()

    try:
        chunk = world_obj.get_chunk(cx, cz, "minecraft:overworld")
        
        # Determine surface Y at chunk's relative center (block x=8, z=8 within chunk 0-15 range)
        temp_surface_y = min_y_dim - 1
        for y_coord in range(max_y_dim - 1, min_y_dim - 1, -1):
            block = chunk.get_block(8, y_coord, 8) # x,z are chunk-relative
            if block.base_name != "air": 
                temp_surface_y = y_coord
                break
        
        if temp_surface_y < min_y_dim: # No valid surface point found for this chunk
            logging.debug(f"No valid surface point found for chunk ({cx},{cz}).")
            return cx, cz, set(), temp_surface_y # temp_surface_y is the sentinel

        base_y = temp_surface_y - 7

        # Collect SNBTs from the 16x16x16 region defined by base_y
        for dx_in_chunk in range(16):
            for dz_in_chunk in range(16):
                for dy_offset in range(16): # dy_offset from 0 to 15
                    world_y = base_y + dy_offset
                    if not (min_y_dim <= world_y < max_y_dim):
                        continue # Skip Y-levels outside the valid dimension range
                    
                    block_in_region = chunk.get_block(dx_in_chunk, world_y, dz_in_chunk)
                    local_unique_snbt.add(block_in_region.snbt_blockstate)
                    
    except amulet.api.errors.ChunkDoesNotExist:
        logging.warning(f"Chunk ({cx},{cz}) does not exist.")
        # surface_y_at_center remains the initial sentinel value
    except Exception as e:
        logging.error(f"Error in collect_chunk_metadata_and_snbt for chunk ({cx},{cz}): {e}", exc_info=True)
        # surface_y_at_center remains the initial sentinel value

    return cx, cz, local_unique_snbt, base_y

def extract_block_id_array(world_obj, cx, cz, base_y, 
                           snbt_to_id_map, air_snbt_id):
    """
    Extracts a 16x16x16 region of block IDs from a chunk based on SNBT strings.
    Returns only the NumPy array.
    """
    block_id_array = np.full((16, 16, 16), air_snbt_id, dtype=np.uint32)
    
    try:
        chunk = world_obj.get_chunk(cx, cz, "minecraft:overworld")
        for dx_in_chunk in range(16):
            for dz_in_chunk in range(16):
                for dy_offset in range(16):
                    current_world_y = base_y + dy_offset
                    block = chunk.get_block(dx_in_chunk, current_world_y, dz_in_chunk)
                    snbt_string = block.snbt_blockstate
                    block_id_array[dx_in_chunk, dy_offset, dz_in_chunk] = snbt_to_id_map.get(snbt_string, air_snbt_id)
    
    except amulet.api.errors.ChunkDoesNotExist:
        logging.warning(f"Chunk ({cx},{cz}) vanished before ID array extraction. Resulting array is air-filled.")
    except Exception as e:
        logging.error(f"Error extracting block ID array for chunk ({cx},{cz}): {e}", exc_info=True)
                
    return block_id_array

def main():
    args = arg_parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Loading world from {args.world_folder}...")
    try:
        world = amulet.load_level(args.world_folder)
    except Exception as e:
        logging.error(f"Failed to load world: {e}", exc_info=True)
        return
    
    logging.info(f"Processing Overworld chunks around (0,0) with radius {args.chunk_radius}.")

    target_chunks_coords = [(cx, cz) for cx in range(-args.chunk_radius, args.chunk_radius + 1)
                            for cz in range(-args.chunk_radius, args.chunk_radius + 1)]
    all_unique_snbt = set()
    chunk_processing_params = [] 
    logging.info("Phase 1: Collecting SNBT palette and surface Y for specified regions...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(collect_chunk_metadata_and_snbt, world, cx, cz)
                   for cx, cz in target_chunks_coords]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 1"):
            try:
                r_cx, r_cz, local_snbt, base_y = future.result()
                all_unique_snbt.update(local_snbt)
                chunk_processing_params.append({'cx': r_cx, 'cz': r_cz, 'base_y': base_y})
            except Exception as e:
                logging.error(f"Critical error retrieving result from Phase 1 task: {e}", exc_info=True)
    
    if not chunk_processing_params:
        logging.info("Phase 1 did not identify any processable chunk regions. Exiting.")
        world.close()
        return
    logging.info(f"Phase 1 completed. Found {len(all_unique_snbt)} unique SNBT blockstates from {len(chunk_processing_params)} chunk regions.")

    air_snbt_str = amulet.Block("universal_minecraft", "air").snbt_blockstate
    all_unique_snbt.discard(air_snbt_str) 
    sorted_other_snbt = sorted(list(all_unique_snbt))
    
    snbt_to_id_map = {air_snbt_str: 0}
    snbt_to_id_map.update({snbt: i + 1 for i, snbt in enumerate(sorted_other_snbt)})
    air_snbt_id_val = 0 

    id_to_snbt_map = {i: snbt for snbt, i in snbt_to_id_map.items()}
    palette_map_path = os.path.join(args.output_dir, "snbt_palette_map.json")
    with open(palette_map_path, 'w') as f:
        json.dump(id_to_snbt_map, f, indent=2)
    logging.info(f"Saved SNBT palette map to {palette_map_path}")

    all_block_id_arrays = []
    logging.info(f"Phase 2: Extracting {len(chunk_processing_params)} block ID arrays...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for params in chunk_processing_params:
            cx, cz, base_y = params['cx'], params['cz'], params['base_y']
            futures.append(executor.submit(extract_block_id_array, world, cx, cz,
                                           base_y, snbt_to_id_map, air_snbt_id_val))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 2"):
            try:
                block_id_array = future.result()
                all_block_id_arrays.append(block_id_array)
            except Exception as e:
                logging.error(f"Critical error retrieving block ID array from Phase 2 task: {e}", exc_info=True)

    logging.info(f"Phase 2 completed. Successfully generated {len(all_block_id_arrays)} block ID arrays.")
    world.close()
    logging.info("Minecraft world closed.")

    if all_block_id_arrays:
        try:
            final_npy_array = np.stack(all_block_id_arrays, axis=0)
            output_npy_path = os.path.join(args.output_dir, "processed_block_ids.npy")
            np.save(output_npy_path, final_npy_array)
            logging.info(f"Saved {final_npy_array.shape[0]} samples (shape: {final_npy_array.shape}) to {output_npy_path}")
        except Exception as e:
            logging.error(f"Failed to stack or save the final NumPy array: {e}", exc_info=True)
    else:
        logging.info("No block ID arrays were generated to save.")
        
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()