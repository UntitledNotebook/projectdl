import numpy as np
import pickle
import os

def transform_dataset_with_embeddings(original_dataset_path,
                                      embedding_pickle_path,
                                      output_dataset_path,
                                      embedding_size=None): # embedding_size can be inferred
    """
    Loads an original dataset (n, D, H, W) of block IDs and an embedding dictionary,
    then transforms the dataset into (n, embedding_size, D, H, W) where each
    block ID is replaced by its embedding vector.

    Args:
        original_dataset_path (str): Path to the original .npy dataset (e.g., "processed_block_ids.npy").
        embedding_pickle_path (str): Path to the .pkl file containing the
                                     block_id -> embedding_vector dictionary.
        output_dataset_path (str): Path to save the transformed .npy dataset.
        embedding_size (int, optional): The dimension of the embeddings. If None,
                                        it's inferred from the loaded embeddings.
    """
    # 1. Load original dataset
    print(f"Loading original dataset from: {original_dataset_path}")
    original_dataset = np.load(original_dataset_path)
    print(f"Original dataset shape: {original_dataset.shape}")

    if original_dataset.ndim != 4:
        raise ValueError(f"Original dataset must be 4D (n_samples, D, H, W), "
                         f"but got shape {original_dataset.shape}")

    # 2. Load embedding dictionary
    print(f"Loading embedding dictionary from: {embedding_pickle_path}")
    with open(embedding_pickle_path, "rb") as f:
        embedding_dict = pickle.load(f)
    print(f"Loaded embedding dictionary with {len(embedding_dict)} entries.")

    if not embedding_dict:
        raise ValueError("Embedding dictionary is empty!")

    # 3. Determine embedding size
    # Infer embedding_size from the first item if not provided or for validation
    first_key = next(iter(embedding_dict))
    inferred_embedding_size = embedding_dict[first_key].shape[0]

    if embedding_size is None:
        embedding_size = inferred_embedding_size
        print(f"Inferred embedding size: {embedding_size}")
    elif embedding_size != inferred_embedding_size:
        raise ValueError(f"Provided embedding_size {embedding_size} does not match "
                         f"inferred size {inferred_embedding_size} from embeddings dictionary.")
    else:
        print(f"Using provided embedding size: {embedding_size}")

    # Define a default embedding for block IDs not found in the dictionary (e.g., padding or unknown blocks)
    # It's important that the embedding_dict covers all relevant block IDs from your dataset.
    default_embedding = np.zeros(embedding_size, dtype=np.float32)

    # 4. Initialize the new dataset array
    n_samples = original_dataset.shape[0]
    D, H, W = original_dataset.shape[1], original_dataset.shape[2], original_dataset.shape[3]
    
    # New shape: (n_samples, embedding_size, D, H, W)
    transformed_dataset = np.zeros((n_samples, embedding_size, D, H, W), dtype=np.float32)
    print(f"Initialized transformed dataset with shape: {transformed_dataset.shape}")

    # 5. Iterate through the original dataset and replace block IDs with embeddings
    print("Starting transformation...")
    for i in range(n_samples):
        if (i + 1) % max(1, n_samples // 20) == 0 or i == n_samples - 1: # Print progress roughly 20 times
            print(f"Processing sample {i+1}/{n_samples}...")
        for d_idx in range(D):
            for h_idx in range(H):
                for w_idx in range(W):
                    block_id = original_dataset[i, d_idx, h_idx, w_idx]
                    # Ensure block_id is a Python int for dictionary lookup,
                    # as keys in the pickled dict are likely Python ints.
                    embedding_vector = embedding_dict.get(int(block_id), default_embedding)
                    
                    # Sanity check for embedding vector dimension
                    if embedding_vector.shape[0] != embedding_size:
                        print(f"Warning: Block ID {block_id} at sample {i}, pos ({d_idx},{h_idx},{w_idx}) "
                              f"yielded an embedding of shape {embedding_vector.shape}, "
                              f"but expected {embedding_size}. Using default embedding instead.")
                        embedding_vector = default_embedding
                        
                    transformed_dataset[i, :, d_idx, h_idx, w_idx] = embedding_vector
    
    # 6. Save the new dataset
    output_dir = os.path.dirname(output_dataset_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Saving transformed dataset to: {output_dataset_path}")
    np.save(output_dataset_path, transformed_dataset)
    print("Transformation complete and dataset saved.")

if __name__ == '__main__':
    # --- Example Usage ---
    # Replace these paths with your actual file paths

    # Path to your original dataset (e.g., the output of transform.py)
    # This should be an (N, 16, 16, 16) array of block IDs.
    ORIGINAL_DATASET_NPY = "processed_block_ids_transformed.npy" # Or your original data file

    # Path to the pickled dictionary mapping block IDs to embeddings
    # This is "index_embeddings.pkl" from your block2vec.py output_path
    EMBEDDING_PICKLE = "output/block2vec/index_embeddings.pkl"

    # Path where the new transformed dataset will be saved
    OUTPUT_TRANSFORMED_DATASET_NPY = "output/embedded_dataset/transformed_embedded_dataset.npy"

    # Desired embedding size (e.g., 32). Can be None to infer.
    # If your save_embedding function is correct, this should match the dimension of vectors in the pickle.
    EMBEDDING_DIM = 32 

    # --- Create dummy files for testing if real files don't exist ---
    if not os.path.exists(ORIGINAL_DATASET_NPY):
        print(f"Creating dummy original dataset: {ORIGINAL_DATASET_NPY}")
        dummy_data = np.random.randint(0, 5, size=(10, 16, 16, 16)) # 10 samples, 5 block types
        np.save(ORIGINAL_DATASET_NPY, dummy_data)

    if not os.path.exists(os.path.dirname(EMBEDDING_PICKLE)):
        os.makedirs(os.path.dirname(EMBEDDING_PICKLE), exist_ok=True)

    if not os.path.exists(EMBEDDING_PICKLE):
        print(f"Creating dummy embedding pickle: {EMBEDDING_PICKLE}")
        dummy_embeddings_dict = {
            i: np.random.rand(EMBEDDING_DIM).astype(np.float32) for i in range(5) # Embeddings for block types 0-4
        }
        # Add a block ID that might be in dummy_data but not in dict to test default
        dummy_embeddings_dict[0] = np.random.rand(EMBEDDING_DIM).astype(np.float32) # ensure 0 is there
        with open(EMBEDDING_PICKLE, "wb") as f:
            pickle.dump(dummy_embeddings_dict, f)
    # --- End of dummy file creation ---

    print("\n--- Starting Transformation Script ---")
    transform_dataset_with_embeddings(
        original_dataset_path=ORIGINAL_DATASET_NPY,
        embedding_pickle_path=EMBEDDING_PICKLE,
        output_dataset_path=OUTPUT_TRANSFORMED_DATASET_NPY,
        embedding_size=EMBEDDING_DIM
    )

    # Optional: Load and verify the output
    if os.path.exists(OUTPUT_TRANSFORMED_DATASET_NPY):
        print("\n--- Verifying Output ---")
        loaded_transformed_data = np.load(OUTPUT_TRANSFORMED_DATASET_NPY)
        print(f"Shape of loaded transformed data: {loaded_transformed_data.shape}")
        # Expected shape: (N, EMBEDDING_DIM, 16, 16, 16)
        # For dummy data: (10, 32, 16, 16, 16)
        
        # You can add more specific checks here, e.g.,
        # original_data_check = np.load(ORIGINAL_DATASET_NPY)
        # embeddings_dict_check = pickle.load(open(EMBEDDING_PICKLE, "rb"))
        # sample_idx, d,h,w = 0,0,0,0
        # original_id = original_data_check[sample_idx,d,h,w]
        # expected_vec = embeddings_dict_check.get(int(original_id), np.zeros(EMBEDDING_DIM, dtype=np.float32))
        # actual_vec = loaded_transformed_data[sample_idx, :, d,h,w]
        # if np.allclose(expected_vec, actual_vec):
        #     print("Verification of one voxel passed.")
        # else:
        #     print("Verification of one voxel FAILED.")