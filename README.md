# 3D Diffusion Model for Minecraft Block Generation

This project presents a voxel-based procedural content generation model for Minecraft, leveraging a conditional Denoising Diffusion Probabilistic Model (DDPM). The model generates 32×32×32 block structures using a 3D U-Net architecture with self-conditioning to ensure temporal consistency and contextual coherence.

## Key Innovations and Features

*   **Adaptive Block Representation**:
    *   **Block2vec Embeddings**: Used for encoding smooth terrains (e.g., grasslands). See implementation in [`model/by_block2vec/block2vec/`](model/by_block2vec/block2vec/).
    *   **Analog Bits Representation**: Handles discrete, high-variance structures (e.g., trees, buildings) at a bit-level. This is integral to the diffusion process in [`model/ddpm/diffusion.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\diffusion.py).
*   **Conditional Generation**:
    *   **Biome-Aware**: Supports generation tailored to specific biomes, configurable via files like [`config_Forest.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\config_Forest.py) and [`config_Village.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\config_Village.py).
    *   **Inpainting**: Enables natural integration of generated chunks into diverse landscapes. See [`model/ddpm/inpaint.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\inpaint.py).
*   **Advanced Model Architecture & Training**:
    *   **3D U-Net**: The core architecture for generation, defined in [`model/ddpm/model.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\model.py).
    *   **Self-Conditioning**: Enhances temporal consistency and contextual coherence, configured in the diffusion process.
    *   **Cosine-Based Noise Scheduling**: Incorporated for improved denoising performance.
    *   **Spatiotemporal Attention**: Utilized within the U-Net for enhanced detail and structure.

## Codebase Structure

The repository includes the following key components:

*   **Data Preprocessing**:
    *   [`chunk_generation.py`](d:\homework\2025Spr\DL\project\projectdl\chunk_generation.py): Script for automating the generation of Minecraft worlds.
    *   [`chunk_process.py`](d:\homework\2025Spr\DL\project\projectdl\chunk_process.py): Script for processing generated chunks and extracting data for training.
*   **Representation Learning (Block2vec)**:
    *   Located under [`model/by_block2vec/block2vec/`](model/by_block2vec/block2vec/).
    *   [`block2vec.py`](d:\homework\2025Spr\DL\project\projectdl\model\by_block2vec\block2vec\block2vec.py): Defines the CustomBlock2Vec Lightning module.
    *   [`skipgram.py`](d:\homework\2025Spr\DL\project\projectdl\model\by_block2vec\block2vec\skipgram.py): Implements the SkipGram model.
    *   [`train.py`](d:\homework\2025Spr\DL\project\projectdl\model\by_block2vec\block2vec\train.py): Training script for Block2vec embeddings.
*   **DDPM (Denoising Diffusion Probabilistic Model)**:
    *   Located under [`model/ddpm/`](model/ddpm/).
    *   [`model.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\model.py): Contains the 3D U-Net definition ([`UNet3D`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\model.py)) and time embeddings ([`SinusoidalTimeEmbedding`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\model.py)).
    *   [`diffusion.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\diffusion.py): Implements the BitDiffusion process.
    *   [`train.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\train.py): Main training script for the diffusion model.
    *   [`data.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\data.py): Data loading and preprocessing for DDPM training.
    *   Configuration files: [`config.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\config.py), [`config_Forest.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\config_Forest.py), [`config_Village.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\config_Village.py).
*   **Generation and Utility Scripts**:
    *   [`model/ddpm/inpaint.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\inpaint.py): Script for performing inpainting using a trained model.
    *   [`demo/`](demo/): Contains scripts for interacting with Minecraft worlds.
        *   [`write_to_world.py`](d:\homework\2025Spr\DL\project\projectdl\demo\write_to_world.py): Writes generated structures to a Minecraft world.
        *   [`read_from_world.py`](d:\homework\2025Spr\DL\project\projectdl\demo\read_from_world.py): Reads data from a Minecraft world.
    *   [`model/ddpm/utils.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\utils.py): Utility functions, including logging samples.

## Setup and Usage

1.  **Prerequisites**:
    *   Java (e.g., OpenJDK 21)
    *   Python and relevant libraries (PyTorch, Accelerate, Diffusers, Amulet-Core, etc.)
    *   Minecraft server (PaperMC is used in [`chunk_generation.py`](d:\homework\2025Spr\DL\project\projectdl\chunk_generation.py))
    *   Chunky plugin (downloaded by [`chunk_generation.py`](d:\homework\2025Spr\DL\project\projectdl\chunk_generation.py))

2.  **Data Preparation**:
    *   Run [`chunk_generation.py`](d:\homework\2025Spr\DL\project\projectdl\chunk_generation.py) to generate raw Minecraft world data.
        ```bash
        python chunk_generation.py --memory 6G --chunk-radius 16
        ```
    *   Process the generated world data using [`chunk_process.py`](d:\homework\2025Spr\DL\project\projectdl\chunk_process.py) to create training samples.

3.  **Representation Learning (Optional but Recommended for Block2vec)**:
    *   Train Block2vec embeddings:
        ```bash
        python model/by_block2vec/block2vec/train.py
        ```

4.  **Train Diffusion Model**:
    *   Configure your training settings in one of the `config_*.py` files (e.g., [`model/ddpm/config_Forest.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\config_Forest.py)).
    *   Run the training script:
        ```bash
        accelerate launch model/ddpm/train.py
        ```
        (Ensure your `accelerate` config is set up if not using default, or modify `train.py` if it uses a specific config module like `TrainConfig` directly).

5.  **Generation/Inpainting**:
    *   Use [`model/ddpm/inpaint.py`](d:\homework\2025Spr\DL\project\projectdl\model\ddpm\inpaint.py) for generating structures or performing inpainting, loading a trained model checkpoint.
    *   Use scripts in [`demo/`](demo/) to integrate generated data with Minecraft worlds.

Refer to individual script arguments (`--help`) and configuration files for more detailed options.
The [`setup.sh`](setup.sh) script might assist with initial environment setup.
Server properties can be configured via [`server.properties`](server.properties).