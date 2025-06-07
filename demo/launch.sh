#!/bin/bash

# Bash Script for Minecraft World Inpainting Automation

# --- Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error.
set -u
# Ensure that pipelines fail on the first command that fails.
set -o pipefail

# Remote server details
REMOTE_USER_HOST="autodl-716" # Your SSH alias or user@host for the remote server
REMOTE_PROJECT_DIR="/root/autodl-tmp/projectdl" # Base directory for the project on the remote server

# Python executables for local scripts (if not in PATH, specify full path)
# Ensure these python scripts are in your PATH or in the directory where this bash script is run.
# If they are in the current directory, you might call them as ./read_from_world.py if they have a shebang and are executable.
LOCAL_PYTHON_EXEC="/home/f3f3x0/miniconda3/envs/amulet/bin/python" # Or python3, /usr/bin/python3 etc.
READ_SCRIPT_NAME="read_from_world.py"
WRITE_SCRIPT_NAME="write_to_world.py"
# For inpaint.py, it's assumed to be on the remote server at REMOTE_INPAINT_SCRIPT_PATH

# --- Argument Parsing ---
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <world_type> <cx> <y> <cz>"
    echo "  <world_type>: The type of the world (e.g., Forest, Village)."
    echo "  <cx>:         Base X coordinate of the chunk."
    echo "  <y>:          Base Y coordinate of the region."
    echo "  <cz>:         Base Z coordinate of the chunk."
    echo ""
    echo "Example: $0 Forest 0 64 0"
    exit 1
fi

wt="$1"    # World Type
cx_arg="$2" # Chunk X
y_arg="$3"  # Y-level
cz_arg="$4" # Chunk Z

echo "--- Script Started ---"
echo "World Type: ${wt}"
echo "Coordinates: cx=${cx_arg}, y=${y_arg}, cz=${cz_arg}"

# --- Path Definitions ---
# Local Minecraft paths
minecraft_saves_dir="${HOME}/.minecraft/saves"
world_name_template_suffix="_" # The suffix for the template world
world_name_template="${wt}${world_name_template_suffix}" # e.g., Forest_
world_name_active="${wt}"                              # e.g., Forest (this is the one used by scripts)

source_world_path="${minecraft_saves_dir}/${world_name_template}"
active_world_path="${minecraft_saves_dir}/${world_name_active}" # Path used by read_from_world.py and write_to_world.py

# Local data paths
# IMPORTANT: Ensure this base path is correct and the drive is mounted.
local_data_base_path="/media/f3f3x0/UBUNTU 24_0"
local_data_dir="${local_data_base_path}/${wt}" # e.g., /media/f3f3x0/UBUNTU 24_0/Forest

local_raw_npy_path="${local_data_dir}/raw.npy"
local_inpaint_npy_path="${local_data_dir}/inpaint.npy"
# Files required by read_from_world.py inside local_data_dir
local_block_id_npy_path="${local_data_dir}/block_id.npy"
local_snbt_to_id_json_path="${local_data_dir}/snbt_to_id.json"
# File required by write_to_world.py inside local_data_dir
local_id_to_snbt_json_path="${local_data_dir}/id_to_snbt.json"


# Remote paths
remote_raw_npy_path="${REMOTE_PROJECT_DIR}/raw.npy"
remote_inpaint_npy_path="${REMOTE_PROJECT_DIR}/inpaint.npy"
remote_model_base_dir="${REMOTE_PROJECT_DIR}/model/ddpm"
remote_checkpoint_path="${remote_model_base_dir}/outputs/${wt}/final_ema_model.pt"
remote_inpaint_script_path="${remote_model_base_dir}/inpaint.py"
remote_config_template_path="${remote_model_base_dir}/config_${wt}.py" # e.g. config_Forest.py
remote_active_config_path="${remote_model_base_dir}/config.py"     # Target config file for inpaint.py

# --- Helper Functions ---
check_local_file_exists() {
    local file_path="$1"
    local file_description="$2"
    if [ ! -f "${file_path}" ]; then
        echo "Error: Required local file '${file_description}' not found at ${file_path}."
        exit 1
    fi
}

# --- Main Script Logic ---

# Step 1: Verify local data directory and its essential contents for read_from_world.py
echo ""
echo "Step 1: Verifying local data directory and required files..."
if [ ! -d "${local_data_dir}" ]; then
    echo "Error: Local data directory not found at ${local_data_dir}."
    echo "This directory is required and should contain '${local_block_id_npy_path##*/}' and '${local_snbt_to_id_json_path##*/}' for reading the world."
    echo "And '${local_id_to_snbt_json_path##*/}' for writing back to the world."
    exit 1
fi
check_local_file_exists "${local_block_id_npy_path}" "Block ID npy"
check_local_file_exists "${local_snbt_to_id_json_path}" "SNBT to ID JSON"
# id_to_snbt.json is needed for write_to_world.py, check it now for completeness
check_local_file_exists "${local_id_to_snbt_json_path}" "ID to SNBT JSON"

echo "Local data directory and necessary files verified."

# Step 2: Copy the Minecraft world
# From "~/.minecraft/saves/{wt}_" to "~/.minecraft/saves/{wt}"
echo ""
echo "Step 2: Preparing Minecraft world for processing..."
if [ ! -d "${source_world_path}" ]; then
    echo "Error: Source Minecraft world template '${source_world_path}' not found."
    exit 1
fi

# Remove the active world directory if it already exists to ensure a fresh copy
if [ -d "${active_world_path}" ]; then
    echo "Removing existing active world directory: ${active_world_path}"
    rm -rf "${active_world_path}"
fi

echo "Copying world from '${source_world_path}' to '${active_world_path}'"
cp -r "${source_world_path}" "${active_world_path}"
echo "Minecraft world copied successfully."

# Step 3: Run read_from_world.py
# Uses the world path without the hyphen (active_world_path)
echo ""
echo "Step 3: Reading data from Minecraft world using ${READ_SCRIPT_NAME}..."
${LOCAL_PYTHON_EXEC} "${READ_SCRIPT_NAME}" \
    --world_path "${active_world_path}" \
    --data_path "${local_data_dir}" \
    --cx "${cx_arg}" \
    --y "${y_arg}" \
    --cz "${cz_arg}"
echo "${READ_SCRIPT_NAME} finished. Raw data should be at ${local_raw_npy_path}"
check_local_file_exists "${local_raw_npy_path}" "Generated raw.npy"

# Step 4: Copy raw.npy to remote server
echo ""
echo "Step 4: Copying '${local_raw_npy_path}' to remote server (${REMOTE_USER_HOST}:${remote_raw_npy_path})..."
scp "${local_raw_npy_path}" "${REMOTE_USER_HOST}:${remote_raw_npy_path}"
echo "File copied to remote server."

# Step 5: SSH to remote, activate conda, copy config, run inpaint.py
echo ""
echo "Step 5: Running inpainting script on remote server (${REMOTE_USER_HOST})..."
# Using bash -lc to ensure .bashrc or .profile is sourced for conda
ssh "${REMOTE_USER_HOST}" "bash -lc '
    set -e # Exit on error within the SSH session
    echo \"  Remote: Changed directory to ${REMOTE_PROJECT_DIR}\"
    cd \"${REMOTE_PROJECT_DIR}\"

    echo \"  Remote: Copying config file: ${remote_config_template_path} to ${remote_active_config_path}\"
    cp \"${remote_config_template_path}\" \"${remote_active_config_path}\"

    echo \"  Remote: Running inpainting script: ${remote_inpaint_script_path}\"
    /root/miniconda3/envs/torch/bin/python \"${remote_inpaint_script_path}\" \
        --input_file_path \"${remote_raw_npy_path}\" \
        --checkpoint_path \"${remote_checkpoint_path}\" \
        --output_file_path \"${remote_inpaint_npy_path}\"
    echo \"  Remote: Inpainting script finished.\"
'"
echo "Remote operations completed."

# Step 6: Copy inpaint.npy from remote to local data_path
echo ""
echo "Step 6: Copying '${remote_inpaint_npy_path}' from remote to '${local_inpaint_npy_path}'..."
scp "${REMOTE_USER_HOST}:${remote_inpaint_npy_path}" "${local_inpaint_npy_path}"
echo "Inpainted file copied locally."
check_local_file_exists "${local_inpaint_npy_path}" "Downloaded inpaint.npy"

# Step 7: Run write_to_world.py
# Uses the world path without the hyphen (active_world_path)
echo ""
echo "Step 7: Writing inpainted data back to Minecraft world using ${WRITE_SCRIPT_NAME}..."
${LOCAL_PYTHON_EXEC} "${WRITE_SCRIPT_NAME}" \
    --world_path "${active_world_path}" \
    --data_path "${local_data_dir}" \
    --cx "${cx_arg}" \
    --y "${y_arg}" \
    --cz "${cz_arg}"
echo "${WRITE_SCRIPT_NAME} finished. Minecraft world '${active_world_path}' should be updated."

echo ""
echo "--- Script Finished ---"
echo "✅ Automated Minecraft inpainting process completed successfully!"
echo "----------------------------------------"
