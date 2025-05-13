# Project DL

## SETUP

- Java version: OpenJDK 21
- Minecraft version: 1.21.1
- Minecraft server: PaperMC

Run [`setup.sh`](setup.sh). This script should ensure Java and necessary tools are available.

## Data Collection Workflow

The data collection process involves two main Python scripts:

1.  **`data.py`**: Automates the generation of Minecraft worlds.
    *   Downloads the PaperMC server JAR and the Chunky plugin if they are not already present.
    *   Sets up a new server instance in a timestamped directory (e.g., `server_YYYYMMDD_HHMMSS`).
    *   Configures `server.properties` with a unique world name (e.g., `world_YYYYMMDD_HHMMSS`) and a random seed.
    *   Launches the Minecraft server.
    *   Automatically issues Chunky commands to pre-generate specified regions of the world.
    *   **Usage**:
        ```bash
        python data.py --memory-per-server 6G --base-port 25565
        ```
        (Adjust memory and port as needed. Currently, it's simplified to run one server instance).
        The regions to be generated are defined in the `REGIONS` list within `data.py`.

2.  **`mca2json.py`**: Extracts data from the generated worlds and converts it into JSON format for model training.
    *   Scans the current directory (or a specified `--base-scan-dir`) for `server_*` folders.
    *   Identifies world save directories within these server folders.
    *   Uses the Amulet-Core library to read `.mca` region files from these worlds.
    *   Extracts 128x128x24 (height configurable) block samples from the surface of the Minecraft world, within a specified chunk radius around (0,0).
    *   For each block in a sample, it records its namespaced ID, blockstate properties, and relative position.
    *   Combines all extracted samples from all found worlds into a single JSON output file.
    *   **Usage**:
        ```bash
        python mca2json.py --radius 16 --y-base 60 --slice-height 24 --output-file training_data.json --base-scan-dir .
        ```
        *   `--radius`: Specifies the radius in chunks (e.g., 16 means processing chunks from -16 to +15 in X and Z).
        *   `--y-base`: The starting Y-level for the 24-block high slice.
        *   `--slice-height`: The height of the Y-slice.
        *   `--output-file`: Name of the JSON file to save the data.
        *   `--base-scan-dir`: Directory containing the `server_*` folders.

### Supporting Files

-   `server.properties`: Template for server configuration.
-   `Chunky_config.yml`: Example configuration for the Chunky plugin (if needed, currently `data.py` uses commands).
-   `spigot.yml`, `bukkit.yml`: Minecraft server configuration files.

## Original Manual Steps (for reference or direct server interaction)

- Tool for exploring `.mca` files: [Amulet Editor](https://www.amuletmc.com/) (GUI based on Amulet-Core).
- Server download: [PaperMC](https://api.papermc.io/v2/projects/paper/versions/1.21.1/builds/133/downloads/paper-1.21.1-133.jar)
- Plugin for generation: [Chunky](https://hangarcdn.papermc.io/plugins/pop4959/Chunky/versions/1.4.36/PAPER/Chunky-Bukkit-1.4.36.jar)

If running manually via [`start.sh`](start.sh): Configure your memory properly. After the console appears, you might type commands like `chunky world <worldname>`, `chunky center 0 0`, `chunky radius 256` (for block radius) or `chunky radius 16c` (for chunk radius), and then `chunky start` to start generation.

## TODO
1.  ~~A python script to launch multiple worlds at the same time for data generation.~~ (Partially done with `data.py` for single instance, can be re-extended if needed).
2.  ~~A python script to read in the `.mca` file and print out the json formatted training data.~~ (Completed with `mca2json.py`).
    *   In presentation, remove the particular selected region.
    *   Put in the regenerated version through json file.
3.  Develop and train the model using the JSON data.
4.  Implement a method to convert model output back into a usable Minecraft format (e.g., schematic, or directly modifying world files).