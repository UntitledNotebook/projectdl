import os
import subprocess
import time
import multiprocessing
import shutil
from pathlib import Path
import logging
import random
from datetime import datetime
import argparse
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set BASE_DIR to the script's directory
BASE_DIR = Path(__file__).parent

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate multiple Minecraft worlds in parallel.")
parser.add_argument("--server-count", type=int, default=5, help="Number of server instances (default: 5)")
parser.add_argument("--memory-per-server", default="10G", help="Memory per server, e.g., 6G, 10G (default: 10G)")
parser.add_argument("--base-port", type=int, default=25565, help="Starting port for servers (default: 25565)")
args = parser.parse_args()

# Server configuration
PAPER_JAR = "server.jar"
SERVER_COUNT = max(1, args.server_count)
MEMORY_PER_SERVER = args.memory_per_server
BASE_PORT = max(1024, args.base_port)
JAVA_PATH = "java"
CHUNKY_JAR = "Chunky-Bukkit-1.4.36.jar"
CHUNKY_COMMANDS_FILE = "chunky_commands.txt"
SERVER_PROPERTIES_TEMPLATE = BASE_DIR / "server.properties"

# Generate timestamp-based server directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SERVER_DIRS = [BASE_DIR / f"server_{TIMESTAMP}_{i}" for i in range(SERVER_COUNT)]

# Aikar’s JVM flags with GC logging
AIKAR_FLAGS = [
    f"-Xms{MEMORY_PER_SERVER}",
    f"-Xmx{MEMORY_PER_SERVER}",
    "-XX:+UseG1GC",
    "-XX:+ParallelRefProcEnabled",
    "-XX:MaxGCPauseMillis=200",
    "-XX:+UnlockExperimentalVMOptions",
    "-XX:+DisableExplicitGC",
    "-XX:+AlwaysPreTouch",
    "-XX:G1NewSizePercent=30",
    "-XX:G1MaxNewSizePercent=40",
    "-XX:G1HeapRegionSize=8M",
    "-XX:G1ReservePercent=20",
    "-XX:G1HeapWastePercent=5",
    "-XX:G1MixedGCCountTarget=4",
    "-XX:InitiatingHeapOccupancyPercent=15",
    "-XX:G1MixedGCLiveThresholdPercent=90",
    "-XX:G1RSetUpdatingPauseTimePercent=5",
    "-XX:SurvivorRatio=32",
    "-XX:+PerfDisableSharedMem",
    "-XX:MaxTenuringThreshold=1",
    "-Dusing.aikars.flags=https://mcflags.emc.gs",
    "-Daikars.new.flags=true",
    "-Xlog:gc*:file=gc.log:time,uptime:filecount=5,filesize=10M"
]

# Regions to generate per world
REGIONS = [
    {"center_x": 0, "center_z": 0, "radius": 16384},
    {"center_x": -32768, "center_z": -32768, "radius": 16384},
    {"center_x": 32768, "center_z": -32768, "radius": 16384},
    {"center_x": -32768, "center_z": 32768, "radius": 16384},
    {"center_x": 32768, "center_z": 32768, "radius": 16384},
    {"center_x": 0, "center_z": -32768, "radius": 16384},
    {"center_x": 0, "center_z": 32768, "radius": 16384},
    {"center_x": -32768, "center_z": 0, "radius": 16384},
    {"center_x": 32768, "center_z": 0, "radius": 16384},
]

def download_files():
    """Download Paper JAR and Chunky plugin if not present."""
    paper_jar_path = BASE_DIR / PAPER_JAR
    if not paper_jar_path.exists():
        logging.info(f"Downloading Paper JAR to {paper_jar_path}")
        try:
            subprocess.run([
                "wget",
                "https://api.papermc.io/v2/projects/paper/versions/1.21.1/builds/133/downloads/paper-1.21.1-133.jar",
                "-O", str(paper_jar_path)
            ], check=True, cwd=BASE_DIR)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to download Paper JAR: {e}")
            raise
    
    chunky_jar_path = BASE_DIR / CHUNKY_JAR
    if not chunky_jar_path.exists():
        logging.info(f"Downloading Chunky JAR to {chunky_jar_path}")
        try:
            subprocess.run([
                "wget",
                "https://hangarcdn.papermc.io/plugins/pop4959/Chunky/versions/1.4.36/PAPER/Chunky-Bukkit-1.4.36.jar",
                "-O", str(chunky_jar_path)
            ], check=True, cwd=BASE_DIR)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to download Chunky JAR: {e}")
            raise

def setup_server_directory(server_dir: Path, world_name: str, seed: int, port: int):
    """Set up a server directory with necessary files and customized server.properties."""
    server_dir.mkdir(parents=True, exist_ok=True)
    
    paper_jar = server_dir / PAPER_JAR
    if not paper_jar.exists():
        shutil.copy(BASE_DIR / PAPER_JAR, paper_jar)
    
    server_properties = server_dir / "server.properties"
    if SERVER_PROPERTIES_TEMPLATE.exists():
        with open(SERVER_PROPERTIES_TEMPLATE, "r") as f:
            props = f.readlines()
        with open(server_properties, "w") as f:
            for line in props:
                if line.startswith("level-name="):
                    f.write(f"level-name={world_name}\n")
                elif line.startswith("level-seed="):
                    f.write(f"level-seed={seed}\n")
                elif line.startswith("server-port="):
                    f.write(f"server-port={port}\n")
                else:
                    f.write(line)
    else:
        logging.error("server.properties template not found")
        raise FileNotFoundError("server.properties template not found")
    
    config_dir = server_dir / "config"
    if (BASE_DIR / "config").exists():
        shutil.copytree(BASE_DIR / "config", config_dir, dirs_exist_ok=True)
    
    for config_file in ["spigot.yml", "bukkit.yml"]:
        src = BASE_DIR / config_file
        if src.exists():
            shutil.copy(src, server_dir / config_file)
    
    plugins_dir = server_dir / "plugins"
    plugins_dir.mkdir(exist_ok=True)
    chunky_jar = server_dir / "plugins" / CHUNKY_JAR
    if not chunky_jar.exists():
        shutil.copy(BASE_DIR / CHUNKY_JAR, chunky_jar)
    
    chunky_config = BASE_DIR / "Chunky_config.yml"
    if chunky_config.exists():
        shutil.copy(chunky_config, plugins_dir / "config.yml")
    
    with open(server_dir / "eula.txt", "w") as f:
        f.write("eula=true\n")

def generate_chunky_commands(region: dict, world_name: str):
    """Generate Chunky commands for a region."""
    commands = [
        f"chunky world {world_name}",
        f"chunky center {region['center_x']} {region['center_z']}",
        f"chunky radius {region['radius']}",
        "chunky start"
    ]
    return commands

def run_server(server_dir: Path, world_name: str, seed: int, regions: list, port: int, instance_id: int, progress_dict: dict):
    """Run a server instance to generate all regions for a world."""
    logging.info(f"Starting server {instance_id} in {server_dir} for world {world_name} with seed {seed} on port {port}")
    
    setup_server_directory(server_dir, world_name, seed, port)
    
    cmd = [
        JAVA_PATH,
        *AIKAR_FLAGS,
        "-jar", PAPER_JAR,
        "--nogui"
    ]
    
    process = subprocess.Popen(
        cmd,
        cwd=server_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    start_time = time.time()
    server_ready = False
    current_region_idx = 0
    progress_dict[instance_id] = {"world": world_name, "region": 0, "chunks": 0, "percent": 0.0, "eta": "N/A", "rate": 0.0}
    
    try:
        while process.poll() is None:
            line = process.stdout.readline().strip()
            if line:
                logging.info(f"[{world_name}_{instance_id}] {line}")
            
            if "Done" in line and "For help, type" in line and not server_ready:
                logging.info(f"Server {world_name}_{instance_id} ready, starting region generation")
                server_ready = True
            
            if server_ready and current_region_idx < len(regions):
                # Write and send commands for the current region
                commands = generate_chunky_commands(regions[current_region_idx], world_name)
                with open(server_dir / CHUNKY_COMMANDS_FILE, "w") as f:
                    f.write("\n".join(commands) + "\n")
                
                logging.info(f"[{world_name}_{instance_id}] Generating region {current_region_idx + 1}/{len(regions)}: center ({regions[current_region_idx]['center_x']}, {regions[current_region_idx]['center_z']}), radius {regions[current_region_idx]['radius']}")
                with open(server_dir / CHUNKY_COMMANDS_FILE, "r") as f:
                    for cmd in f.readlines():
                        process.stdin.write(cmd.strip() + "\n")
                        process.stdin.flush()
                        time.sleep(0.1)
            
            # Parse Chunky progress
            match = re.search(r"\[Chunky\] Task running for ([^.]+)\. Processed: (\d+) chunks \(([\d.]+)%\), ETA: ([\d:]+), Rate: ([\d.]+) cps", line)
            if match:
                _, chunks, percent, eta, rate = match.groups()
                progress_dict[instance_id] = {
                    "world": world_name,
                    "region": current_region_idx + 1,
                    "chunks": int(chunks),
                    "percent": float(percent),
                    "eta": eta,
                    "rate": float(rate)
                }
            
            if "Chunk generation complete" in line and server_ready:
                logging.info(f"[{world_name}_{instance_id}] Region {current_region_idx + 1}/{len(regions)} generation complete")
                current_region_idx += 1
                if current_region_idx >= len(regions):
                    logging.info(f"[{world_name}_{instance_id}] All regions generated, stopping server")
                    process.stdin.write("stop\n")
                    process.stdin.flush()
            
            if time.time() - start_time > 24 * 3600:
                logging.error(f"Timeout for {world_name}_{instance_id}")
                process.terminate()
                break
        
        process.wait()
        logging.info(f"Server {world_name}_{instance_id} stopped")
    
    except Exception as e:
        logging.error(f"Error in {world_name}_{instance_id}: {e}")
        process.terminate()

def log_progress(progress_dict):
    """Log consolidated progress for all servers every 10 seconds."""
    while True:
        time.sleep(60)
        logging.info("=== Server Progress Summary ===")
        for instance_id, progress in progress_dict.items():
            if progress["region"] == 0:
                logging.info(f"Server {instance_id} ({progress['world']}): Not started")
            else:
                logging.info(
                    f"Server {instance_id} ({progress['world']}, Region {progress['region']}/{len(REGIONS)}): "
                    f"{progress['chunks']} chunks ({progress['percent']}%), ETA: {progress['eta']}, Rate: {progress['rate']} cps"
                )
        logging.info("==============================")
        # Stop logging if all servers are done
        if all(progress["region"] > len(REGIONS) for progress in progress_dict.values()):
            break

def worker(task):
    """Worker function for multiprocessing."""
    server_dir, world_name, seed, regions, port, instance_id, progress_dict = task
    run_server(server_dir, world_name, seed, regions, port, instance_id, progress_dict)

def main():
    """Main function to orchestrate parallel world generation."""
    download_files()
    
    try:
        memory_value = int(MEMORY_PER_SERVER.replace("G", ""))
        total_memory = memory_value * SERVER_COUNT
        if total_memory > 60:
            logging.warning(f"Total memory ({total_memory}GB) exceeds 60GB. Consider reducing server-count or memory-per-server.")
    except ValueError:
        logging.error("Invalid memory-per-server format. Use e.g., 6G, 10G.")
        return
    
    manager = multiprocessing.Manager()
    progress_dict = manager.dict()
    
    tasks = []
    for i in range(SERVER_COUNT):
        world_name = f"world_{i}"
        seed = random.randint(1, 1000000)
        port = BASE_PORT + i
        tasks.append((SERVER_DIRS[i], world_name, seed, REGIONS, port, i, progress_dict))
    
    max_processes = min(SERVER_COUNT, multiprocessing.cpu_count())
    logging.info(f"Starting {len(tasks)} worlds with {max_processes} processes, each generating {len(REGIONS)} regions")
    
    # Start progress logging thread
    import threading
    progress_thread = threading.Thread(target=log_progress, args=(progress_dict,), daemon=True)
    progress_thread.start()
    
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.map(worker, tasks)
    
    # Wait for progress thread to finish
    progress_thread.join()
    logging.info("All generation tasks completed")

if __name__ == "__main__":
    main()