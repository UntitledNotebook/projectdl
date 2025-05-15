import os
import subprocess
import time
import shutil
from pathlib import Path
import logging
import random
from datetime import datetime
import argparse
import re
import select

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set BASE_DIR to the script's directory
BASE_DIR = Path(__file__).parent

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate a Minecraft world with a specified radius using Chunky.")
parser.add_argument("--memory", default="6G", help="Memory for the server, e.g., 6G, 10G (default: 6G)")
parser.add_argument("--port", type=int, default=25565, help="Port for the server (default: 25565)")
parser.add_argument("--chunk-radius", type=int, required=True, help="Radius in chunks for world generation (e.g., 16 for a 256 block radius from center 0,0).")
args = parser.parse_args()

# Server configuration
PAPER_JAR = "server.jar"
MEMORY = args.memory
SERVER_PORT = max(1024, args.port) # Port for the single server
JAVA_PATH = "java"
CHUNKY_JAR = "Chunky-Bukkit-1.4.36.jar"
CHUNKY_COMMANDS_FILE = "chunky_commands.txt"
SERVER_PROPERTIES_TEMPLATE = BASE_DIR / "server.properties"

# Generate timestamp-based server directories
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SERVER_DIR = Path(f"worlds/server_{TIMESTAMP}")

# Aikar’s JVM flags with GC logging
AIKAR_FLAGS = [
    f"-Xms{MEMORY}",
    f"-Xmx{MEMORY}",
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

# REGIONS will be defined in main based on args.chunk_radius

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

def setup_server_directory(server_dir: Path, seed: int, port: int):
    """Set up a server directory with necessary files and customized server.properties."""
    server_dir.mkdir(parents=True, exist_ok=True)
    
    paper_jar_dest = server_dir / PAPER_JAR
    if not paper_jar_dest.exists():
        shutil.copy(BASE_DIR / PAPER_JAR, paper_jar_dest)
    
    server_properties_dest = server_dir / "server.properties"
    if SERVER_PROPERTIES_TEMPLATE.exists():
        with open(SERVER_PROPERTIES_TEMPLATE, "r") as f:
            props = f.readlines()
        with open(server_properties_dest, "w") as f:
            for line in props:
                if line.startswith("level-seed="):
                    f.write(f"level-seed={seed}\n")
                elif line.startswith("server-port="):
                    f.write(f"server-port={port}\n")
                else:
                    f.write(line)
    else:
        logging.warning(f"server.properties template not found at {SERVER_PROPERTIES_TEMPLATE}. Using default server generation.")

    # Copy spigot.yml and bukkit.yml if they exist
    for config_file_name in ["spigot.yml", "bukkit.yml"]:
        src_config_path = BASE_DIR / config_file_name
        if src_config_path.exists():
            shutil.copy(src_config_path, server_dir / config_file_name)
        else:
            logging.info(f"{config_file_name} not found at {src_config_path}, skipping copy.")

    plugins_dir = server_dir / "plugins"
    plugins_dir.mkdir(exist_ok=True)
    chunky_jar_dest = plugins_dir / CHUNKY_JAR # Note: Chunky JAR name includes version
    if not chunky_jar_dest.exists():
        shutil.copy(BASE_DIR / CHUNKY_JAR, chunky_jar_dest)

    with open(server_dir / "eula.txt", "w") as f:
        f.write("eula=true\n")

def generate_chunky_commands(radius: int):
    """Generate Chunky commands for a region."""
    commands = [
        f"chunky radius {radius}",
        "chunky start"
    ]
    return commands

def run_server(server_dir: Path, seed: int, radius: int, port: int):
    """Run a server instance to generate all regions for a world."""
    logging.info(f"Starting server in {server_dir} with seed {seed} on port {port}")

    setup_server_directory(server_dir, seed, port)
    
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
        stderr=subprocess.STDOUT, # Capture stderr to stdout
        text=True,
        bufsize=1 # Line-buffered
    )
    
    start_time = time.time()
    
    try:
        while process.poll() is None:
            ready_to_read, _, _ = select.select([process.stdout], [], [], 0.1) 

            if ready_to_read:
                line = process.stdout.readline()
                if not line: # EOF
                    logging.info(f"{server_dir} stdout EOF. Process likely exited.")
                    break 
                line = line.strip()

                if line:
                    logging.info(f"{line}")
                    if "Done" in line and "For help, type" in line:
                        logging.info(f"{server_dir} ready, starting region generation")
                        commands = generate_chunky_commands(radius)
                        for command in commands:
                            try:
                                process.stdin.write(f"{command}\n")
                                process.stdin.flush()
                            except (BrokenPipeError, OSError) as e:
                                logging.error(f"[{server_dir}] Error writing command '{command}': {e}. Process may have exited.")
                                if process.poll() is None: process.terminate()
                                break
                            time.sleep(0.2) # Give server time to process command
            
                    if "Task finished" in line:
                        try:
                            process.stdin.write("stop\n")
                            process.stdin.flush()
                        except (BrokenPipeError, OSError) as e:
                            logging.error(f"[{server_dir}] Error writing 'stop' command: {e}. Process may have exited.")
                            if process.poll() is None: process.terminate()
                            break 
            # Timeout for the entire generation process (e.g., 24 hours)
            if time.time() - start_time > 24 * 3600: 
                logging.error(f"Timeout for {server_dir}. Terminating process.")
                if process.poll() is None:
                    process.terminate()
                break
        
        final_exit_code = process.poll()
        if final_exit_code is None: 
            logging.info(f"Waiting for {server_dir} to stop (up to 5 mins)...")
            try:
                final_exit_code = process.wait(timeout=300) 
            except subprocess.TimeoutExpired:
                logging.error(f"{server_dir} did not exit cleanly. Killing.")
                if process.poll() is None: process.kill()
                final_exit_code = process.wait()
        logging.info(f"{server_dir} stopped. Final exit code: {final_exit_code}")
    
    except Exception as e:
        logging.error(f"Unhandled exception in run_server for {server_dir}: {e}", exc_info=True)
        if process and process.poll() is None:
            logging.info(f"Terminating process {server_dir} due to unhandled exception.")
            process.terminate()
            try: process.wait(timeout=60)
            except subprocess.TimeoutExpired: process.kill()
    finally:
        if process and process.stdin:
            try: process.stdin.close()
            except Exception: pass
        if process and process.stdout:
            try: process.stdout.close()
            except Exception: pass


def main():
    """Main function to orchestrate world generation."""
    download_files()
    
    try:
        memory_value = int(MEMORY.replace("G", ""))
        if memory_value > 60: 
            logging.warning(f"Memory ({MEMORY}) is very high ({memory_value}GB). Ensure this is intended.")
    except ValueError:
        logging.error("Invalid memory format. Use e.g., 6G, 10G.")
        return
    
    seed = random.randint(1, 1000000000) # Increased seed range
    
    # Define the region to generate based on command-line argument
    chunky_radius_blocks = args.chunk_radius * 16

    
    logging.info(f"Starting world generation in {SERVER_DIR}, using port {SERVER_PORT}")
    logging.info(f"Targeting radius {chunky_radius_blocks} blocks.")
    
    try:
        run_server(SERVER_DIR, seed, chunky_radius_blocks, SERVER_PORT)
    except Exception as e:
        logging.error(f"Main execution error: {e}", exc_info=True)
    
    logging.info("World generation task completed.")

if __name__ == "__main__":
    main()