# Project DL

## SETUP

- Java version: OpenJDK 21
- Minecraft version: 1.21.1
- Minecraft server: PaperMC

Run [`setup.sh`](setup.sh).

## Data Collection

- Tool for extracting `.mca` files: [AmuletCore](https://github.com/Amulet-Team/Amulet-Core) can be used to extract `.mca` region files.
- Server download: [PaperMC](https://api.papermc.io/v2/projects/paper/versions/1.21.1/builds/133/downloads/paper-1.21.1-133.jar)
- Plugin for generation: [Chunky](https://hangarcdn.papermc.io/plugins/pop4959/Chunky/versions/1.4.36/PAPER/Chunky-Bukkit-1.4.36.jar)

Run [`start.sh`](start.sh). Configure your memory properly.
After the console appears, type in command `chunky radius 256c` to set the generation area, and `chunky start` to start generation.
