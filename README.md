# Project DL

## SETUP

- Minecraft version: TODO
- Minecraft server: PaperMC

## Data Collection

- [AmuletCore](https://github.com/Amulet-Team/Amulet-Core) can be used to extract `.mca` region files.
- Server download: [PaperMC](https://api.papermc.io/v2/projects/paper/versions/1.21.1/builds/133/downloads/paper-1.21.1-133.jar)

```bash
mkdir paper-server && cd paper-server
wget https://api.papermc.io/v2/projects/paper/versions/1.21.1/builds/133/downloads/paper-1.21.1-133.jar
mv paper-1.21.1-133.jar paper.jar
chmod +x start.sh
```