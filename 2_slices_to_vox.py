import os
import glob
from subprocess import Popen

##################
POOL_SIZE = 2
N_REALMS = 5
IN_FOLDER = "output"
OUT_FOLDER = "voxmaps"
##################

if os.name == 'nt':
    FTV_EXEC = "FileToVox\\FileToVox.exe"
else:
    FTV_EXEC = "FileToVox-v1.13-win/FileToVox.exe"

paths = glob.glob(f"{IN_FOLDER}/height_*.png")
idxs = [path.replace(f"{IN_FOLDER}/height_", "").replace(".png", "") for path in paths]
done_folders = glob.glob(f"{OUT_FOLDER}/wmap_*.vox")
done_idxs = [path.replace(f"{OUT_FOLDER}/wmap_", "").replace(".vox", "") for path in paths]
candidates = [path for path in paths if path not in done_idxs]
candidates = candidates[:N_REALMS]

if __name__ == '__main__':
    commands = []
    for realm_number in candidates:
        commands.append(
            f"{FTV_EXEC} " +
            f"--i {IN_FOLDER}/hslices_{realm_number}" +
            f"--o {OUT_FOLDER}/wmap_{realm_number}"
        )
    procs = [Popen(i, shell=True) for i in commands]
    for p in procs:
        p.wait()
