from multiprocessing import Pool
import glob
import sys
sys.path.append('./pipeline')
from pipeline.vox_chirurgy import operate

##################
POOL_SIZE = 2
N_REALMS = 5
IN_FOLDER = "voxmaps"
OUT_FOLDER = "voxmaps"
##################

paths = glob.glob(f"{IN_FOLDER}/wmap_*.vox")
idxs = [path.replace(f"{IN_FOLDER}/wmap", "").replace(".vox", "") for path in paths]
done_folders = glob.glob(f"{OUT_FOLDER}/fmap_*.vox")
done_idxs = [path.replace(f"{OUT_FOLDER}/fmap_", "").replace(".vox", "") for path in paths]
candidates = [path for path in paths if path not in done_idxs]
candidates = candidates[:N_REALMS]

paths = glob.glob("svgs/*.svg")[:5]
realm_numbers = [int(p.replace("svgs/", "").replace("svgs\\", "").replace(".svg", "")) for p in paths]

POOL_SIZE = 2

if __name__ == '__main__':
    print(f"Found realm numbers: {realm_numbers}")
    with Pool(POOL_SIZE) as p:
        p.map(operate, realm_numbers)
