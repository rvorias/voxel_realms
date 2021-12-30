import time
from multiprocessing import Pool
from functools import partial
import glob
from omegaconf import OmegaConf

from pipeline.run import run_pipeline

config = OmegaConf.load("pipeline/config.yaml")
f = partial(run_pipeline, config=config)

##################
POOL_SIZE = 2
N_REALMS = 5
IN_FOLDER = "./svgs"
OUT_FOLDER = "./output"
##################

paths = glob.glob(f"{IN_FOLDER}/*.svgs")
idxs = [path.replace(f"{IN_FOLDER}/", "").replace(".svg", "") for path in paths]
done_folders = glob.glob(f"{OUT_FOLDER}/height_*.png")
done_idxs = [path.replace(f"{OUT_FOLDER}/height_", "").replace(".png", "") for path in paths]
candidates = [path for path in paths if path not in done_idxs]
candidates = candidates[:N_REALMS]

if __name__=="__main__":
    start = time.time()
    print(f"Found svgs: {paths}")
    with Pool(POOL_SIZE) as p:
        p.map(f, paths)

    end = time.time()
    total_time = end-start
    time_per_realm = total_time / POOL_SIZE
    print(f"Generating all will take {time_per_realm*8000/3600} hours.")
