import time
from multiprocessing import Pool
from functools import partial
import glob
from omegaconf import OmegaConf

from pipeline.run_direction import run_pipeline

config = OmegaConf.load("pipeline/config.yaml")
f = partial(run_pipeline, config=config)

##################
POOL_SIZE = 3
IN_FOLDER = "svgs"
CHECK_FOLDER = "output/directions"
##################

paths = glob.glob(f"{IN_FOLDER}/*.svg")
idxs = [path.replace(f"{IN_FOLDER}/", "").replace(".svg", "") for path in paths]
# print(idxs)
done_paths = glob.glob(f"{CHECK_FOLDER}/*.direction")
done_idxs = [path.replace(f"{CHECK_FOLDER}/", "").split("_")[0] for path in done_paths]
candidates = [paths[i] for i in range(len(paths)) if idxs[i] not in done_idxs]

if __name__=="__main__":
    pass
    print(f"starting with {len(candidates)} candidates")
    start = time.time()
    with Pool(POOL_SIZE) as p:
        p.map(f, candidates)

    end = time.time()
    total_time = end-start
    time_per_realm = total_time / POOL_SIZE
    print(f"Generating all will take {time_per_realm*8000/3600} hours.")
