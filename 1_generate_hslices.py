import time
from multiprocessing import Pool
from functools import partial
import glob
from omegaconf import OmegaConf
import json

from pipeline.run import run_pipeline

config = OmegaConf.load("pipeline/config.yaml")
f = partial(run_pipeline, config=config)

##################
POOL_SIZE = 12
N_REALMS = None
IN_FOLDER = "svgs"
CHECK_FOLDER = "output/heights"
generate_directions = True
##################

paths = glob.glob(f"{IN_FOLDER}/*.svg")
idxs = [path.replace(f"{IN_FOLDER}/", "").replace(".svg", "") for path in paths]
# print(idxs)
done_paths = glob.glob(f"{CHECK_FOLDER}/height_*.png")
done_idxs = [path.replace(f"{CHECK_FOLDER}/height_", "").replace(".png", "") for path in done_paths]

candidates = [paths[i] for i in range(len(paths)) if idxs[i] not in done_idxs]
if N_REALMS is not None:
    candidates = candidates[:N_REALMS]

# redo for lo and mid
done_paths = glob.glob(f"output/metadata/*.json")
no_hi = []
for jfile in done_paths:
    with open(jfile, "r") as file:
        d = json.load(file)
    if d["landscape_height"] != "hi":
        no_hi.append("svgs/" + jfile.replace("output/metadata/","").replace(".json","") + ".svg")
candidates = no_hi

if __name__=="__main__":
    print(f"done {len(done_paths)} realms")
    print(f"starting with {len(candidates)} candidates")
    start = time.time()
    with Pool(POOL_SIZE) as p:
        p.map(f, candidates)

    end = time.time()
    total_time = end-start
    time_per_realm = total_time / POOL_SIZE
    print(f"Generating all will take {time_per_realm*80/36} hours.")
