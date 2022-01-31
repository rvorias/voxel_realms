import json
import requests
from multiprocessing import Pool
import glob
import sys

NUM_SVGS = 8000

paths = glob.glob("svgs/*.svg")
done = [int(path.replace("svgs/","").replace("svgs\\","").replace(".svg","")) for path in paths]
print(done)
candidates = [i for i in range(1,8001) if i not in done]
print(candidates) 

with open("resources/database.json", encoding="utf8") as file:
    data = json.load(file)

def download(i):
    r = requests.get(f"https://d23fdhqc1jb9no.cloudfront.net/_Realms/{i}.svg")
    with open(f"svgs/{i}.svg", "wb") as f:
        f.write(r.content)
    print(i)

if __name__=="__main__":
    with Pool(12) as p:
        p.map(download, candidates)
