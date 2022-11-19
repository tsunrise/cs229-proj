import os
import requests
import tqdm
import toml
import tarfile
import shutil
from dataclasses import dataclass

DATASETS_CONFIG_PATH = "datasets.toml"
CACHE_DIR = ".cs229_cache"

def download(filename: str, url: str, force: bool = False) -> str:
    """Download the tar file from crate.io, and output the path to the tar file."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cache_path = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(cache_path) or force:
        with requests.get(url, stream=True) as r:
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            chunk_size = 1024
            with open(cache_path, "wb") as f:
                print(f"Downloading {filename} from {url}")
                progress = tqdm.tqdm(total = total_size_in_bytes, unit = 'iB', unit_scale = True, colour="blue")
                for chunk in r.iter_content(chunk_size=chunk_size):
                    progress.update(len(chunk))
                    f.write(chunk)
                progress.close()
    else:
        print("Using cached: ", filename)
    return cache_path

@dataclass
class CratesIOCSVPath:
    categories: str
    crates: str
    crates_categories: str
    dependencies: str

def dump_crate_io(force_download = False) -> CratesIOCSVPath:
    """Download the tar file from crate.io, and output the path to the csv data."""
    config = toml.load(DATASETS_CONFIG_PATH)
    tar_path = download("crates-io.tar.gz", config["crates-io"]["url"], force=force_download)
    # extract the tar file
    crates_io_path = os.path.join(CACHE_DIR, "crates-io")
    if os.path.exists(crates_io_path):
        print("Cleaning up old crates-io data")
        shutil.rmtree(crates_io_path)
    os.makedirs(crates_io_path)
    print("Extracting crates-io data")
    with tarfile.open(tar_path) as tar:
        paths = {}
        for info in tqdm.tqdm(tar.getmembers(), unit="files", colour="green"):
            filename = os.path.basename(info.name)
            if filename.endswith(".csv"):
                info.name = os.path.basename(info.name)
                tar.extract(info, crates_io_path)
                paths[filename] = os.path.join(crates_io_path, info.name)
    return CratesIOCSVPath(paths["categories.csv"], paths["crates.csv"], paths["crates_categories.csv"], paths["dependencies.csv"])

if __name__ == "__main__":
    crates_io_path = dump_crate_io()
    print("Crates.io data is at: ", crates_io_path)