#!/usr/bin/env python3
"""
Download and extract MovieLens datasets.
"""
import argparse, os, zipfile, io, sys
from urllib.request import urlopen

URLS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=URLS.keys(), default="small")
    ap.add_argument("--dest", default="data")
    args = ap.parse_args()
    url = URLS[args.variant]
    os.makedirs(args.dest, exist_ok=True)
    print(f"Downloading {args.variant} from {url} ...")
    with urlopen(url) as r:
        data = r.read()
    print("Unzipping ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(args.dest)
    print("Done. Files in:", args.dest)

if __name__ == "__main__":
    main()
