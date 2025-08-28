# Data Folder

Place the **MovieLens** dataset here. Recommended: **ml-latest-small** (100k) for quick experiments.

## Quick Download (Python)
```bash
python data/get_movielens.py --variant small --dest data/
```

This will download and unpack MovieLens to `data/ml-latest-small/`.

Expected files:
- `ratings.csv`
- `movies.csv`
- `tags.csv` (optional)
