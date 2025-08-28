# 🎬 Movie Recommendation System

A clean, end-to-end recommendation engine built from scratch using:
- **Collaborative Filtering** (user-based & item-based)
- **Content-Based Filtering** (metadata + tags)
- **Hybrid & Deep Learning** (weighted ensemble + autoencoder)

> Built for learning, interviews, and real demos. Modular, testable, and production-friendly.

## ✨ Features
- User & Item KNN with cosine similarity
- TF‑IDF + genres encoding for content vectors
- User taste profiles from history and explicit weights
- Hybrid ensemble (tunable weights)
- Simple **Autoencoder** (PyTorch) for latent factors
- Streamlit app to try recommendations interactively
- Clean repo structure + tests

## 📁 Structure
```
movie-recommendation-system/
├── app/                 # Streamlit demo app
├── data/                # MovieLens data goes here
├── notebooks/           # EDA & experiments
├── src/                 # Core library code
├── tests/               # Unit tests (sample)
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Quickstart

1) **Install deps**
```bash
pip install -r requirements.txt
```

2) **Download data**
```bash
python data/get_movielens.py --variant small --dest data/
```

3) **Run Streamlit app**
```bash
streamlit run app/streamlit_app.py
```

## 🧠 Models

- `CollaborativeFiltering` (user/item KNN):
  - Builds user–item matrix
  - Computes cosine similarity
  - Predicts via neighborhood weighted mean

- `ContentBasedRecommender`:
  - TF‑IDF of tags/genres text + multi‑hot genres
  - User profiles as weighted average of liked movie vectors

- `HybridRecommender`:
  - Normalizes scores from CF & CBF
  - Weighted sum: `score = α·CF + (1-α)·CBF`

- `AutoencoderRecommender`:
  - Learns compressed latent vectors for users
  - Reconstructs ratings for Top‑N suggestions

## 📊 Evaluation
Run the example script to compute RMSE and ranking metrics:
```bash
python src/example_evaluate.py
```

## 📜 License
MIT — do anything, just keep the notice.
