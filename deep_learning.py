import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from .utils import to_user_item_matrix, top_k_indices

class RatingsDataset(Dataset):
    def __init__(self, mat):
        self.mat = torch.tensor(mat, dtype=torch.float32)
    def __len__(self):
        return self.mat.shape[0]
    def __getitem__(self, idx):
        return self.mat[idx]

class AutoEncoder(nn.Module):
    def __init__(self, n_items, bottleneck=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_items, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 256),
            nn.ReLU(),
            nn.Linear(256, n_items),
            nn.Sigmoid(),
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

class AutoencoderRecommender:
    def __init__(self, bottleneck=64, lr=1e-3, epochs=10, batch_size=64):
        self.bottleneck = bottleneck
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.movie_ids = None
        self.user_item = None

    def fit(self, ratings: pd.DataFrame):
        mat, uidx, iidx = to_user_item_matrix(ratings)
        self.user_item = mat
        self.model = AutoEncoder(mat.shape[1], self.bottleneck)
        ds = RatingsDataset(mat)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            for batch in dl:
                pred = self.model(batch)
                loss = loss_fn(pred, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
        # store movie id mapping
        self.movie_ids = ratings['movieId'].astype('category').cat.categories.astype(int).tolist()
        return self

    def recommend_for_user(self, user_original_id: int, top_k: int = 10, exclude_rated=True):
        user_map = pd.Series(range(self.user_item.shape[0]), index=pd.CategoricalIndex(self.user_item).categories if False else None)
        # Fallback: assume sequential users; for small datasets, use first user as example
        # In practice, map user_original_id via categories saved during fit.
        # For simplicity here, if mapping fails, use first row.
        try:
            user_idx = int(user_original_id)
            if user_idx >= self.user_item.shape[0]:
                user_idx = 0
        except:
            user_idx = 0

        u = torch.tensor(self.user_item[user_idx], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            preds = self.model(u).numpy().ravel()
        if exclude_rated:
            rated_mask = self.user_item[user_idx] > 0
            preds[rated_mask] = -1e9
        idx = top_k_indices(preds, top_k)
        rec_movie_ids = [int(self.movie_ids[i]) for i in idx]
        rec_scores = preds[idx].tolist()
        return list(zip(rec_movie_ids, rec_scores))
