"""Reusable item-based collaborative-filtering recommender for Online Retail II.

Extracted from the notebook so the logic is importable and testable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing customers, cancellations and returns; normalise dtypes."""
    df = df.dropna(subset=["Customer ID"]).copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


class ItemCFRecommender:
    """Binary item-based collaborative filtering via cosine similarity.

    Example
    -------
    >>> rec = ItemCFRecommender(min_customers=20).fit(train_df)
    >>> rec.recommend_for_customer(12345, k=10)
    """

    def __init__(self, min_customers: int = 20):
        self.min_customers = min_customers

    def fit(self, train: pd.DataFrame) -> "ItemCFRecommender":
        pop = train.groupby("StockCode")["Customer ID"].nunique()
        keep = pop[pop >= self.min_customers].index
        train = train[train["StockCode"].isin(keep)]

        self.users = train["Customer ID"].unique()
        self.items = train["StockCode"].unique()
        self.uidx = {u: i for i, u in enumerate(self.users)}
        self.iidx = {p: i for i, p in enumerate(self.items)}
        self.inv = {i: p for p, i in self.iidx.items()}

        ui = csr_matrix(
            (np.ones(len(train)),
             (train["Customer ID"].map(self.uidx).values,
              train["StockCode"].map(self.iidx).values)),
            shape=(len(self.users), len(self.items)))
        ui.data[:] = 1.0
        self.ui = ui
        self.item_sim = cosine_similarity(ui.T, dense_output=True)
        np.fill_diagonal(self.item_sim, 0.0)
        # per-customer owned item indices
        self._owned = train.groupby("Customer ID")["StockCode"].apply(
            lambda s: [self.iidx[p] for p in set(s)]).to_dict()
        return self

    def recommend(self, hist_idx, k: int = 10):
        """Top-k product codes for a list of owned item indices."""
        if not hist_idx:
            return []
        scores = self.item_sim[hist_idx].sum(axis=0)
        scores[hist_idx] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.inv[i] for i in top]

    def recommend_for_customer(self, customer_id: int, k: int = 10):
        return self.recommend(self._owned.get(customer_id, []), k)
