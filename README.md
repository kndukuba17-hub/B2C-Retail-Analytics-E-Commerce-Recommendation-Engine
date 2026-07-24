# 🛒 E-Commerce Recommendation Engine (Item-Based Collaborative Filtering)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-CosineSimilarity-orange)
![Status](https://img.shields.io/badge/Status-Real--data%20upgrade%20in%20progress-yellow)

An item-based collaborative-filtering recommender that suggests the next product a customer is most likely to buy, based on co-purchase behaviour across the customer base.

> **Behavioural angle:** "customers who bought X also bought Y" is collective behaviour made useful. The engine learns implicit product relationships from what people actually do, not from product metadata.

---

## ⚙️ How it works
1. **User–item matrix** — transactional data is pivoted into a sparse customer × product matrix of purchase history.
2. **Item similarity** — **cosine similarity** measures how often products are co-purchased, producing an item-to-item similarity matrix.
3. **Recommendation** — given a customer's past purchases, the engine returns the top-N most similar products they haven't bought yet.

This is an **unsupervised** recommender, so it's evaluated by inspecting recommendation quality rather than a single accuracy score. A held-out evaluation (see roadmap) is the next step.

## ⚠️ Data status (honest note)
The committed notebook uses a **synthetic transaction dataset** (a DTC accessories brand). The repo is being upgraded to real transaction data — the **[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)** dataset — so the co-purchase relationships are real.

## 🧰 Tech Stack
Python · pandas · NumPy · scikit-learn (cosine similarity / pairwise metrics) · Matplotlib · Seaborn

---

## 📁 Repository Structure
```
├── README.md
├── requirements.txt
├── notebooks/
│   └── ecommerce_recommendation_engine.ipynb
├── src/
├── data/          # UCI Online Retail II download instructions — see data/README.md
├── images/
└── docs/
```

## 🚀 How to Run
```bash
git clone https://github.com/kndukuba17-hub/B2C-Retail-Analytics-E-Commerce-Recommendation-Engine.git
cd B2C-Retail-Analytics-E-Commerce-Recommendation-Engine
pip install -r requirements.txt
jupyter notebook notebooks/ecommerce_recommendation_engine.ipynb
```
Runs on Jupyter or Google Colab.

## 🗺️ Roadmap
- Swap synthetic transactions for real UCI Online Retail II data.
- Add an offline evaluation (precision@k / recall@k) using a held-out set of purchases.
- Compare item-based CF against a matrix-factorisation baseline.
