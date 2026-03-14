# B2C Retail Analytics: E-Commerce Recommendation Engine

## Executive Summary
In the Direct-to-Consumer (DTC) retail space, customer retention and cross-selling are the primary drivers of profitability. Generic digital storefronts suffer from high bounce rates, whereas personalised shopping experiences significantly increase Customer Lifetime Value (CLV) and average order value.

This project bridges consumer psychology and machine learning by developing an **Item-Based Collaborative Filtering Recommendation Engine**. 

**Commercial Objective:** Deploy a programmatic recommendation system that analyses historical purchasing behaviour to accurately predict and suggest the next product a customer is most likely to buy, driving cross-sell revenue and personalising the e-commerce experience.

## Technical Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (Cosine Similarity, Pairwise Metrics)
* **Data Processing & Transformation:** Pandas, NumPy
* **Market Visualisation:** Matplotlib, Seaborn

## Core Methodology
1. **DTC Data Simulation:** Generates a high-fidelity dataset of retail transactions, mimicking a DTC accessories brand (e.g., purchases of rings, necklaces, and bracelets). 
2. **User-Item Matrix Transformation:** Transforms raw transactional data into a sparse matrix, mapping every customer's purchase history against the entire product catalogue.
3. **Collaborative Filtering:** Utilises **Cosine Similarity** to calculate the mathematical distance between products based on co-purchasing behaviour. If customers frequently buy 'Product A' and 'Product B' together, the algorithm learns this implicit psychological link.
4. **Dynamic Recommendation:** A custom Python function that ingests a target customer's past purchases and outputs the top 3 highest-probability recommendations to surface on their checkout page.
