CSCE 676 — Data Mining Project


## Project Goals

Project Overview

This project aims to build an effective recommendation system by leveraging both user interaction data and item information. We explore how different representation learning methods (e.g., collaborative filtering, Word2Vec, and TF-IDF) and ranking models impact recommendation performance. The focus is on understanding the role of user behavior sequences, feature engineering, and model design in improving recommendation quality.

👉 Start here: main_notebook.ipynb

## Research Questions
How effective are sequential user behavior embeddings (e.g., Word2Vec) in recommendation tasks?
Does incorporating item content (e.g., TF-IDF features) improve recommendation performance?
How does a two-stage recommendation pipeline (recall + ranking) compare to simpler similarity-based methods?
What is the impact of time-aware modeling (e.g., time decay) on recommendation quality?

## Methods Used
This project follows a two-stage recommendation pipeline:

1. Recall Stage (Candidate Generation)
Item-based Collaborative Filtering (ItemCF) is used to generate candidate items.
Word2Vec is applied by treating user interaction sequences as sentences to learn item embeddings.
TF-IDF is applied on item descriptions and interaction data to capture semantic features.
2. Ranking Stage
User features, item features, and multiple embeddings (Word2Vec, TF-IDF) are concatenated.
A Deep Neural Network (DNN) is trained to learn non-linear interactions and predict relevance scores.
The DNN refines the candidate pool and produces the final ranking.
Additional Experiments
Direct similarity-based ranking using cosine similarity on embeddings.
Time decay to emphasize recent user interactions.

## Data

This project uses a large-scale user-item interaction dataset.

Due to GitHub file size limitations, the raw data and generated embeddings are not included in this repository. The dataset includes:

User information
Item metadata
Transaction (interaction) records

To reproduce the results, please download the dataset from the original source (e.g., Kaggle / course-provided data) and place it under the data/ directory.


## Key Dependencies
Python 3.x
pandas
numpy
scikit-learn
gensim (Word2Vec)


## Repository Structure

.
├── main_notebook.ipynb
├── checkpoints/
│   ├── checkpoint_1.ipynb
│   ├── checkpoint_2.ipynb
├── data/            # not included
├── embd/            # not included
├── models/          # not included
├── requirements.txt
├── README.md
## Results Summary

ItemCF-based recall achieves the lowest MAP@K performance among all methods. Incorporating embedding-based approaches (Word2Vec and TF-IDF) leads to moderate improvements. The best results are obtained using Word2Vec embeddings derived from user interaction sequences, highlighting the importance of sequential behavioral information.

Direct similarity-based ranking using cosine similarity performs worse than ItemCF, suggesting that simple similarity metrics cannot capture complex feature interactions. In contrast, the DNN-based ranking model significantly improves performance by learning non-linear relationships.

We also observe that applying time decay substantially boosts performance, indicating that recent user behavior is more informative than older interactions.

## Limitations
Only a limited window of user interaction history is used, which may miss long-term preferences.
The data split does not fully simulate a realistic temporal recommendation setting.
The dataset exhibits strong long-tail and sparsity issues, affecting model performance.
No multimodal features (e.g., item images) are incorporated.
