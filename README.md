CSCE 676 — Data Mining Project


## Project Goals

The goal of this project is to build an effective recommendation system by leveraging both user behavior data and item information. Specifically, we aim to explore how different representation learning methods (e.g., collaborative filtering, Word2Vec, and TF-IDF) and ranking models can improve recommendation performance. The project focuses on understanding the impact of user interaction sequences, feature engineering, and model design on the overall system quality.


## Methods Used
This project follows a typical two-stage recommendation pipeline consisting of candidate generation and ranking.

In the recall stage, we use Item-based Collaborative Filtering (ItemCF) to generate a pool of candidate items based on user interaction history. In addition, we learn item embeddings using Word2Vec by treating user interaction sequences as sentences, and apply TF-IDF on item descriptions and interaction data to capture semantic information.

In the ranking stage, we concatenate user features, item features, and multiple embedding representations, and feed them into a Deep Neural Network (DNN) to learn complex, non-linear interactions. The model is trained to predict user-item relevance and produce a refined ranking.

We also explore alternative approaches such as directly computing cosine similarity between embeddings for ranking, and investigate techniques like time decay to better capture the importance of recent user behaviors.



