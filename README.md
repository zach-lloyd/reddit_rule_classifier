# Reddit Rule Violation Classifier (Ensemble Approach)

![Status](https://img.shields.io/badge/Kaggle-Top%2050%25%20Leaderboard-blue) ![Role](https://img.shields.io/badge/Role-AI%20Engineer%20\(Solo\)-blue) ![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)

## Project Overview
End-to-end implementation of an automated content moderation system designed to classify whether Reddit comments violate specific subreddit rules. Unlike standard sentiment analysis, this system must understand the **context** of specific community guidelines (e.g., *"No referrals"* vs. *"No hate speech"*).

---

## Technical Architecture
This solution implements a **Weighted Ensemble** strategy combining Large Language Models (LLM) and Semantic Retrieval (RAG-lite).

### 1. LLM Classification (Qwen2.5-32B)
* **High-Throughput Inference:** Utilized `vllm` to serve 32B parameter models efficiently within GPU memory constraints.
* **Controlled Generation:** Implemented a custom `LogitsProcessor` to constrain output tokens to strict binary `Yes`/`No` probabilities, eliminating hallucinated responses.

### 2. Semantic Similarity (Qwen2.5-Embedding)
* **Vector Space Alignment:** Used `SentenceTransformers` to embed complex rules and user comments into a shared high-dimensional vector space.
* **Asymmetric Retrieval:** Calculated cosine similarity to measure the geometric alignment between a specific comment and the subreddit's rule definitions.

### 3. Ensemble Logic
* **Rank-Weighted Averaging:** Combined LLM prediction probabilities with semantic similarity scores using a **0.7 / 0.3 split**.
* **Outcome:** This hybrid approach maximized AUC by balancing the deep reasoning of the LLM with the broad contextual retrieval of the embedding model.

---

## Repository Structure

| Directory | Description |
| :--- | :--- |
| `notebooks/` | Contains the Jupyter Notebook for my highest performing submission. |

---

## Technologies
* **Models:** Qwen2.5-32B-Instruct (GPTQ Int4), Qwen2.5-Embedding
* **Libraries:** `PyTorch`, `vLLM`, `SentenceTransformers`, `Scikit-Learn`
