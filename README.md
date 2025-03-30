# Maximal Marginal Relevance (MMR)

This is a Python implementation of the Maximal Marginal Relevance (MMR) algorithm for reranking text chunks based on their relevance and diversity.

## MMR
The formula for MMR score for a candidate is

$$
\text{score(candidate)} = \lambda \cdot \text{similarity(query, candidate)} - (1 - \lambda) \cdot \max\Big(\text{similarity(candidate, selected texts)}\Big)
$$

NOTE : Update value of lambda as per requirements in main.py
1. **Higher lambda value** - Algorithm prioritizes relevance over diversity.
2. **Lower lambda value** - Algorithm prioritizes diversity over relevance.
3. **High candidate score(MMR score)** - A high score indicates that the candidate is both highly relevant to the query and not too similar to any of the chunks already selected. In other words, it adds new, useful information while still being on-topic.
4. **Low candidate score(MMR score)** - could mean two things
    * The candidate is not very similar to the query, so it isn’t relevant.
    * The candidate is too similar to what’s already been selected, meaning it doesn’t contribute additional, diverse information.

## SETUP
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync
```

## RUN
```bash
uv run main.py
```
In the main.py pass the desired embedding model name as an argument to the MMR class.

### EXAMPLE
```python
from mmr import MMR

if __name__ == "__main__":
    reranker = MMR(embedding_model_name="nomic-ai/modernbert-embed-base")
"""
