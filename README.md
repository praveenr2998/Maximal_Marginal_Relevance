# Maximal Marginal Relevance (MMR)

This is a Python implementation of the Maximal Marginal Relevance (MMR) algorithm for reranking text chunks based on their relevance and diversity.

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