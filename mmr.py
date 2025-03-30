import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class MMR:
    def __init__(self, embedding_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name)

    def embed_text(self, text):
        """
        Convert text to vectors/embeddings using the embedding model provided

        :param text: str - text to be converted to embedding
        :return: embedding - numpy array - embeddings of the text
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # We use the mean pooling over the token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Compute cosine similarity between two vectors.

        :param vec1: numpy array - first vector
        :param vec2: numpy array - second vector
        :return: cosine_similarity - float - cosine similarity between the two vectors
        """
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

    def mmr(self, query, chunks, top_n=5, lambda_param=0.5):
        """
        Selects top_n chunks based on Maximal Marginal Relevance (MMR).

        :param query: str - query text
        :param chunks: list of str - list of candidate chunks
        :param top_n: int - number of top chunks to be selected
        :param lambda_param: float - lambda parameter for MMR
        :return: selected_chunks - list of str - list of selected chunks
        """
        # Step 1: Embed the query and each candidate chunk
        query_embedding = self.embed_text(query)
        chunk_embeddings = [self.embed_text(chunk) for chunk in chunks]

        # Step 2: Compute cosine similarities between the query and each chunk
        query_similarities = np.array(
            [self.cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]
        )

        # Step 3: Compute pairwise cosine similarities between chunks for redundancy measure
        n = len(chunks)
        redundancy_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                redundancy_matrix[i, j] = self.cosine_similarity(
                    chunk_embeddings[i], chunk_embeddings[j]
                )

        # Step 4: Initialize MMR selection
        selected_indices = []
        candidate_indices = list(range(n))

        # First, select the chunk with the highest similarity to the query
        first_index = int(np.argmax(query_similarities))
        selected_indices.append(first_index)
        candidate_indices.remove(first_index)

        # Step 5: Iteratively select remaining chunks
        while len(selected_indices) < top_n and candidate_indices:
            mmr_scores = []
            for idx in candidate_indices:
                # Calculate redundancy: maximum similarity with any already selected chunk
                redundancy = (
                    max([redundancy_matrix[idx][sel] for sel in selected_indices])
                    if selected_indices
                    else 0
                )
                # MMR score: balance between query relevance and redundancy penalty
                score = (
                    lambda_param * query_similarities[idx]
                    - (1 - lambda_param) * redundancy
                )
                mmr_scores.append(score)
            # Select the candidate with the highest MMR score
            best_candidate = candidate_indices[int(np.argmax(mmr_scores))]
            selected_indices.append(best_candidate)
            candidate_indices.remove(best_candidate)

        # Return the selected chunks
        return [chunks[i] for i in selected_indices]
