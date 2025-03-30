from mmr import MMR

if __name__ == "__main__":
    reranker = MMR(embedding_model_name="nomic-ai/modernbert-embed-base")

    query = "Benefits of renewable energy"
    chunks = [
        "Renewable energy reduces greenhouse gas emissions.",
        "Solar and wind energy are popular forms of renewable energy.",
        "Investments in renewable energy are growing worldwide.",
        "Fossil fuels contribute to climate change.",
        "Renewable energy can lead to economic growth.",
        "There are many challenges in transitioning to renewable energy.",
    ]

    reranked_chunks = reranker.mmr(query, chunks, top_n=3, lambda_param=0.2)
    print("Reranked chunks based on MMR:")
    for chunk in reranked_chunks:
        print("-", chunk)
