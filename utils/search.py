"""
search.py — Semantic search over node embeddings using cosine similarity.
No vector database needed — pure numpy.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.graph import KnowledgeGraph, _get_model


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (D,)  query vector (already L2-normalised)
    b: (N, D) matrix of node vectors (already L2-normalised)
    Returns: (N,) similarity scores in [-1, 1]
    """
    return b @ a  # dot product = cosine sim when both are unit-normalised


def semantic_search(query: str, kg: KnowledgeGraph, top_k: int = 5) -> list[dict]:
    """
    Find the top-k graph nodes most semantically similar to `query`.
    Returns list of dicts: {name, type, score}.
    """
    names, matrix = kg.all_embeddings()
    if len(names) == 0:
        return []

    model = _get_model()
    q_vec = model.encode([query], normalize_embeddings=True)[0]  # (D,)

    scores = cosine_similarity(q_vec, matrix)  # (N,)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        name = names[idx]
        results.append({
            "name": name,
            "type": kg.G.nodes[name].get("type", "Other"),
            "score": float(scores[idx]),
        })
    return results
