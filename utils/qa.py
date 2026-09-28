"""
qa.py — Natural language Q&A over the knowledge graph.

Pipeline:
  1. Semantic search → find most relevant nodes
  2. Expand to their 1-hop neighbours (subgraph)
  3. Render subgraph as text context
  4. Claude generates an answer grounded in that context
"""
import json
import urllib.request
from utils.graph import KnowledgeGraph
from utils.search import semantic_search


def _call_claude(messages: list, system: str, api_key: str) -> str:
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 500,
        "system": system,
        "messages": messages,
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"].strip()


def answer_question(question: str, kg: KnowledgeGraph, api_key: str) -> dict:
    """
    Full RAG pipeline:
      retrieve relevant nodes → build subgraph text → Claude answers.
    """
    # Step 1: semantic search — get top 5 relevant nodes
    top_nodes = semantic_search(question, kg, top_k=5)
    seed_names = [n["name"] for n in top_nodes]

    # Step 2: expand to 1-hop neighbours
    expanded = set(seed_names)
    for name in seed_names:
        neighbours = kg.get_neighbours(name)
        for nb in neighbours:
            expanded.add(nb["node"])

    # Step 3: render subgraph as readable text
    context_text = kg.get_subgraph_text(list(expanded))

    # Step 4: Claude answers grounded in context
    system = """You are a helpful knowledge graph assistant.
Answer the user's question using ONLY the graph context provided.
Be concise and factual. Use bullet points when listing multiple items.
If the answer isn't in the context, say so honestly."""

    user_msg = f"""Question: {question}

Knowledge Graph Context:
{context_text}

Answer the question based on this context."""

    answer = _call_claude([{"role": "user", "content": user_msg}], system, api_key)

    context_data = [
        {"node": n["name"], "type": n["type"], "similarity": round(n["score"], 3)}
        for n in top_nodes
    ]

    return {"answer": answer, "context": context_data}
