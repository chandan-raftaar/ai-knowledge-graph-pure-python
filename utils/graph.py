"""
graph.py — In-memory knowledge graph backed by NetworkX.
Handles entities (nodes), relations (edges), multi-hop traversal,
and stores sentence-transformer embeddings on each node.
"""
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

_MODEL = None  # lazy-loaded once


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


class KnowledgeGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_entity(self, name: str, entity_type: str):
        """Add or update a node."""
        self.G.add_node(name, type=entity_type, embedding=None)

    def add_relation(self, from_name: str, to_name: str, relation: str):
        """Add an edge. Creates bare nodes if they don't exist."""
        if from_name not in self.G:
            self.G.add_node(from_name, type="Other", embedding=None)
        if to_name not in self.G:
            self.G.add_node(to_name, type="Other", embedding=None)
        self.G.add_edge(from_name, to_name, relation=relation)

    def clear(self):
        self.G.clear()

    # ── Embeddings ────────────────────────────────────────────────────────────

    def rebuild_embeddings(self):
        """Encode all node names and store embeddings on nodes."""
        model = _get_model()
        names = list(self.G.nodes())
        if not names:
            return
        vecs = model.encode(names, show_progress_bar=False, normalize_embeddings=True)
        for name, vec in zip(names, vecs):
            self.G.nodes[name]["embedding"] = vec

    def get_embedding(self, name: str) -> np.ndarray | None:
        return self.G.nodes[name].get("embedding") if name in self.G else None

    def all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Return (names, matrix) for all nodes that have embeddings."""
        names, vecs = [], []
        for n, data in self.G.nodes(data=True):
            emb = data.get("embedding")
            if emb is not None:
                names.append(n)
                vecs.append(emb)
        return names, np.array(vecs) if vecs else np.empty((0, 384))

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_all_entities(self) -> list[dict]:
        return [
            {"Name": n, "Type": d.get("type", "Other")}
            for n, d in sorted(self.G.nodes(data=True))
        ]

    def get_all_relations(self) -> list[dict]:
        return [
            {"From": u, "Relation": d.get("relation", ""), "To": v}
            for u, v, d in self.G.edges(data=True)
        ]

    def get_stats(self) -> dict:
        types = {d.get("type") for _, d in self.G.nodes(data=True) if d.get("type")}
        return {
            "entities": self.G.number_of_nodes(),
            "relations": self.G.number_of_edges(),
            "types": len(types),
        }

    def get_neighbours(self, name: str, hops: int = 1) -> list[dict]:
        """Return immediate neighbours with relation labels (both directions)."""
        if name not in self.G:
            return []
        results = []
        for _, tgt, data in self.G.out_edges(name, data=True):
            results.append({"node": tgt, "relation": data.get("relation", ""), "direction": "out"})
        for src, _, data in self.G.in_edges(name, data=True):
            results.append({"node": src, "relation": data.get("relation", ""), "direction": "in"})
        return results

    def get_subgraph_text(self, names: list[str]) -> str:
        """Return a readable text description of a subgraph for the LLM."""
        lines = []
        seen_edges = set()
        for name in names:
            node_type = self.G.nodes[name].get("type", "Other")
            lines.append(f"- {name} (type: {node_type})")
            for _, tgt, data in self.G.out_edges(name, data=True):
                key = (name, tgt)
                if key not in seen_edges:
                    lines.append(f"  → [{data.get('relation','')}] → {tgt}")
                    seen_edges.add(key)
            for src, _, data in self.G.in_edges(name, data=True):
                key = (src, name)
                if key not in seen_edges:
                    lines.append(f"  ← [{data.get('relation','')}] ← {src}")
                    seen_edges.add(key)
        return "\n".join(lines)

    def find_path(self, src: str, tgt: str) -> list[str] | None:
        """Find shortest path between two nodes."""
        try:
            return nx.shortest_path(self.G.to_undirected(), src, tgt)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
