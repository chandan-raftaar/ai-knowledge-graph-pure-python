# 🕸️ AI Knowledge Graph — Pure Python

A Streamlit app that builds and queries a **knowledge graph** using only Python — **no database, no Docker, no cloud setup** required.

> **Built by Chandan M** | [GitHub](https://github.com/chandan-raftaar) | Part of DS portfolio

---

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)

---

## 🧠 What it does

| Feature | Description |
|---|---|
| **Entity Extraction** | Paste raw text → Claude extracts People, Companies, Places, Events |
| **In-memory Graph** | NetworkX DiGraph — nodes = entities, edges = typed relations |
| **Semantic Embeddings** | Every node encoded with `sentence-transformers/all-MiniLM-L6-v2` |
| **Semantic Search** | Find nodes by *meaning*, not exact name — cosine similarity over embeddings |
| **Natural Language Q&A** | Ask questions → relevant subgraph retrieved → Claude explains |
| **Multi-hop Path Finder** | Shortest path between any two entities |
| **Sample Datasets** | Tech Companies · Indian Startups · Bollywood — one click to load |

---

## 🏗️ Architecture

```
Raw Text
   │
   ▼
Claude API ──► Entity + Relation Extraction (JSON)
   │
   ▼
NetworkX DiGraph ──► In-memory property graph
   │
   ▼
sentence-transformers ──► Node embeddings (all-MiniLM-L6-v2)
   │
   ▼
Cosine Similarity Search + 1-hop Subgraph Expansion
   │
   ▼
Claude API ──► Answer grounded in retrieved context
   │
   ▼
Streamlit UI (5 tabs)
```

---

## 🛠️ Tech Stack

| Library | Role |
|---|---|
| `networkx` | Graph structure, traversal, shortest path |
| `sentence-transformers` | Node embeddings — semantic meaning |
| `numpy` | Cosine similarity computation |
| `streamlit` | UI |
| Claude API | Entity extraction + answer generation (pure `urllib`) |

**No Neo4j. No LangChain. No Docker.** Runs entirely in Python memory.

---

## ⚡ Quick Start

### 1. Clone
```bash
git clone https://github.com/chandan-raftaar/ai-knowledge-graph-pure-python.git
cd ai-knowledge-graph-pure-python
```

### 2. Install
```bash
pip install -r requirements.txt
```

> First run downloads the `all-MiniLM-L6-v2` model (~90 MB). Subsequent runs are instant.

### 3. Get Anthropic API key
Sign up at [console.anthropic.com](https://console.anthropic.com)

### 4. Run
```bash
streamlit run app.py
```

### 5. In the app sidebar
- Paste your Anthropic API key
- Load a sample dataset or paste your own text

---

## 📁 Project Structure

```
ai-knowledge-graph-pure-python/
├── app.py                 # Streamlit UI — 5 tabs
├── requirements.txt
├── README.md
├── utils/
│   ├── graph.py           # NetworkX KG — CRUD, embeddings, traversal
│   ├── extractor.py       # Claude: text → JSON entities + relations
│   ├── search.py          # Cosine similarity semantic search
│   └── qa.py              # Subgraph retrieval + Claude answer
└── data/
    └── samples.py         # Tech Companies, Indian Startups, Bollywood
```

---

## 💡 Sample Questions

Load **Indian Startups** and try:
- *Who founded CRED?*
- *Which startups are based in Bengaluru?*
- *Who invested in Razorpay?*
- *What is the relation between Walmart and Flipkart?*

Or try **Semantic Search** with: `"ride sharing company"`, `"payment startup"`, `"Bollywood director"`

---

## 🔭 What's Next (planned)

- [ ] Persist graph to disk with `nx.write_graphml()`
- [ ] PDF/URL ingestion — auto-build graph from documents
- [ ] Link prediction — infer missing relations from embeddings
- [ ] Interactive visualization with `pyvis`
- [ ] Port to **Option A** (Neo4j) for production scale

---

## 🆚 Option A vs Option B

| | This repo (Option B) | Option A (Neo4j) |
|---|---|---|
| Setup | `pip install` only | Neo4j AuraDB account |
| Storage | Python memory | Persistent graph DB |
| Query | Cosine similarity + NetworkX | Cypher queries |
| Scale | Small-medium graphs | Production scale |
| Best for | Learning internals | Portfolio / interviews |

---

## 📊 Skills Demonstrated

`NetworkX` · `Graph Algorithms` · `NLP` · `Sentence Transformers` · `Embeddings` · `Semantic Search` · `RAG` · `Anthropic Claude API` · `Streamlit` · `Python`

---

## 🤝 Connect

Built as part of my Data Science portfolio transition from Oracle MTS.  
Open to DS / ML Engineer roles in Bengaluru.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/your-profile)
