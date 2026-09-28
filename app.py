import streamlit as st
from utils.graph import KnowledgeGraph
from utils.extractor import extract_graph
from utils.search import semantic_search
from utils.qa import answer_question

st.set_page_config(
    page_title="AI Knowledge Graph — Pure Python",
    page_icon="🕸️",
    layout="wide"
)

# ── Init session state ────────────────────────────────────────────────────────
if "kg" not in st.session_state:
    st.session_state["kg"] = KnowledgeGraph()

kg: KnowledgeGraph = st.session_state["kg"]

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🕸️ AI Knowledge Graph")
    st.caption("Pure Python · NetworkX · Sentence Transformers · Claude API")

    api_key = st.text_input("Anthropic API Key", type="password",
                            help="Get one at console.anthropic.com")
    if api_key:
        st.session_state["api_key"] = api_key

    st.divider()
    st.markdown("**Quick load sample data**")
    sample = st.selectbox("Dataset", ["— pick one —", "Tech Companies", "Indian Startups", "Bollywood"])
    if sample != "— pick one —" and st.button("Load", use_container_width=True):
        from data.samples import SAMPLES
        kg.clear()
        for item in SAMPLES[sample]:
            kg.add_entity(item["name"], item["type"])
        for rel in SAMPLES[sample + "_rels"]:
            kg.add_relation(rel["from"], rel["to"], rel["rel"])
        kg.rebuild_embeddings()
        st.success(f"'{sample}' loaded ✓")

    st.divider()
    stats = kg.get_stats()
    st.metric("Entities", stats["entities"])
    st.metric("Relations", stats["relations"])
    if st.button("🗑️ Clear Graph", use_container_width=True):
        kg.clear()
        st.success("Graph cleared.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Build Graph", "🔍 Ask Questions", "🔎 Semantic Search",
    "🗺️ Explore", "📖 How It Works"
])

# ── TAB 1: Build Graph ────────────────────────────────────────────────────────
with tab1:
    st.header("Build your knowledge graph from text")
    st.caption("Paste any text — Claude extracts entities & relations and adds them to the graph.")

    text_input = st.text_area(
        "Raw text",
        height=180,
        placeholder="e.g. Sundar Pichai is the CEO of Google. Google was founded by Larry Page and Sergey Brin in 1998."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        extract_btn = st.button("⚡ Extract & Add", use_container_width=True, type="primary")

    if extract_btn and text_input:
        if not st.session_state.get("api_key"):
            st.warning("Enter your Anthropic API key in the sidebar.")
        else:
            with st.spinner("Extracting with Claude..."):
                try:
                    result = extract_graph(text_input, st.session_state["api_key"])
                    for ent in result["entities"]:
                        kg.add_entity(ent["name"], ent["type"])
                    for rel in result["relations"]:
                        kg.add_relation(rel["from"], rel["to"], rel["relation"])
                    kg.rebuild_embeddings()
                    st.success(
                        f"Added **{len(result['entities'])} entities** and "
                        f"**{len(result['relations'])} relations**."
                    )
                    with st.expander("See extracted data"):
                        st.json(result)
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

    st.divider()
    st.subheader("Add manually")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Add entity**")
        ent_name = st.text_input("Name", key="ent_name")
        ent_type = st.selectbox("Type", ["Person", "Company", "Product", "Place", "Event", "Technology", "Other"])
        if st.button("Add Entity"):
            if ent_name:
                kg.add_entity(ent_name, ent_type)
                kg.rebuild_embeddings()
                st.success(f"Added '{ent_name}' ({ent_type})")
    with c2:
        st.markdown("**Add relation**")
        from_node = st.text_input("From", key="from_node")
        to_node   = st.text_input("To",   key="to_node")
        relation  = st.text_input("Relation (e.g. founded, works at)", key="rel_text")
        if st.button("Add Relation"):
            if from_node and to_node and relation:
                kg.add_relation(from_node, to_node, relation)
                st.success(f"{from_node} —[{relation}]→ {to_node}")

# ── TAB 2: Q&A ────────────────────────────────────────────────────────────────
with tab2:
    st.header("Ask questions in plain English")
    st.caption("Semantic search finds relevant nodes → Claude reasons over the subgraph → answer.")

    question = st.text_input("Your question",
                             placeholder="Who founded Google? Which companies are in Bengaluru?")

    if st.button("🔍 Ask", type="primary"):
        if not st.session_state.get("api_key"):
            st.warning("Enter your Anthropic API key.")
        elif kg.get_stats()["entities"] == 0:
            st.warning("Add some data first (Build Graph tab or load a sample).")
        elif not question:
            st.warning("Enter a question.")
        else:
            with st.spinner("Searching graph + generating answer..."):
                try:
                    result = answer_question(question, kg, st.session_state["api_key"])
                    st.markdown("### 💬 Answer")
                    st.success(result["answer"])
                    with st.expander("📦 Context used from graph"):
                        st.json(result["context"])
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()
    st.markdown("**Try these:**")
    for q in [
        "Who is the CEO of Google?",
        "Which companies are headquartered in Bengaluru?",
        "What products has Apple developed?",
        "Who co-founded Google?",
    ]:
        st.markdown(f"- *{q}*")

# ── TAB 3: Semantic Search ────────────────────────────────────────────────────
with tab3:
    st.header("Semantic node search")
    st.caption("Find nodes by meaning, not exact name — powered by sentence-transformers embeddings.")

    query = st.text_input("Search query", placeholder="e.g. tech billionaire, ride sharing app")
    top_k = st.slider("Top K results", 1, 10, 5)

    if st.button("🔎 Search", type="primary"):
        if kg.get_stats()["entities"] == 0:
            st.warning("Add some data first.")
        elif not query:
            st.warning("Enter a search query.")
        else:
            with st.spinner("Computing similarity..."):
                try:
                    results = semantic_search(query, kg, top_k=top_k)
                    st.markdown("### Results")
                    for i, r in enumerate(results, 1):
                        score_pct = round(r["score"] * 100, 1)
                        st.markdown(
                            f"**{i}. {r['name']}** `{r['type']}` — "
                            f"similarity: `{score_pct}%`"
                        )
                        # Show neighbours
                        neighbours = kg.get_neighbours(r["name"])
                        if neighbours:
                            nb_str = ", ".join(
                                [f"{nb['relation']} → **{nb['node']}**" for nb in neighbours[:4]]
                            )
                            st.caption(f"Connected via: {nb_str}")
                except Exception as e:
                    st.error(f"Search failed: {e}")

# ── TAB 4: Explore ────────────────────────────────────────────────────────────
with tab4:
    st.header("Graph explorer")

    if kg.get_stats()["entities"] == 0:
        st.info("No data yet — add entities in Build Graph or load a sample.")
    else:
        import pandas as pd

        col1, col2, col3 = st.columns(3)
        stats = kg.get_stats()
        col1.metric("Entities", stats["entities"])
        col2.metric("Relations", stats["relations"])
        col3.metric("Entity types", stats["types"])

        st.divider()
        st.subheader("Entities")
        ents = kg.get_all_entities()
        if ents:
            st.dataframe(pd.DataFrame(ents), use_container_width=True, hide_index=True)

        st.subheader("Relations")
        rels = kg.get_all_relations()
        if rels:
            st.dataframe(pd.DataFrame(rels), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Multi-hop path finder")
        all_names = [e["Name"] for e in ents]
        src = st.selectbox("From entity", all_names, key="path_src")
        tgt = st.selectbox("To entity",   all_names, key="path_tgt")
        if st.button("Find path"):
            path = kg.find_path(src, tgt)
            if path:
                st.success(" → ".join(path))
            else:
                st.info("No path found between these two entities.")

# ── TAB 5: How It Works ──────────────────────────────────────────────────────
with tab5:
    st.header("How it works — Pure Python edition")
    st.markdown("""
### Architecture

```
Raw Text
   │
   ▼
Claude API ──► Entity + Relation Extraction (JSON)
   │
   ▼
NetworkX DiGraph ──► In-memory property graph
   │                 (nodes = entities, edges = relations)
   ▼
sentence-transformers ──► Node embeddings (all-MiniLM-L6-v2)
   │                       stored as numpy arrays
   ▼
Semantic Search (cosine similarity) + Multi-hop traversal
   │
   ▼
Claude API ──► Reason over retrieved subgraph → Answer
   │
   ▼
Streamlit UI
```

### Key files

| File | Role |
|---|---|
| `utils/graph.py` | NetworkX graph — add/query nodes & edges, multi-hop traversal |
| `utils/extractor.py` | Claude API call — text → JSON entities + relations |
| `utils/search.py` | Cosine similarity search over sentence-transformer embeddings |
| `utils/qa.py` | Retrieves subgraph context → Claude generates answer |
| `data/samples.py` | Pre-built datasets for demo |

### Why no Neo4j here?

This version runs **100% in Python memory** — no database to set up.  
Perfect for learning internals. For production, swap NetworkX → Neo4j (Option A).

### Core Python libraries

- `networkx` — graph structure + traversal algorithms
- `sentence-transformers` — semantic embeddings (no OpenAI key needed)
- `numpy` — cosine similarity computation
- `streamlit` — UI

### Extend this project

1. **Persist the graph** — `nx.write_graphml()` to save/load to disk
2. **Add Neo4j** — port to Option A for production scale
3. **Link prediction** — train a simple model on node embeddings
4. **PDF/URL ingestion** — auto-build graph from documents
    """)
