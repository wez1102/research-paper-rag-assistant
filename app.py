"""
Step 5 (original version): a simple Streamlit UI.

    streamlit run app.py

The main web app is now server.py + static/ (model picker for local Ollama
models, streaming answers, footnote citations). This page is kept because it's
the quickest thing to hack on and it has its own headless test.

Type a question, pick a retrieval mode, and get an answer with
[paper title, section] citations. The "Retrieved chunks" panel shows exactly what
the model was allowed to read, which is the quickest way to see *why* an answer
is good or bad.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from generate import answer_question, resolve_provider, model_for
from retrieve import MODES, Retriever

st.set_page_config(page_title="Research Paper RAG Assistant", page_icon="📄", layout="wide")

MODE_LABELS = {"dense": "Dense", "hybrid": "Hybrid (BM25 + dense)", "hybrid_rerank": "Hybrid + rerank"}

# Picked because they work across the whole corpus (no paper filter needed).
EXAMPLES = [
    "Which datasets are used for Armenian named entity recognition?",
    "How do the authors handle noisy crowd annotations for Chinese NER?",
    "What languages are used for zero-shot neural machine translation?",
    "What metric correlates best with human judgements for table-to-text generation?",
]


@st.cache_resource(show_spinner="Loading the index and models (first run only)...")
def get_retriever() -> Retriever:
    # cache_resource means the embedding model, BM25 index and reranker are
    # loaded once per server, not on every button click.
    return Retriever(load_reranker=True)


@st.cache_data
def load_results() -> pd.DataFrame | None:
    path = settings.results_dir / "results.csv"
    return pd.read_csv(path) if path.exists() else None


try:
    retriever = get_retriever()
except RuntimeError as err:
    st.error(str(err))
    st.stop()

provider = resolve_provider()

# --- Sidebar: settings ------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Retrieval mode", MODES, index=2, format_func=MODE_LABELS.get,
                    help="Dense = embeddings only. Hybrid adds BM25 keyword search. "
                         "Rerank re-scores the top 30 with a cross-encoder.")
    k = st.slider("Chunks to retrieve (k)", 3, 10, settings.top_k)

    papers = retriever.papers()
    options = ["All papers"] + [title for _, title in papers]
    choice = st.selectbox("Limit to one paper", options)
    paper_id = None if choice == "All papers" else dict((t, p) for p, t in papers)[choice]

    st.divider()
    st.caption(f"LLM: **{provider}**" + (f" · `{model_for(provider)}`" if provider != "extractive" else ""))
    if provider == "extractive":
        st.caption("No LLM configured, so answers are the most relevant sentences quoted "
                   "from the papers. Set an API key or run Ollama for written answers.")
    st.caption(f"Index: {len(retriever.ids):,} chunks from {len(papers)} QASPER papers")

    results = load_results()
    if results is not None:
        with st.expander("Retrieval benchmark"):
            view = results[results["scope"] == "corpus"][["mode", "hit@1", "hit@5", "mrr"]]
            st.dataframe(view.set_index("mode").style.format("{:.2f}"), use_container_width=True)

# --- Main panel ---------------------------------------------------------------
st.title("📄 Research Paper RAG Assistant")
st.write("Ask a question about 50 NLP papers from the QASPER dataset. Every claim in the answer "
         "is cited as **[paper title, section]**, and you can inspect exactly what was retrieved.")

# Example buttons just fill the text box via session state.
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state["question"] = example

question = st.text_input("Your question", key="question", placeholder="e.g. Which baselines do they compare against?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the papers..."):
        try:
            ans = answer_question(question.strip(), retriever, mode=mode, k=k, paper_id=paper_id)
        except Exception as err:   # network / API problems shouldn't crash the page
            st.error(f"Something went wrong while generating the answer: {err}")
            st.stop()

    st.subheader("Answer")
    if ans.refused:
        st.warning(ans.text)
    else:
        st.markdown(ans.text)

    t = ans.timings
    st.caption(f"Retrieval {t.get('retrieval_s', 0):.2f}s · generation {t.get('generation_s', 0):.2f}s · "
               f"mode: {MODE_LABELS[mode]} · LLM: {ans.provider}")

    with st.expander(f"Retrieved chunks ({len(ans.chunks)})", expanded=False):
        for i, c in enumerate(ans.chunks, start=1):
            marker = "✅ cited" if i in ans.cited else ""
            st.markdown(f"**[{i}] {c.title}** — *{c.section}*  \n"
                        f"`{c.score_type} score: {c.score:.3f}` {marker}")
            st.write(c.text)
            st.divider()
