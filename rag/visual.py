import streamlit as st
import numpy as np
import json
import faiss
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from typing import List, Dict
from matplotlib.patches import Rectangle

# ======================
# Config
# ======================
EMBED_FILE = "node_embeddings.npy"
META_FILE = "node_metadata.json"
DATA_FILE = "dataset.json"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3

# ======================
# Load data
# ======================
@st.cache_data
def load_embeddings_and_meta():
    embeddings = np.load(EMBED_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, metadata, chunks

# ======================
# Build FAISS index
# ======================
def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ======================
# Search function
# ======================
def search(query: str, index, embeddings, metadata, chunks, model, top_k=TOP_K):
    query_vec = model.encode([query])[0].reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)

    results = []
    seen_chunks = set()

    for i, idx in enumerate(indices[0]):
        node_meta = metadata[idx]
        chunk_id = node_meta["chunk_id"]
        node_type = node_meta["type"]
        node_id = node_meta["node_id"]

        if chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            chunk_idx = int(chunk_id.split("_")[-1])
            chunk = chunks[chunk_idx]

            results.append({
                "matched_node": node_type,
                "node_id": node_id,
                "chunk_id": chunk_id,
                "score": float(distances[0][i]),
                "prompt": chunk.get("prompt", ""),
                "code": chunk.get("code", ""),
                "output": chunk.get("output", {}),
                "circuit_space_representation": chunk.get("circuit_space_representation", "")
            })

    return results, query_vec

# ======================
# Visualize Embeddings in 2D Space
# ======================
def visualize_tsne(embeddings_2d, selected_indices, query_point, labels):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all embeddings as faint gray
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], color='lightgray', alpha=0.4)

    # Highlight top-K retrieved chunks as rectangles
    for i in selected_indices:
        ax.add_patch(Rectangle(
            (embeddings_2d[i][0] - 0.5, embeddings_2d[i][1] - 0.5),
            1.0, 1.0, linewidth=2, edgecolor='blue', facecolor='none'
        ))
        ax.text(embeddings_2d[i][0], embeddings_2d[i][1], labels[i], fontsize=9, color='blue')

    # Plot query vector
    ax.scatter(query_point[0], query_point[1], color='red', s=100, marker='x', label='Query')

    ax.set_title("t-SNE Projection of Chunk Embeddings")
    ax.legend()
    st.pyplot(fig)

# ======================
# Main Streamlit App
# ======================
def main():
    st.set_page_config(page_title="Chunk Search & Visualization", layout="wide")
    st.title("🔍 Chunk Search + Graph Embedding Visualization")

    embeddings, metadata, chunks = load_embeddings_and_meta()
    index = build_faiss_index(embeddings)
    model = SentenceTransformer(MODEL_NAME)

    query = st.text_input("Enter your search query:")

    if query:
        with st.spinner("Searching..."):
            results, query_vector = search(query, index, embeddings, metadata, chunks, model, TOP_K)

            st.subheader(f"Top {TOP_K} Retrieved Chunks")

            for i, r in enumerate(results):
                st.markdown(f"### 🔹 Result {i+1} — Chunk ID: `{r['chunk_id']}` (Matched on `{r['matched_node']}`)")
    
                st.markdown("**🟡 Prompt**")
                st.code(r['prompt'], language="text")
    
                st.markdown("**🟢 Code**")
                st.code(r['code'], language="cpp")

                st.markdown("**🟣 Circuit Space Representation**")
                st.code(r['circuit_space_representation'], language="text")
    
                st.markdown("**🔵 Output**")
                output_text = r["output"]
                if isinstance(output_text, dict):
                    output_text = json.dumps(output_text, indent=2)
                st.code(output_text, language="json")
    
                


            # Run t-SNE on combined embeddings
            st.subheader("📊 t-SNE Visualization of Embedding Space")

            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            all_embeddings = np.vstack([embeddings, query_vector])
            all_2d = tsne.fit_transform(all_embeddings)

            query_2d = all_2d[-1]  # last one is query
            labels = [meta["chunk_id"].split("_")[-1] for meta in metadata]

            top_indices = [metadata.index(m) for m in metadata if m["chunk_id"] in [r["chunk_id"] for r in results]]

            visualize_tsne(all_2d[:-1], top_indices, query_2d, labels)

if __name__ == "__main__":
    main()
