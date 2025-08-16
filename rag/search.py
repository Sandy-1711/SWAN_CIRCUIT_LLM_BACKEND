import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pprint import pprint

# ======================
# Config
# ======================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBED_FILE = "node_embeddings.npy"
META_FILE = "node_metadata.json"
DATA_FILE = "data/dataset.json"
TOP_K = 5

# ======================
# Load Data
# ======================
def load_data():
    embeddings = np.load(EMBED_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, metadata, chunks

# ======================
# Build FAISS Index
# ======================
def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ======================
# Embed Query
# ======================
def embed_query(query: str, model: SentenceTransformer):
    return model.encode([query])[0]

# ======================
# Retrieve Top-k Nodes
# ======================
def search(query: str, index, embeddings, metadata, chunks, model, top_k=TOP_K):
    query_vector = embed_query(query, model).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = []
    seen_chunk_ids = set()
    for idx in indices[0]:
        node_meta = metadata[idx]
        chunk_id = node_meta["chunk_id"]
        node_type = node_meta["type"]
        node_id = node_meta["node_id"]

        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            chunk_idx = int(chunk_id.split("_")[-1])
            chunk = chunks[chunk_idx]

            results.append({
                "matched_node": node_type,
                "node_id": node_id,
                "chunk_id": chunk_id,
                "score": float(distances[0][list(indices[0]).index(idx)]),
                "prompt": chunk.get("prompt", ""),
                "code": chunk.get("code", ""),
                "output": chunk.get("output", {}),
                "circuit_space": chunk.get("circuit_space_representation", "")
            })

    return results

# ======================
# Main
# ======================
    
def main():
    print("[INFO] Loading data...")
    embeddings, metadata, chunks = load_data()
    print(f"[INFO] Loaded {len(metadata)} node embeddings.")

    print("[INFO] Building FAISS index...")
    index = build_faiss_index(embeddings)

    model = SentenceTransformer(EMBEDDING_MODEL)

    while True:
        query = input("\n🔍 Enter your query (or type 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break

        print(f"\n[INFO] Searching for: '{query}'")
        results = search(query, index, embeddings, metadata, chunks, model)

        for i, r in enumerate(results):
            print(f"\nResult {i+1} (matched on: {r['matched_node']})")
            print("-" * 60)
            print(f"Chunk ID       : {r['chunk_id']}")
            print(f"Matched Node   : {r['node_id']} ({r['matched_node']})")
            print(f"Score          : {r['score']:.4f}")
            print(f"\n🧠 Prompt:\n{r['prompt']}")
            print(f"\n🧾 Code:\n{r['code']}")
            print(f"\n🔌 Output Connections:")
            pprint(r['output'].get("connections", []), indent=2)
            print(f"\n🔧 Parts:")
            pprint([p.get("type") for p in r['output'].get("parts", [])], indent=2)
            print(f"\n📐 Circuit Space:\n{r['circuit_space']}")
            print("-" * 60)

if __name__ == "__main__":
    main()
