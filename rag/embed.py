import json
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

# ===========================================
# CONFIGURATION
# ===========================================
DATA_PATH = "data/dataset.json"  # Input JSON with all chunks
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_EMBED_FILE = "node_embeddings.npy"
OUTPUT_META_FILE = "node_metadata.json"

# ===========================================
# LOAD CHUNKS FROM JSON
# ===========================================
def load_chunks(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ===========================================
# CONVERT CHUNK TO GRAPH NODES
# Each chunk becomes 5 nodes:
# - Prompt
# - Code
# - Parts
# - Output (connections)
# - Circuit_Space
# ===========================================
def chunk_to_nodes(chunk: Dict, chunk_idx: int) -> List[Tuple[str, str, str, str]]:
    chunk_id = f"chunk_{chunk_idx}"
    nodes = []

    # Prompt node
    prompt = chunk.get("prompt", "")
    nodes.append((f"{chunk_id}_prompt", "Prompt", chunk_id, prompt))

    # Code node
    code = chunk.get("code", "")
    nodes.append((f"{chunk_id}_code", "Code", chunk_id, code))

    # Parts node (device types list)
    parts = chunk.get("output", {}).get("parts", [])
    parts_text = ", ".join(p.get("type", "") for p in parts)
    nodes.append((f"{chunk_id}_parts", "Parts", chunk_id, parts_text))

    # Output node (connection list)
    connections = chunk.get("output", {}).get("connections", [])
    conn_text = "\n".join(" → ".join(conn[:2]) for conn in connections)
    nodes.append((f"{chunk_id}_output", "Output", chunk_id, conn_text))

    # Circuit_Space node (raw multiline text block)
    circuit_space = chunk.get("circuit_space_representation", "")
    nodes.append((f"{chunk_id}_circuit", "Circuit_Space", chunk_id, circuit_space))

    return nodes

# ===========================================
# EMBED ALL NODES
# ===========================================
def embed_nodes(all_nodes: List[Tuple[str, str, str, str]],
                model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    texts = [node[3] for node in all_nodes]
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)

# ===========================================
# MAIN PIPELINE
# ===========================================
def main():
    print("[INFO] Loading data...")
    chunks = load_chunks(DATA_PATH)

    print(f"[INFO] Processing {len(chunks)} chunks into graph nodes...")
    all_nodes = []
    for idx, chunk in enumerate(chunks):
        all_nodes.extend(chunk_to_nodes(chunk, idx))

    print(f"[INFO] Total nodes to embed: {len(all_nodes)}")

    print("[INFO] Generating embeddings...")
    embeddings = embed_nodes(all_nodes)

    # Save embeddings to disk
    np.save(OUTPUT_EMBED_FILE, embeddings)

    # Save metadata (excluding raw text)
    metadata = [
        {
            "node_id": n[0],
            "type": n[1],
            "chunk_id": n[2]
        }
        for n in all_nodes
    ]
    with open(OUTPUT_META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✅ DONE] Saved {len(all_nodes)} node embeddings to '{OUTPUT_EMBED_FILE}'")
    print(f"[✅ DONE] Metadata written to '{OUTPUT_META_FILE}'")

if __name__ == "__main__":
    main()
