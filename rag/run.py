import os
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ======================
# Config
# ======================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBED_FILE = "node_embeddings.npy"
META_FILE = "node_metadata.json"
DATA_FILE = "data/dataset.json"
TOP_K = 5


# ======================
# Internal: I/O helpers
# ======================
def _ensure_files_exist():
    # Minimal guards to avoid surprises on first run.
    if not os.path.exists(EMBED_FILE):
        raise FileNotFoundError(f"Missing {EMBED_FILE}")
    if not os.path.exists(META_FILE):
        raise FileNotFoundError(f"Missing {META_FILE}")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Missing {DATA_FILE}")


def _load_data():
    _ensure_files_exist()
    embeddings = np.load(EMBED_FILE).astype(np.float32)
    with open(META_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return embeddings, metadata, chunks


def _save_embeddings(embeds: np.ndarray):
    # Overwrite is fine; format remains contiguous .npy
    np.save(EMBED_FILE, embeds.astype(np.float32))


def _save_json(path: str, obj):
    print(path)
    # os.makedirs(path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ======================
# Internal: Build Index
# ======================
def _build_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# ======================
# Internal: Chunk→Nodes
# ======================
def _chunk_to_nodes(chunk: Dict, chunk_idx: int) -> List[Tuple[str, str, str, str]]:
    """
    Returns a list of nodes:
    Each node: (node_id, type, chunk_id, text)
    """
    chunk_id = f"chunk_{chunk_idx}"
    nodes: List[Tuple[str, str, str, str]] = [
        (f"{chunk_id}_prompt", "Prompt", chunk_id, chunk.get("prompt", "")),
        (f"{chunk_id}_code", "Code", chunk_id, chunk.get("code", "")),
    ]

    # Parts summary
    parts = chunk.get("output", {}).get("parts", [])
    parts_text = ", ".join(p.get("type", "") for p in parts)
    nodes.append((f"{chunk_id}_parts", "Parts", chunk_id, parts_text))

    # Connections summary
    conns = chunk.get("output", {}).get("connections", [])
    conn_text = "\n".join(
        " → ".join(map(str, c[:2]))
        for c in conns
        if isinstance(c, list) and len(c) >= 2
    )
    nodes.append((f"{chunk_id}_output", "Output", chunk_id, conn_text))

    # Circuit space
    circuit = chunk.get("circuit_space_representation", "")
    nodes.append((f"{chunk_id}_circuit", "Circuit_Space", chunk_id, circuit))

    return nodes


# ======================
# Runtime: load once
# ======================
embeddings, metadata, chunks = _load_data()
index = _build_index(embeddings)
model = SentenceTransformer(EMBEDDING_MODEL)


# ======================
# Public: Query Function
# ======================
def query(text: str, top_k: int = TOP_K, distance_threshold: float = 0.4) -> List[Dict]:
    query_vector = model.encode([text])[0].astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    # print(distances, indices)  # debug if needed

    results = []
    seen_chunk_ids = set()
    for i, idx in enumerate(indices[0]):
        distance = float(distances[0][i])
        if distance > distance_threshold:
            continue  # Skip weak matches

        node_meta = metadata[idx]
        chunk_id = node_meta["chunk_id"]
        node_type = node_meta["type"]
        node_id = node_meta["node_id"]

        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        # chunk index is the trailing integer of "chunk_{idx}"
        try:
            chunk_idx = int(chunk_id.split("_")[-1])
        except Exception:
            # Fallback: if naming deviates, just skip safely
            continue
        if 0 <= chunk_idx < len(chunks):
            ch = chunks[chunk_idx]
        else:
            # Out-of-range safety
            continue

        results.append(
            {
                "matched_node": node_type,
                "node_id": node_id,
                "chunk_id": chunk_id,
                "score": distance,
                "prompt": ch.get("prompt", ""),
                "code": ch.get("code", ""),
                "output": ch.get("output", {}),
                "circuit_space": ch.get("circuit_space_representation", ""),
            }
        )

    return results


# ======================
# Public: Append/ingest a single feedback chunk
# ======================
def ingest_feedback_chunk(
    chunk: Dict,
    chunk_idx: Optional[int] = None,
) -> Dict:
    """
    Ingest ONE feedback 'chunk' (same schema as items in feedback.json).
    - Converts the chunk to 5 nodes (Prompt, Code, Parts, Output, Circuit_Space)
    - Embeds nodes using SentenceTransformer
    - Appends to on-disk files: node_embeddings.npy, node_metadata.json, data/dataset.json
    - Updates in-memory globals (embeddings, metadata, chunks) and FAISS index
    Returns a summary dict.
    """
    global embeddings, metadata, chunks, index, model

    # Determine where to place this chunk: default = append at end
    if chunk_idx is None:
        chunk_idx = len(chunks)
    chunk_id = f"chunk_{chunk_idx}"

    # 1) Build nodes & embeddings
    nodes = _chunk_to_nodes(chunk, chunk_idx)
    texts = [n[3] for n in nodes]
    new_embeds = model.encode(texts, show_progress_bar=False)
    new_embeds = np.asarray(new_embeds, dtype=np.float32)

    # 2) Update in-memory first
    #    (so a crash after disk write is less likely to leave FAISS behind,
    #     but we will persist all three: embeddings, metadata, dataset)
    embeddings = np.vstack([embeddings, new_embeds])
    for node_id, node_type, ch_id, _text in nodes:
        metadata.append({"node_id": node_id, "type": node_type, "chunk_id": ch_id})
    # Put/replace the chunk at chunk_idx (pad if someone passed a future index)
    if chunk_idx == len(chunks):
        chunks.append(chunk)
    elif 0 <= chunk_idx < len(chunks):
        # Replace if you're intentionally overwriting
        chunks[chunk_idx] = chunk
    else:
        # If an index gap is given, pad with empty dicts to keep alignment
        while len(chunks) < chunk_idx:
            chunks.append({})
        chunks.append(chunk)

    # 3) Update FAISS in-memory
    index.add(new_embeds)

    # 4) Persist to disk
    _save_embeddings(embeddings)
    _save_json(META_FILE, metadata)
    _save_json(DATA_FILE, chunks)

    return {
        "status": "ok",
        "chunk_id": chunk_id,
        "chunk_index": chunk_idx,
        "new_nodes_added": len(nodes),
        "total_nodes": int(embeddings.shape[0]),
        "total_chunks": len(chunks),
    }


# ======================
# Utility: context filter
# ======================
def filter_rag_context(chunks_list: List[Dict], fields: List[str]):
    return [{key: ch[key] for key in fields if key in ch} for ch in chunks_list]

