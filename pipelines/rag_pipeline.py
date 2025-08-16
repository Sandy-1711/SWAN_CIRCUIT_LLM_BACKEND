from inferences.coder_inference import generate as generate_code
from inferences.compressor_inference import generate as compress
from inferences.generator_inference import generate as generate_json

from rag.run import query

def invoke(prompt):
    results = query(prompt, 3)
    # return results[1:] if len(results) > 1 else []
    return results if len(results) > 0 else []


def filter_rag_context(chunks, fields):
    return [{key: chunk[key] for key in fields if key in chunk} for chunk in chunks]


def run_rag_pipeline(user_prompt: str):
    prompt = user_prompt

    # Coder RAG context
    coder_rag_context_raw = invoke(prompt)
    print(coder_rag_context_raw)
    if coder_rag_context_raw:
        best_score = min(item.get("score", 1.0) for item in coder_rag_context_raw)
    else:
        best_score = 1.0

    # ABORT LOGIC
    if best_score > 0.5:
        yield {
            "status": "abort",
            "reason": "No similar examples found in our dataset. Please try another query.",
            "score": best_score
        }
        return
    
    coder_rag_context = filter_rag_context(coder_rag_context_raw, ["prompt", "code"])
    yield {"stage": "rag_stage_1_done", "context": coder_rag_context}

    # Code generation (stream)
    code = ""
    for chunk in generate_code(prompt, coder_rag_context):
        yield chunk
        if chunk.get("stage") == "code_done":
            code = chunk["code"]

    # Compressor RAG context
    compressor_rag_context_raw = invoke(code)
    compressor_rag_context = filter_rag_context(
        compressor_rag_context_raw, ["prompt", "circuit_space"]
    )
    # print(compressor_rag_context)
    yield {"stage": "rag_stage_2_done", "context": compressor_rag_context}

    # IR generation (stream)
    ir = ""
    for chunk in compress(prompt, code, compressor_rag_context):
        yield chunk
        if chunk.get("stage") == "ir_done":
            ir = chunk["ir"]

    # Generator RAG context
    enriched_ir_raw = invoke(ir)
    enriched_ir = filter_rag_context(enriched_ir_raw, ["circuit_space", "output"])
    yield {"stage": "rag_stage_3_done", "context": enriched_ir}

    # JSON generation (stream)
    for chunk in generate_json(ir, enriched_ir):
        yield chunk
        if chunk.get("stage") == "json_done":
            code = chunk["output"]
