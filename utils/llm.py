from models.shared_llms import base_llm
import re
def generate_chat_name(prompt: str):
    """Stream a 3-word descriptive title from the user prompt using a GGUF LLM."""
    yield {"stage": "naming_start", "prompt": prompt}

    prompt = prompt.strip()
    if len(prompt) < 5:
        yield {"stage": "naming_done", "name": "New Chat"}

    title_prompt = (
        "You are a title generator.\n"
        "Given a user request, respond with exactly **3 meaningful words** that summarize it.\n"
        "- No explanations.\n"
        "- No punctuation or formatting.\n"
        "- Do not return anything other than the 3 words.\n\n"
        f"User Request: {prompt}\n\n"
        "Title:"
    )

    name = ""
    for chunk in base_llm(title_prompt, max_tokens=5, stream=True):
        # name += token
        if chunk.get("choices", [{}])[0].get("text", "") == "\n":
            break
        name += chunk.get("choices", [{}])[0].get("text", "")
        yield {"stage": "naming_progress", "token": chunk.get("choices", [{}])[0].get("text", "")}
        name = name.replace('"', "")
        name = name.replace(".", "")
        name = name.replace(",", "")

    yield {"stage": "naming_done", "name": name.strip()}

    #     name_generated = ""
    #     for chunk in base_llm(title_prompt, max_tokens=1024, stream=True):
    #         if "choices" in chunk and chunk["choices"]:
    #             name_generated += chunk["choices"][0].get("text", "")

    #     # Normalize spaces and clean unwanted characters
    #     cleaned = re.sub(r"\s+", " ", name_generated).strip()
    #     cleaned = cleaned.replace('"', "").replace(".", "")

    #     words = cleaned.split()

    #     if len(words) >= 3:
    #         return " ".join(words[:3])
    #     elif words:
    #         return " ".join(words)
    #     else:
    #         return prompt[:20]
    # except Exception:
    #     return prompt[:20]