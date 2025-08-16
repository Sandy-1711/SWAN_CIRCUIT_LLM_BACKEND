from models.shared_llms import generator_llm_2048


EXAMPLE_JSON = (
    '{"parts":[{"id":"esp","type":"wokwi-esp8266"},'
    '{"id":"dht","type":"wokwi-dht11"},'
    '{"id":"led1","type":"wokwi-led","attrs":{"color":"red"}},'
    '{"id":"bb1","type":"wokwi-breadboard"}],'
    '"connections":[["esp:3V3","bb1:tp.36","red",["v0"]]]}'
)
GENERATOR_PROMPT = (
    "You are an **experienced Arduino Systems Engineer**.\n"
    "Given the following circuit specification, produce **only** a JSON object\n"
    "with *two* top‑level keys: `parts` (array) and `connections` (array).\n"
    "Each element of `parts` must have `id`, `type` and optional `attrs`.\n"
    "Each element of `connections` must be an array⃰ of the form\n"
    "[from, to, color, path]. Use the same IDs as in `parts`.\n\n"
    "Return strictly valid JSON — no markdown, code fences, or commentary.\n"
    "If any attribute is missing, infer sensible defaults.\n\n"
    "### Example format\n" + EXAMPLE_JSON.replace("{", "{{").replace("}", "}}") + "\n\n"
    "Now read the specification and output the JSON: \n"
    "{specification} "
)
GENERATOR_PROMPT_WITH_CONTEXT = (
    "You are an **experienced Arduino Systems Engineer**.\n"
    "Given the following circuit specification, produce **only** a JSON object\n"
    "with *two* top‑level keys: `parts` (array) and `connections` (array).\n"
    "Each element of `parts` must have `id`, `type` and optional `attrs`.\n"
    "Each element of `connections` must be an array⃰ of the form\n"
    "[from, to, color, path]. Use the same IDs as in `parts`.\n\n"
    "**Ignore it if it does not apply. Do not copy irrelevant information.**\n\n"
    "Return strictly valid JSON — no markdown, code fences, or commentary.\n"
    "If any attribute is missing, infer sensible defaults.\n\n"
    "### Example format\n" + EXAMPLE_JSON.replace("{", "{{").replace("}", "}}") + "\n\n"
    "Now read the specification and output the JSON: \n"
    "{specification} "
    "You may optionally refer to this **REFERENCE CONTEXT** for inspiration if relevant:\n"
    "<<< {context} >>>\n"
)

def generate_json_stream(llm, specification: str, context: str):
    # if context != "":
    #     message = GENERATOR_PROMPT_WITH_CONTEXT.format(
    #         specification=specification, context=context
    #     ).strip()
    # else:
    #     message = GENERATOR_PROMPT.format(specification=specification).strip()
    message = GENERATOR_PROMPT.format(specification=specification).strip()
    print(message)
    chat_input = f"<|im_start|>system\n{message}\n<|im_end|>\n<|im_start|>assistant\n"

    for chunk in llm(prompt=chat_input, max_tokens=1024, stream=True):
        yield chunk.get("choices", [{}])[0].get("text", "")


def generate(ir: str, context: str):
    output = ""
    yield {"stage": "json_start"}
    for token in generate_json_stream(generator_llm_2048, ir, context):
        yield {"stage": "json_progress", "token": token}
        output += token
    yield {"stage": "json_done", "output": output}
