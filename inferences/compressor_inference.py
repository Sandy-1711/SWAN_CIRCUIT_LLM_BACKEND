from models.shared_llms import compressor_llm_2048


COMPRESSOR_PROMPT = (
    "You are an **experienced Arduino Systems Engineer**.\n"
    "Given:\n"
    "  • User prompt: {user_prompt}\n"
    "  • Arduino code:\n{code}\n\n"
    "Generate a **compressed hardware spec** in *Wokwi* nomenclature using **_exactly_** "
    "this scaffold (do not add / remove headers or blank lines):\n"
    "<<=components=>>\n"
    "<<=connections=>>\n"
    "<<=attrs=>>\n\n"
    "• **components** - one per line as `<id>:<wokwi-part-id>`\n"
    "• **connections** - one per line as `<src> <dst>`\n"
    "• **attrs** - optional key-value extras, one per line as `<id> <key>:<value>`\n"
    "Capture every pin / wiring detail needed to reproduce the circuit, "
    "omit text that is not required for the diagram.\n "
)
COMPRESSOR_PROMPT_WITH_CONTEXT = (
    "You are an **experienced Arduino Systems Engineer**.\n"
    "Given:\n"
    "  • User prompt: {user_prompt}\n"
    "  • Arduino code:\n{code}\n\n"
    "**Only refer to this if directly relevant. Ignore it otherwise.**\n\n"
    "Generate a **compressed hardware spec** in *Wokwi* nomenclature using **_exactly_** "
    "this scaffold (do not add / remove headers or blank lines):\n"
    "<<=components=>>\n"
    "<<=connections=>>\n"
    "<<=attrs=>>\n\n"
    "• **components** - one per line as `<id>:<wokwi-part-id>`\n"
    "• **connections** - one per line as `<src> <dst>`\n"
    "• **attrs** - optional key-value extras, one per line as `<id> <key>:<value>`\n"
    "Capture every pin / wiring detail needed to reproduce the circuit, "
    "omit text that is not required for the diagram.\n "
    "**REFERENCE CONTEXT (Optional):**"
    "If relevant, you may refer to the following example for inspiration. However, do NOT copy it or include any of its logic unless it directly applies to the user request:"
    "<<< {context} >>>"
)


def compress_to_ir_stream(llm, prompt: str, code: str, context: str):
    if context != "":
        message = COMPRESSOR_PROMPT_WITH_CONTEXT.format(
            user_prompt=prompt, code=code, context=context
        ).strip()
    else:
        message = COMPRESSOR_PROMPT.format(
            user_prompt=prompt,
            code=code,
        ).strip()

    user_block = f"{prompt}\n\n{code}"
    chat_input = (
        f"<|im_start|>system\n{message}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_block}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    for chunk in llm(prompt=chat_input, max_tokens=1024, stream=True):
        yield chunk.get("choices", [{}])[0].get("text", "")


def generate(user_prompt: str, code: str, context: str):
    output = ""
    yield {"stage": "ir_start"}
    for token in compress_to_ir_stream(compressor_llm_2048, user_prompt, code, context):
        yield {"stage": "ir_progress", "token": token}
        output += token
    yield {"stage": "ir_done", "ir": output}
