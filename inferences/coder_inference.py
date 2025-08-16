from models.shared_llms import coder_llm_2048

CODER_PROMPT = """You are an expert IoT code generation engine. Your sole purpose is to convert a user request into a single, complete, and functional block of code for the specified microcontroller.

    **User Request:** <<< {user_prompt} >>>

    **KEY INSTRUCTIONS:**
    1.  **Complete Requirement Fulfillment:** Your code MUST implement every feature, sensor, and logic step mentioned in the user request.
    2.  **Library and API Precision (CRITICAL):** You MUST use the exact, correct libraries and function calls for the specified hardware and components. For example, use `Adafruit_BME280.h` for a BME280 sensor or `WiFi.h` for an ESP32. Do not use placeholder or generic libraries.
    3.  **Self-Contained Code:** Generate a single, complete code file. It must include all necessary parts: library includes, variable/object declarations, a full `setup()` function, and a full `loop()` function containing the main logic.

    **OUTPUT MANDATE:**
    -   **Return ONLY raw source code.**
    -   Your response MUST NOT contain any explanations, comments, markdown, or any text other than the code itself.
    -   The first line of your output must be the first line of the code (e.g., an `#include` statement)."""
CODER_PROMPT_WITH_CONTEXT = """You are an expert IoT code generation engine. Your sole purpose is to convert a user request into a single, complete, and functional block of code for the specified microcontroller.

    **User Request:** <<< {user_prompt} >>>
    
    
    **KEY INSTRUCTIONS:**
    1.  **Complete Requirement Fulfillment:** Your code MUST implement every feature, sensor, and logic step mentioned in the user request.
    2.  **Library and API Precision (CRITICAL):** You MUST use the exact, correct libraries and function calls for the specified hardware and components. For example, use `Adafruit_BME280.h` for a BME280 sensor or `WiFi.h` for an ESP32. Do not use placeholder or generic libraries.
    3.  **Self-Contained Code:** Generate a single, complete code file. It must include all necessary parts: library includes, variable/object declarations, a full `setup()` function, and a full `loop()` function containing the main logic.

    **OUTPUT MANDATE:**
    -   **Return ONLY raw source code.**
    -   Your response MUST NOT contain any explanations, comments, markdown, or any text other than the code itself.
    -   The first line of your output must be the first line of the code (e.g., an `#include` statement).
    **REFERENCE CONTEXT (Optional):**
    If relevant, you may refer to the following example for inspiration. However, do NOT copy it or include any of its logic unless it directly applies to the user request:
    <<< {context} >>>"""


def generate_code(llm, prompt: str, context) -> str:
    if context != "":
        system_message = CODER_PROMPT_WITH_CONTEXT.format(
            user_prompt=prompt.strip(), context=context
        )
    else:
        system_message = CODER_PROMPT.format(user_prompt=prompt.strip())
    chat_input = f"<|im_start|>system\n{system_message.strip()}\n<|im_end|>\n<|im_start|>assistant\n"

    output = llm(
        chat_input,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )
    return output.get("choices", [{}])[0].get("text", "").strip()


def generate_code_stream(llm, prompt: str, context: str):
    if context != "":
        system_message = CODER_PROMPT_WITH_CONTEXT.format(
            user_prompt=prompt.strip(), context=context
        )
    else:
        system_message = CODER_PROMPT.format(user_prompt=prompt.strip())

    chat_input = f"<|im_start|>system\n{system_message.strip()}\n<|im_end|>\n<|im_start|>assistant\n"

    for chunk in llm(prompt=chat_input, max_tokens=1024, stream=True):
        yield chunk.get("choices", [{}])[0].get("text", "")
       

def generate(user_prompt: str, context: str) -> str:
    output = ""
    yield {"stage": "code_start"}
    for token in generate_code_stream(coder_llm_2048, user_prompt, context=context):
        yield {"stage": "code_progress", "token": token}
        output += token
    yield {"stage": "code_done", "code": output}
