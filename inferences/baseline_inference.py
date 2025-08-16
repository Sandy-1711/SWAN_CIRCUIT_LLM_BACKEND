import re
from models.shared_llms import baseline_llm


SYSTEM_PROMPT = (
    "You are an **expert IoT Systems Engineer and Embedded Architect** specializing in translating high-level human instructions into complete low-level hardware+software implementations.\n\n"
    "**Input:**\n"
    "  • **User Prompt:** {user_prompt}\n\n"
    "**Your Task:**\n"
    "1. **C++ Arduino Code Generation**\n"
    "   - Generate fully functional and industry-standard C++ Arduino code based on the user prompt.\n"
    "   - Ensure the code fulfills all requested functionalities and uses appropriate hardware libraries.\n"
    "   - Include necessary macros (`#define`) for pin assignments and constants.\n"
    "   - Do **not** include any comments, explanations, or unnecessary whitespace.\n"
    "   - Focus on efficiency, correctness, and clean structure.\n\n"
    "2. **Wokwi-Compatible JSON Circuit Description**\n"
    "   - Create a valid JSON object compatible with Wokwi that fully describes the circuit used in the code.\n"
    "   - The JSON must contain **exactly two top-level keys**:\n"
    "     • `parts`: an array of all components used.\n"
    "     • `connections`: an array of all connections between components, matching the code.\n"
    "   - Ensure all part types, pins, and connections are consistent with actual Wokwi-supported components and match the Arduino code precisely.\n\n"
    "**STRICT OUTPUT FORMAT REQUIREMENTS**\n"
    "- Return **only** the raw Arduino C++ code and raw JSON object.\n"
    "- **Do not** include any explanations, titles, headers, markdown outside of code fences, or extra output.\n"
    "- Wrap the C++ code inside a ` ```cpp ` code fence.\n"
    "- Wrap the JSON object inside a ` ```json ` code fence.\n"
    "- Output format must be:\n"
    "  ```cpp\n"
    "  // Arduino code\n"
    "  ```\n"
    "  ```json\n"
    "  // Wokwi JSON\n"
    "  ```\n"
)


def strip_assistant_output(raw_output: str):
    code_match = re.search(r"```cpp\s*(.*?)\s*```", raw_output, re.DOTALL)
    json_match = re.search(r"```json\s*(.*?)\s*```", raw_output, re.DOTALL)

    if code_match and json_match:
        generated_code = code_match.group(1).strip()
        generated_json = json_match.group(1).strip()
    else:
        json_start = raw_output.find('{"parts"')

        if json_start != -1:
            generated_code = raw_output[:json_start].strip()
            generated_json = raw_output[json_start:].strip()
        else:
            generated_code = raw_output.strip()
            generated_json = ""

    return generated_code, generated_json


def generate_code_and_json(llm, prompt: str) -> str:
    message = SYSTEM_PROMPT.format(user_prompt=prompt.strip())
    chat_input = (
        f"<|im_start|>system\n{message.strip()}\n<|im_end|>\n<|im_start|>assistant\n"
    )

    output = llm(
        chat_input,
        max_tokens=1024,
    )

    return output.get("choices", [{}])[0].get("text", "").strip()


def run_baseline_pipeline(user_prompt: str) -> str:
    print("Generating code...")
    result = generate_code_and_json(baseline_llm, user_prompt)
    generated_code, generated_json = strip_assistant_output(result)

    return generated_code, generated_json
