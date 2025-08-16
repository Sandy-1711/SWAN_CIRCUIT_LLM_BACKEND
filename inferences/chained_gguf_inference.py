import psutil
import os
import time
from models.shared_llms import coder_llm, compressor_llm, generator_llm

model = None
tok = None


def get_resource_usage():
    process = psutil.Process(os.getpid())
    cpu = psutil.cpu_percent(interval=0.1)
    mem = process.memory_info().rss / (1024 * 1024)  # MB
    return cpu, mem


def get_temperature():
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            core_temps = temps["coretemp"]
            return sum(t.current for t in core_temps) / len(core_temps)
    except Exception:
        pass
    return None


CODER_PROMPT = """You are an expert IoT code generation engine. Your sole purpose is to convert a user request into a single, complete, and functional block of code for the specified microcontroller.

    **User Request:** <<< {user_prompt} >>>

    **KEY INSTRUCTIONS:**
    1.  **Complete Requirement Fulfillment:** Your code MUST implement every feature, sensor, and logic step mentioned in the user request.
    2.  **Library and API Precision (CRITICAL):** You MUST use the exact, correct libraries and function calls for the specified hardware and components. For example, use `Adafruit_BME280.h` for a BME280 sensor or `WiFi.h` for an ESP32. Do not use placeholder or generic libraries.
    3.  **Self-Contained Code:** Generate a single, complete code file. It must include all necessary parts: library includes, variable/object declarations, a full `setup()` function, and a full `loop()` function containing the main logic.

    **OUTPUT MANDATE:**
    -   **Return ONLY raw source code.**
    -   Your response MUST NOT contain any explanations, comments, markdown, or any text other than the code itself.
    -   The first line of your output must be the first line of the code (e.g., an `#include` statement). """

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

# Example JSON for the generator prompt's few-shot example
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


# === Functions ===


def generate_code(llm, prompt: str) -> str:
    system_message = CODER_PROMPT.format(user_prompt=prompt.strip())
    chat_input = f"<|im_start|>system\n{system_message.strip()}\n<|im_end|>\n<|im_start|>assistant\n"

    output = llm(
        chat_input,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )
    return output.get("choices", [{}])[0].get("text", "").strip()


def compress_to_ir(llm, prompt: str, code: str) -> str:
    message = COMPRESSOR_PROMPT.format(user_prompt=prompt, code=code).strip()
    user_block = f"{prompt}\n\n{code}"
    chat_input = (
        f"<|im_start|>system\n{message}\n<|im_end|>\n"
        f"<|im_start|>user\n{user_block}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    response = llm(
        chat_input,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )
    return response.get("choices", [{}])[0].get("text", "").strip()

def generate_json(llm, specification: str) -> str:
    message = GENERATOR_PROMPT.format(specification=specification).strip()
    chat_input = f"<|im_start|>system\n{message}\n<|im_end|>\n<|im_start|>assistant\n"
    print(chat_input)
    response = llm(
        chat_input,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )
    return response.get("choices", [{}])[0].get("text", "").strip()


# === Main Pipeline ===


def run_chained_gguf_pipeline(user_prompt: str):
    metrics = {}

    # --- Step 1: Code Generation ---
    print("\n[1] Generating Code...")
    cpu1, ram1 = get_resource_usage()
    temp1 = get_temperature()
    t1 = time.time()
    code = generate_code(coder_llm, user_prompt)
    t2 = time.time()
    cpu2, ram2 = get_resource_usage()
    temp2 = get_temperature()
    print(code)
    metrics["code_gen"] = {
        "time": t2 - t1,
        "cpu": (cpu1, cpu2),
        "ram": (ram1, ram2),
        "temp": (temp1, temp2),
    }

    # --- Step 2: IR Compression ---
    print("\n[2] Compressing to Intermediate Representation...")
    cpu3, ram3 = get_resource_usage()
    temp3 = get_temperature()
    t3 = time.time()
    ir = compress_to_ir(compressor_llm, user_prompt, code)
    t4 = time.time()
    cpu4, ram4 = get_resource_usage()
    temp4 = get_temperature()
    print(ir)
    metrics["compressor"] = {
        "time": t4 - t3,
        "cpu": (cpu3, cpu4),
        "ram": (ram3, ram4),
        "temp": (temp3, temp4),
    }

    # --- Step 3: JSON Circuit Generation ---
    print("\n[3] Generating JSON Circuit...")
    cpu5, ram5 = get_resource_usage()
    temp5 = get_temperature()
    t5 = time.time()
    json_output = generate_json(generator_llm, ir)
    t6 = time.time()
    cpu6, ram6 = get_resource_usage()
    temp6 = get_temperature()
    print(json_output)
    metrics["generator"] = {
        "time": t6 - t5,
        "cpu": (cpu5, cpu6),
        "ram": (ram5, ram6),
        "temp": (temp5, temp6),
    }

    # --- Overall Metrics ---
    total_time = (
        metrics["code_gen"]["time"]
        + metrics["compressor"]["time"]
        + metrics["generator"]["time"]
    )
    total_cpu_start = metrics["code_gen"]["cpu"][0]
    total_cpu_end = metrics["generator"]["cpu"][1]
    total_ram_start = metrics["code_gen"]["ram"][0]
    total_ram_end = metrics["generator"]["ram"][1]
    total_temp_start = metrics["code_gen"]["temp"][0]
    total_temp_end = metrics["generator"]["temp"][1]

    # --- Summary Output ---
    print("\n" + "=" * 40)
    print("🔍 Inference Performance Summary")
    print("=" * 40)

    def print_block(title, data):
        print(f"\n[{title}]")
        print(f"⏱ Time     : {data['time']:.2f} sec")
        print(f"🧠 CPU     : {data['cpu'][0]:.2f}% → {data['cpu'][1]:.2f}%")
        print(f"💾 RAM     : {data['ram'][0]:.2f}MB → {data['ram'][1]:.2f}MB")
        if data["temp"][0] and data["temp"][1]:
            print(f"🌡 Temp    : {data['temp'][0]:.2f}°C → {data['temp'][1]:.2f}°C")

    print_block("1️⃣ Code Generation", metrics["code_gen"])
    print_block("2️⃣ IR Compression", metrics["compressor"])
    print_block("3️⃣ JSON Generation", metrics["generator"])

    print("\n🧮 [Total Pipeline]")
    print(f"⏱ Total Time : {total_time:.2f} sec")
    print(f"🧠 CPU Total  : {total_cpu_start:.2f}% → {total_cpu_end:.2f}%")
    print(f"💾 RAM Total  : {total_ram_start:.2f}MB → {total_ram_end:.2f}MB")
    if total_temp_start and total_temp_end:
        print(f"🌡 Temp Total : {total_temp_start:.2f}°C → {total_temp_end:.2f}°C")
    print("=" * 40)

    return code, ir, json_output


# === Example Run ===
# if __name__ == "__main__":
#     user_prompt = (
#         "Build a soft real-time robot arm controller using multiple PID loops."
#     )
#     code, ir, json_output = circuit_pipeline(user_prompt)


# def load_base_model_and_tokenizer():
#     """Initializes the global model and tokenizer variables if they haven't been already."""
#     global model, tok
#     if model is None:
#         print("Loading base model and tokenizer for the first time...")
#         model, tok = FastLanguageModel.from_pretrained(
#             model_name="Qwen/Qwen3-4B",  # This MUST match the model used for training
#             max_seq_length=2048,
#             load_in_4bit=True,
#             token="<><>",
#         )
#         print("✅ Base model and tokenizer loaded.")

# load_base_model_and_tokenizer()

# model.load_adapter("code_lora", adapter_name="coder")
# model.set_adapter("coder")
# chat_input = f"<|im_start|>system\n{user_prompt.strip()}\n<|im_end|>\n<|im_start|>assistant\n"

# inputs = tok([chat_input], return_tensors="pt").to("cuda")
# outputs = model.generate(
#     **inputs, max_new_tokens=1024, use_cache=True, pad_token_id=tok.eos_token_id
# )
# raw_output = tok.batch_decode(outputs)[0]

# model.delete_adapter("coder")
# print(raw_output)
