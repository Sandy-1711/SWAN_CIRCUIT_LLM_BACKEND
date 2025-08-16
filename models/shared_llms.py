from llama_cpp import Llama
import os

# coder_llm = Llama(
#     model_path="models/coder_8192_F32/unsloth.F32.gguf",
#     n_ctx=8192,
#     verbose=False,
#     use_mmap=True,
#     use_mlock=True,
#     # n_threads=16,
#     n_gpu_layers=0,
# )

# compressor_llm = Llama(
#     model_path="models/compressor_8192_F32/unsloth.F32.gguf",
#     n_ctx=8192,
#     use_mmap=True,
#     use_mlock=True,
#     n_gpu_layers=0,
#     verbose=False,
# )
# generator_llm = Llama(
#     model_path="models/generator_8192_F32/unsloth.F32.gguf",
#     n_ctx=8192,
#     verbose=False,
#     use_mmap=True,
#     use_mlock=True,  # Set True only if you want to lock it in RAM
#     n_gpu_layers=0,  # if CPU-only
# )

coder_llm_2048 = Llama(
    model_path="models/coder_F32/unsloth.F32.gguf",
    n_ctx=2048,
    verbose=False,
    use_mmap=True,
    use_mlock=True,
    # n_threads=16,
    n_gpu_layers=0,
)

compressor_llm_2048 = Llama(
    model_path="models/compressor_F32/unsloth.F32.gguf",
    n_ctx=2048,
    use_mmap=True,
    use_mlock=True,
    n_gpu_layers=0,
    verbose=False,
)
generator_llm_2048 = Llama(
    model_path="models/generator_F32/unsloth.F32.gguf",
    n_ctx=2048,
    verbose=False,
    use_mmap=True,
    use_mlock=True,  # Set True only if you want to lock it in RAM
    n_gpu_layers=0,  # if CPU-only
)

baseline_llm = Llama(
    model_path="models/baseline_F32/unsloth.F32.gguf",
    n_ctx=2048,
    verbose=False,
    use_mmap=True,
    use_mlock=True,  # Set True only if you want to lock it in RAM
    n_gpu_layers=0,  # if CPU-only
)

base_llm = Llama(
    model_path="models/qwen/unsloth.BF16.gguf",
    n_ctx=2048,
    # n_threads=N_THREADS,
    use_mlock=True,
    use_mmap=True,
    n_gpu_layers=0,  # if CPU-only
    verbose=False,
)
