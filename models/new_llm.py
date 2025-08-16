import os
import threading
import logging
from typing import Dict, Optional, Generator, Any
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import time
try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError("llama-cpp-python is required: pip install llama-cpp-python")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Enumeration of available model types."""
    CODER = "coder"
    COMPRESSOR = "compressor" 
    GENERATOR = "generator"
    BASELINE = "baseline"
    BASE = "base"

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    path: str
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    use_mmap: bool = True
    use_mlock: bool = True
    verbose: bool = False
    n_threads: Optional[int] = None

class OptimizedModelManager:
    """Singleton model manager with lazy loading and resource optimization."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._models: Dict[ModelType, Optional[Llama]] = {}
        self._configs: Dict[ModelType, ModelConfig] = {}
        self._last_used: Dict[ModelType, float] = {}
        self._load_lock = threading.Lock()
        self._setup_configs()
        self._initialized = True
        
        # Auto-detect optimal threading
        self._n_threads = self._detect_optimal_threads()
        logger.info(f"Using {self._n_threads} threads for model inference")
    
    def _detect_optimal_threads(self) -> int:
        """Detect optimal number of threads based on system."""
        cpu_count = os.cpu_count() or 4
        # Use 75% of available cores, but at least 2 and at most 8
        return max(2, min(8, int(cpu_count * 0.75)))
    
    def _setup_configs(self):
        """Setup model configurations with optimized defaults."""
        base_config = {
            'n_ctx': 2048,
            'verbose': False,
            'use_mmap': True,
            'use_mlock': True,
            'n_gpu_layers': 0,  # CPU-only by default
        }
        
        self._configs = {
            ModelType.CODER: ModelConfig(
                path="models/coder_F32/unsloth.Q4_K_M.gguf",
                **base_config
            ),
            ModelType.COMPRESSOR: ModelConfig(
                path="models/compressor_F32/unsloth.F32.gguf", 
                **base_config
            ),
            ModelType.GENERATOR: ModelConfig(
                path="models/generator_F32/unsloth.F32.gguf",
                **base_config
            ),
            ModelType.BASELINE: ModelConfig(
                path="models/baseline_F32/unsloth.F32.gguf",
                **base_config
            ),
            ModelType.BASE: ModelConfig(
                path="models/qwen/unsloth.BF16.gguf",
                **base_config
            ),
        }
    
    def _load_model(self, model_type: ModelType) -> Llama:
        """Load a model with optimized configuration."""
        config = self._configs[model_type]
        
        if not os.path.exists(config.path):
            raise FileNotFoundError(f"Model file not found: {config.path}")
        
        logger.info(f"Loading {model_type.value} model from {config.path}")
        start_time = time.time()
        
        model = Llama(
            model_path=config.path,
            n_ctx=config.n_ctx,
            n_threads=self._n_threads,
            verbose=config.verbose,
            use_mmap=config.use_mmap,
            use_mlock=config.use_mlock,
            n_gpu_layers=config.n_gpu_layers,
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ {model_type.value} model loaded in {load_time:.2f}s")
        
        return model
    
    def get_model(self, model_type: ModelType) -> Llama:
        """Get model with lazy loading and caching."""
        with self._load_lock:
            if model_type not in self._models or self._models[model_type] is None:
                self._models[model_type] = self._load_model(model_type)
            
            self._last_used[model_type] = time.time()
            return self._models[model_type]
    
    @contextmanager
    def use_model(self, model_type: ModelType):
        """Context manager for safe model usage."""
        model = self.get_model(model_type)
        try:
            yield model
        finally:
            # Could add cleanup logic here if needed
            pass
    
    def unload_model(self, model_type: ModelType):
        """Explicitly unload a model to free memory."""
        with self._load_lock:
            if model_type in self._models and self._models[model_type] is not None:
                del self._models[model_type]
                self._models[model_type] = None
                logger.info(f"🗑️ Unloaded {model_type.value} model")
    
    def cleanup_unused_models(self, max_idle_time: float = 1800):  # 30 minutes
        """Cleanup models that haven't been used recently."""
        current_time = time.time()
        
        for model_type, last_used in self._last_used.items():
            if current_time - last_used > max_idle_time:
                self.unload_model(model_type)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        loaded_models = [
            model_type.value for model_type, model in self._models.items() 
            if model is not None
        ]
        return {
            "loaded_models": loaded_models,
            "total_loaded": len(loaded_models),
            "last_used": {k.value: v for k, v in self._last_used.items()}
        }

# Global model manager instance
model_manager = OptimizedModelManager()

# Optimized prompt templates with better formatting
class PromptTemplates:
    """Centralized prompt template management."""
    
    CODER_BASE = """You are an expert IoT code generation engine. Your sole purpose is to convert a user request into a single, complete, and functional block of code for the specified microcontroller.

**User Request:** <<< {user_prompt} >>>

**KEY INSTRUCTIONS:**
1. **Complete Requirement Fulfillment:** Your code MUST implement every feature, sensor, and logic step mentioned in the user request.
2. **Library and API Precision (CRITICAL):** You MUST use the exact, correct libraries and function calls for the specified hardware and components. For example, use `Adafruit_BME280.h` for a BME280 sensor or `WiFi.h` for an ESP32. Do not use placeholder or generic libraries.
3. **Self-Contained Code:** Generate a single, complete code file. It must include all necessary parts: library includes, variable/object declarations, a full `setup()` function, and a full `loop()` function containing the main logic.

**OUTPUT MANDATE:**
- **Return ONLY raw source code.**
- Your response MUST NOT contain any explanations, comments, markdown, or any text other than the code itself.
- The first line of your output must be the first line of the code (e.g., an `#include` statement)."""

    CODER_WITH_CONTEXT = """You are an expert IoT code generation engine. Your sole purpose is to convert a user request into a single, complete, and functional block of code for the specified microcontroller.

**User Request:** <<< {user_prompt} >>>

**KEY INSTRUCTIONS:**
1. **Complete Requirement Fulfillment:** Your code MUST implement every feature, sensor, and logic step mentioned in the user request.
2. **Library and API Precision (CRITICAL):** You MUST use the exact, correct libraries and function calls for the specified hardware and components. For example, use `Adafruit_BME280.h` for a BME280 sensor or `WiFi.h` for an ESP32. Do not use placeholder or generic libraries.
3. **Self-Contained Code:** Generate a single, complete code file. It must include all necessary parts: library includes, variable/object declarations, a full `setup()` function, and a full `loop()` function containing the main logic.

**OUTPUT MANDATE:**
- **Return ONLY raw source code.**
- Your response MUST NOT contain any explanations, comments, markdown, or any text other than the code itself.
- The first line of your output must be the first line of the code (e.g., an `#include` statement).

**REFERENCE CONTEXT (Optional):**
If relevant, you may refer to the following example for inspiration. However, do NOT copy it or include any of its logic unless it directly applies to the user request:
<<< {context} >>>"""

# Optimized generation functions
def generate_code(prompt: str, context: str = "", model_type: ModelType = ModelType.CODER) -> str:
    """Generate code using specified model with optimizations."""
    template = PromptTemplates.CODER_WITH_CONTEXT if context else PromptTemplates.CODER_BASE
    system_message = template.format(user_prompt=prompt.strip(), context=context)
    chat_input = f"<|im_start|>system\n{system_message.strip()}\n<|im_end|>\n<|im_start|>assistant\n"
    full_output = ""
    token_count = 0
    start_time = time.time()
    
    with model_manager.use_model(model_type) as llm:
        for chunk in llm(
            chat_input,
            max_tokens=1024,
            stop=["<|im_end|>"],
            temperature=0.1,
            top_p=0.95,
            repeat_penalty=1.1,
            echo=False,
            stream=True,
        ):
            token = chunk.get("choices", [{}])[0].get("text", "")
            print(token, end="", flush=True)
            full_output += token
            token_count += 1

            # Show live tokens/sec rate, updating on same line
            # elapsed = time.time() - start_time
            # if elapsed > 0:
            #     rate = token_count / elapsed
            #     sys.stdout.write(f"\rTokens: {token_count} | Rate: {rate:.2f} tokens/sec")
            #     sys.stdout.flush()

    end_time = time.time()
    total_elapsed = end_time - start_time
    tokens_per_second = token_count / total_elapsed if total_elapsed > 0 else 0

    print()  # Move to next line after streaming is done

    return {
        "output": full_output.strip(),
        "token_count": token_count,
        "elapsed_time": total_elapsed,
        "tokens_per_second": tokens_per_second,
    }

def generate_code_stream(prompt: str, context: str = "", 
                        model_type: ModelType = ModelType.CODER) -> Generator[str, None, None]:
    """Generate code with streaming for real-time feedback."""
    template = PromptTemplates.CODER_WITH_CONTEXT if context else PromptTemplates.CODER_BASE
    system_message = template.format(user_prompt=prompt.strip(), context=context)
    chat_input = f"<|im_start|>system\n{system_message.strip()}\n<|im_end|>\n<|im_start|>assistant\n"
    
    with model_manager.use_model(model_type) as llm:
        for chunk in llm(
            prompt=chat_input,
            max_tokens=1024,
            stream=True,
            temperature=0.1,
            top_p=0.95,
            repeat_penalty=1.1,
            stop=["<|im_end|>"],
        ):
            yield chunk.get("choices", [{}])[0].get("text", "")

def generate(user_prompt: str, context: str = "", 
            model_type: ModelType = ModelType.CODER) -> Generator[Dict[str, Any], None, None]:
    """High-level generation function with progress tracking."""
    output = ""
    
    yield {"stage": "code_start", "timestamp": time.time()}
    
    try:
        for token in generate_code_stream(user_prompt, context, model_type):
            yield {"stage": "code_progress", "token": token}
            output += token
        
        yield {
            "stage": "code_done", 
            "code": output, 
            "timestamp": time.time(),
            "model_used": model_type.value
        }
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        yield {
            "stage": "error", 
            "error": str(e), 
            "timestamp": time.time()
        }

# Convenience functions for backward compatibility
def get_coder_llm():
    """Get coder model (backward compatibility)."""
    return model_manager.get_model(ModelType.CODER)

def get_compressor_llm():
    """Get compressor model (backward compatibility)."""
    return model_manager.get_model(ModelType.COMPRESSOR)

def get_generator_llm():
    """Get generator model (backward compatibility)."""
    return model_manager.get_model(ModelType.GENERATOR)

def get_baseline_llm():
    """Get baseline model (backward compatibility)."""
    return model_manager.get_model(ModelType.BASELINE)

def get_base_llm():
    """Get base model (backward compatibility)."""
    return model_manager.get_model(ModelType.BASE)

# Batch processing optimization
class BatchProcessor:
    """Optimized batch processing for multiple requests."""
    
    def __init__(self, model_type: ModelType = ModelType.CODER, batch_size: int = 4):
        self.model_type = model_type
        self.batch_size = batch_size
    
    def process_batch(self, prompts: list, contexts: list = None) -> list:
        """Process multiple prompts efficiently."""
        if contexts is None:
            contexts = [""] * len(prompts)
        
        if len(prompts) != len(contexts):
            raise ValueError("Prompts and contexts must have same length")
        
        results = []
        
        with model_manager.use_model(self.model_type):
            for i in range(0, len(prompts), self.batch_size):
                batch_prompts = prompts[i:i + self.batch_size]
                batch_contexts = contexts[i:i + self.batch_size]
                
                batch_results = []
                for prompt, context in zip(batch_prompts, batch_contexts):
                    try:
                        result = generate_code(prompt, context, self.model_type)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
                        batch_results.append("")
                
                results.extend(batch_results)
                logger.info(f"Processed batch {i//self.batch_size + 1}")
        
        return results

# System utilities
def optimize_system():
    """Apply system-level optimizations."""
    # Set optimal environment variables
    os.environ.setdefault('OMP_NUM_THREADS', str(model_manager._n_threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(model_manager._n_threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(model_manager._n_threads))
    
    logger.info("✅ System optimizations applied")

def get_system_info() -> Dict[str, Any]:
    """Get system and model information."""
    return {
        "cpu_count": os.cpu_count(),
        "threads_configured": model_manager._n_threads,
        "models_loaded": model_manager.get_memory_usage(),
        "environment_optimized": True
    }

def cleanup_all_models():
    """Cleanup all loaded models to free memory."""
    for model_type in ModelType:
        model_manager.unload_model(model_type)
    logger.info("🧹 All models unloaded")

# Backward compatibility aliases
coder_llm_2048 = lambda: model_manager.get_model(ModelType.CODER)
compressor_llm_2048 = lambda: model_manager.get_model(ModelType.COMPRESSOR)
generator_llm_2048 = lambda: model_manager.get_model(ModelType.GENERATOR)
baseline_llm = lambda: model_manager.get_model(ModelType.BASELINE)
base_llm = lambda: model_manager.get_model(ModelType.BASE)

# Initialize optimizations on import
optimize_system()

# Example usage and testing
if __name__ == "__main__":
    # Test the optimized system
    print("Testing optimized LLM system...")
    
    # Check system info
    info = get_system_info()
    print(f"System info: {info}")
    
    # Test single generation
    test_prompt = "Set up a gesture-controlled device using the APDS-9960 sensor to detect hand movements and control an LED strip"
    result = generate_code(test_prompt)
    print(result)
    print(f"Generated {len(result)} characters of code")
    
    # Test memory management
    print(f"Memory usage: {model_manager.get_memory_usage()}")
    
    # Test cleanup
    cleanup_all_models()
    print("Cleanup complete")