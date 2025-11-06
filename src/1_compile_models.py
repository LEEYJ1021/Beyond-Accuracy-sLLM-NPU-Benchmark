# src/1_compile_models.py

import os
import time
from typing import Optional, Type

from optimum.rbln import (
    RBLNAutoModelForCausalLM,
    RBLNAutoModelForSeq2SeqLM,
)

try:
    # Specific classes can offer better performance or compatibility
    from optimum.rbln import RBLNLlamaForCausalLM
except ImportError:
    print("Warning: RBLNLlamaForCausalLM not found. Falling back to RBLNAutoModelForCausalLM.")
    RBLNLlamaForCausalLM = RBLNAutoModelForCausalLM


class sLLMCompiler:
    """
    sLLM Compiler for Rebellions ATOM™ NPU.

    This class automates the process of compiling (exporting) various small-to-medium
    language models from Hugging Face Hub into an NPU-optimized format.

    Features:
    - Centralized configuration for multiple models.
    - Automatic application of NPU-specific settings (FlashAttention, Tensor Parallel, KV Cache).
    - Robust OOM (Out-of-Memory) handling with automatic resource reduction and retries.
    """

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initializes the compiler.

        Args:
            hf_token (Optional[str]): Hugging Face API token, if required for private models.
        """
        self.hf_token = hf_token
        self.batch_size = 1  # Standard for single-user latency benchmarks
        self.seq_len_reduction_factor = 0.5  # Factor to reduce seq_len on OOM
        self.kv_cache_reduction_factor = 0.5 # Factor to reduce kv_cache on OOM

        # ======================================================================
        # == MODEL CONFIGURATION                                              ==
        # ======================================================================
        # Define all models to be compiled here.
        # Each entry specifies the model ID, the appropriate optimum-rebellions class,
        # and NPU-specific compilation parameters.
        self.models_config = {
            # --- DeepSeek Family ---
            "deepseek-r1-distill-qwen-1.5b": {
                "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "model_class": RBLNAutoModelForCausalLM,
                "rbln_max_seq_len": 32768,
                "rbln_tensor_parallel_size": 1,
                "rbln_attn_impl": "flash_attn",
                "rbln_kvcache_partition_len": 4096,
            },
            # --- Llama Family ---
            "llama-3.2-3b": {
                "model_id": "meta-llama/Llama-3.2-3B-Instruct",
                "model_class": RBLNLlamaForCausalLM,
                "rbln_max_seq_len": 32768,  # Reduced from 128k for stability
                "rbln_tensor_parallel_size": 1,
                "rbln_attn_impl": "flash_attn",
                "rbln_kvcache_partition_len": 4096, # Reduced from 16k
                "note": "Reduced seq_len & kv_cache for stable compilation on memory-constrained devices.",
            },
            # --- Gemma Family ---
            "gemma-2b-it": {
                "model_id": "google/gemma-2b-it",
                "model_class": RBLNAutoModelForCausalLM,
                "rbln_max_seq_len": 8192,
                "rbln_tensor_parallel_size": 1,
            },
            # --- OPT Family ---
            "opt-2.7b": {
                "model_id": "facebook/opt-2.7b",
                "model_class": RBLNAutoModelForCausalLM,
                "rbln_max_seq_len": 2048,
                "rbln_tensor_parallel_size": 1,
            },
            # --- Qwen2.5 Family ---
            "qwen2.5-0.5b-instruct": {
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "model_class": RBLNAutoModelForCausalLM,
                "rbln_max_seq_len": 32768,
                "rbln_tensor_parallel_size": 1,
            },
            "qwen2.5-1.5b-instruct": {
                "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "model_class": RBLNAutoModelForCausalLM,
                "rbln_max_seq_len": 32768,
                "rbln_tensor_parallel_size": 1,
            },
            "qwen2.5-3b-instruct": {
                "model_id": "Qwen/Qwen2.5-3B-Instruct",
                "model_class": RBLNAutoModelForCausalLM,
                "rbln_max_seq_len": 32768,
                "rbln_tensor_parallel_size": 1,
            },
            # --- Korean Models ---
            "exaone-3.5-2.4b-instruct": {
                "model_id": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
                "model_class": RBLNAutoModelForCausalLM,
                "rbln_max_seq_len": 32768,
                "rbln_tensor_parallel_size": 1,
            },
            "midm-2.0-mini-instruct": {
                "model_id": "K-intelligence/Midm-2.0-Mini-Instruct",
                "model_class": RBLNAutoModelForCausalLM,
                "rbln_max_seq_len": 32768,
                "rbln_tensor_parallel_size": 1,
            },
        }

    def compile_single_model(self, model_key: str, max_retries: int = 3):
        """
        Compiles a single model specified by its key, with OOM handling.

        Args:
            model_key (str): The key corresponding to the model in `self.models_config`.
            max_retries (int): The maximum number of compilation attempts in case of OOM errors.
        """
        if model_key not in self.models_config:
            raise ValueError(f"❌ Unknown model key: {model_key}")

        cfg = self.models_config[model_key]
        model_class: Type = cfg["model_class"]
        model_id = cfg["model_id"]

        print(f"\n{'='*20} Compiling: {model_key.upper()} {'='*20}")
        print(f"🔄 Model ID: {model_id}")

        # Initial compilation parameters from config
        seq_len = cfg.get("rbln_max_seq_len", 8192)
        kv_len = cfg.get("rbln_kvcache_partition_len", 1024)

        for attempt in range(max_retries):
            print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
            print(f"  - Max Seq Length: {int(seq_len)}")
            print(f"  - KV Cache Partition: {int(kv_len)}")
            try:
                # Prepare kwargs for from_pretrained
                kwargs = {
                    "model_id": model_id,
                    "export": True,
                    "rbln_batch_size": self.batch_size,
                    "rbln_max_seq_len": int(seq_len),
                    "rbln_kvcache_partition_len": int(kv_len),
                    "rbln_tensor_parallel_size": cfg.get("rbln_tensor_parallel_size", 1),
                }
                if "rbln_attn_impl" in cfg:
                    kwargs["rbln_attn_impl"] = cfg["rbln_attn_impl"]
                if self.hf_token:
                    kwargs["token"] = self.hf_token

                # Load, compile, and export the model
                model = model_class.from_pretrained(**kwargs)
                
                # Define save directory based on model_id
                save_dir = os.path.join("./compiled_models", os.path.basename(model_id))
                model.save_pretrained(save_dir)

                print(f"✅ Successfully compiled and saved to: {save_dir}")
                return  # Exit the function on success

            except Exception as e:
                err_msg = str(e).lower()
                # Check for common OOM error messages
                if "device_oom" in err_msg or "out of memory" in err_msg:
                    print(f"⚠️ OOM detected on attempt {attempt + 1}. Reducing resources and retrying...")
                    
                    # Reduce sequence length and KV cache partition size
                    seq_len *= self.seq_len_reduction_factor
                    kv_len *= self.kv_cache_reduction_factor
                    
                    # Ensure they don't fall below a minimum threshold
                    seq_len = max(2048, seq_len)
                    kv_len = max(512, kv_len)
                    
                    time.sleep(5)  # Wait a bit for memory to be released
                    continue
                else:
                    # For non-OOM errors, fail immediately
                    print(f"❌ An unexpected error occurred during compilation: {e}")
                    raise e

        # If all retries fail
        raise RuntimeError(
            f"❌ Failed to compile {model_key} after {max_retries} attempts due to persistent memory limits."
        )

    def run_all(self):
        """
        Iterates through all models in the configuration and compiles them.
        """
        print("🤖 sLLM Compilation Tool for Rebellions NPU (Stable Mode)")
        print(f"🔧 Starting compilation for {len(self.models_config)} models...\n")
        
        if not os.path.exists("./compiled_models"):
            os.makedirs("./compiled_models")

        succeeded = []
        failed = []

        for model_key in self.models_config:
            try:
                self.compile_single_model(model_key)
                succeeded.append(model_key)
            except Exception as e:
                print(f"\n{'!'*20} FAILED: {model_key} {'!'*20}")
                print(f"Error: {e}")
                failed.append(model_key)

        print("\n\n{'='*20} Compilation Summary {'='*20}")
        print(f"✅ Succeeded: {len(succeeded)} models -> {succeeded}")
        print(f"❌ Failed: {len(failed)} models -> {failed}")


if __name__ == "__main__":
    # To use a Hugging Face token for private models, uncomment the line below
    # hf_auth_token = "YOUR_HF_TOKEN"
    # compiler = sLLMCompiler(hf_token=hf_auth_token)
    
    compiler = sLLMCompiler()
    compiler.run_all()