# src/2_benchmark_inference.py

import time
import torch
import numpy as np
import pandas as pd
import psutil
from transformers import AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix
)
from optimum.rbln import (
    RBLNAutoModelForCausalLM, RBLNAutoModelForSeq2SeqLM,
    RBLNQwen2ForCausalLM, RBLNLlamaForCausalLM,
    RBLNGemmaForCausalLM, RBLNOPTForCausalLM, RBLNExaoneForCausalLM
)

# ======================================================================
# == BENCHMARK CONFIGURATION                                          ==
# ======================================================================
# IMPORTANT: Modify these settings for each benchmark run.

# 1. Point to the directory of the compiled model from Step 1.
MODEL_DIR = "/path/to/your/compiled_models/Qwen2.5-0.5B-Instruct" 
TOKENIZER_DIR = MODEL_DIR  # Usually the same as the model directory

# 2. Benchmark parameters
NUM_SAMPLES = 100  # Number of samples per task (as in the paper)
MAX_NEW_TOKENS = 50 # Max tokens to generate (as in the paper)
RANDOM_SEED = 42
# ======================================================================

def create_prompt(task: str, data_item: dict) -> str:
    """Creates a standardized, zero-shot prompt for a given task."""
    if task == "NLU":
        sentence = data_item['sentence']
        return (
            f"[System]: You are an expert sentiment analysis model. "
            f"Analyze the sentence and respond with only 'Positive' or 'Negative'. "
            f"Do not include explanations or extra words.\n"
            f"[User]: \"{sentence}\"\n[Assistant]:"
        )
    elif task == "QA":
        context = data_item['context']
        question = data_item['question']
        return (
            f"[System]: You are a world-class question answering model. "
            f"Given the context, provide a concise and accurate answer.\n"
            f"[User]: Context:\n\"{context}\"\nQuestion:\n\"{question}\"\n[Assistant]:"
        )
    elif task == "Summarization":
        article = data_item['article']
        return (
            f"[System]: You are an expert summarization model. "
            f"Summarize the following article concisely and accurately in one paragraph.\n"
            f"[User]: {article}\n[Assistant]:"
        )
    return ""

def evaluate_generative_text(preds: list, labels: list) -> dict:
    """Evaluates generated text using ROUGE-1 F1-Score."""
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    rouge1_f1_scores = []
    for pred, label in zip(preds, labels):
        if not pred or not label:
            rouge1_f1_scores.append(0.0)
            continue
        score = scorer.score(label, pred)
        rouge1_f1_scores.append(score['rouge1'].fmeasure)
    return {"rouge1_f1": np.mean(rouge1_f1_scores)}

def evaluate_classification_text(preds: list, labels: list) -> dict:
    """Evaluates classification text by parsing 'Positive'/'Negative'."""
    # Convert text predictions to integer labels (1 for Positive, 0 for Negative)
    int_preds = [1 if "Positive" in p else 0 if "Negative" in p else -1 for p in preds]
    
    # Filter out invalid predictions (-1)
    valid_indices = [i for i, p in enumerate(int_preds) if p != -1]
    if not valid_indices:
        return {"f1_score": 0, "precision": 0, "recall": 0, "valid_samples": 0}
        
    filtered_preds = [int_preds[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]

    return {
        "f1_score": f1_score(filtered_labels, filtered_preds, zero_division=0),
        "precision": precision_score(filtered_labels, filtered_preds, zero_division=0),
        "recall": recall_score(filtered_labels, filtered_preds, zero_division=0),
        "valid_samples": len(filtered_preds)
    }

def run_benchmark():
    """Main function to execute the benchmarking process."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting benchmark for model: {os.path.basename(MODEL_DIR)}")
    print(f"🔩 Using device: {device}")

    # --- 1. Load Tokenizer and Model ---
    print("\n[1/4] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # IMPORTANT: Uncomment the correct model class for your model.
    # This must match the architecture of the model being tested.
    model = RBLNAutoModelForCausalLM.from_pretrained(
        MODEL_DIR, export=False, rbln_create_runtimes=True
    )
    # model = RBLNQwen2ForCausalLM.from_pretrained(MODEL_DIR, export=False, rbln_create_runtimes=True)
    # model = RBLNLlamaForCausalLM.from_pretrained(MODEL_DIR, export=False, rbln_create_runtimes=True)
    # model = RBLNGemmaForCausalLM.from_pretrained(MODEL_DIR, export=False, rbln_create_runtimes=True)
    # model = RBLNOPTForCausalLM.from_pretrained(MODEL_DIR, export=False, rbln_create_runtimes=True)
    # model = RBLNExaoneForCausalLM.from_pretrained(MODEL_DIR, export=False, rbln_create_runtimes=True)
    # model = RBLNAutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, export=False, rbln_create_runtimes=True)
    
    model.to(device)
    print("✅ Model loaded successfully.")

    # --- 2. Load Datasets ---
    print("\n[2/4] Loading and preparing datasets...")
    datasets_to_run = {
        "NLU": load_dataset("glue", "sst2", split="validation"),
        "QA": load_dataset("squad_v2", split="validation"),
        "Summarization": load_dataset("cnn_dailymail", "3.0.0", split="validation")
    }
    for key in datasets_to_run:
        datasets_to_run[key] = datasets_to_run[key].shuffle(seed=RANDOM_SEED).select(range(NUM_SAMPLES))
    print("✅ Datasets prepared.")

    # --- 3. Run Inference and Collect Metrics ---
    all_results = []
    for task_name, dataset in datasets_to_run.items():
        print(f"\n[3/4] Running task: {task_name.upper()}...")
        
        # Prepare prompts and labels
        prompts = [create_prompt(task_name, item) for item in dataset]
        if task_name == "NLU":
            labels = [item["label"] for item in dataset]
        elif task_name == "QA":
            labels = [item["answers"]["text"][0] if item["answers"]["text"] else "" for item in dataset]
        else: # Summarization
            labels = [item["highlights"] for item in dataset]

        task_trial_results = []
        for i, prompt in enumerate(prompts):
            # Reset system state for independent trials
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            process = psutil.Process()
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            
            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False # Use deterministic generation for reproducibility
                )
            end_time = time.time()

            # --- Collect Metrics for this trial ---
            latency_ms = (end_time - start_time) * 1000
            num_input_tokens = inputs["input_ids"].shape[1]
            num_gen_tokens = generated_ids.shape[1] - num_input_tokens
            tokens_per_sec = num_gen_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
            
            # Decode generated text
            # For causal LMs, we skip the prompt part
            if isinstance(model, RBLNAutoModelForSeq2SeqLM):
                 gen_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            else:
                 gen_text = tokenizer.decode(generated_ids[0][num_input_tokens:], skip_special_tokens=True)

            trial_data = {
                "model": os.path.basename(MODEL_DIR),
                "task": task_name,
                "trial_id": i,
                "latency_ms": latency_ms,
                "tokens_generated": num_gen_tokens,
                "tokens_per_sec": tokens_per_sec,
                "prediction": gen_text,
                "label": labels[i]
            }
            task_trial_results.append(trial_data)
            all_results.append(trial_data)

            print(f"  - Sample {i+1}/{NUM_SAMPLES}: Latency={latency_ms:.2f} ms, Tokens/sec={tokens_per_sec:.2f}")

        # --- 4. Calculate and Print Task Summary ---
        print(f"\n[4/4] Analyzing results for {task_name.upper()}...")
        df_task = pd.DataFrame(task_trial_results)
        
        # Performance Metrics
        if task_name == "NLU":
            perf_metrics = evaluate_classification_text(df_task['prediction'].tolist(), df_task['label'].tolist())
            print(f"  - F1-Score: {perf_metrics['f1_score']:.4f}")
        else:
            perf_metrics = evaluate_generative_text(df_task['prediction'].tolist(), df_task['label'].tolist())
            print(f"  - ROUGE-1 F1: {perf_metrics['rouge1_f1']:.4f}")

        # Latency Metrics
        latencies = df_task['latency_ms']
        print(f"  - Latency (ms):")
        print(f"    - Mean: {latencies.mean():.2f}, Std: {latencies.std():.2f}")
        print(f"    - Median (P50): {latencies.median():.2f}")
        print(f"    - P95: {latencies.quantile(0.95):.2f}, P99: {latencies.quantile(0.99):.2f}")

        # Throughput Metrics
        total_tokens = df_task['tokens_generated'].sum()
        total_time_sec = df_task['latency_ms'].sum() / 1000
        overall_throughput = total_tokens / total_time_sec if total_time_sec > 0 else 0
        print(f"  - Throughput (tokens/sec): {overall_throughput:.2f}")

    # --- Final Output ---
    print("\n\n{'='*20} Benchmark Complete {'='*20}")
    print("The following DataFrame contains the raw data for all trials.")
    print("Copy this data into your `Data_시계열.xlsx` file for the final analysis.")
    
    df_final = pd.DataFrame(all_results)
    print("\n--- Raw Data Output ---")
    # Print in a copy-paste friendly format (e.g., CSV)
    print(df_final.to_csv(index=False))
    
    # Save to a temporary file for convenience
    output_filename = f"benchmark_output_{os.path.basename(MODEL_DIR)}.csv"
    df_final.to_csv(output_filename, index=False)
    print(f"\n✅ Raw data also saved to: {output_filename}")


if __name__ == "__main__":
    run_benchmark()