import pandas as pd, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def create_prompt(task, it):
    if task == "NLU":
        return (f"[System]: You are an expert sentiment analysis model. "
                f"Analyze the sentence and respond with only 'Positive' or 'Negative'. "
                f"Do not include explanations or extra words.\n"
                f"[User]: \"{it['sentence']}\"\n[Assistant]:")
    if task == "QA":
        return (f"[System]: You are a world-class question answering model. "
                f"Given the context, provide a concise and accurate answer.\n"
                f"[User]: Context:\n\"{it['context']}\"\nQuestion:\n\"{it['question']}\"\n[Assistant]:")
    return (f"[System]: You are an expert summarization model. "
            f"Summarize the following article concisely and accurately in one paragraph.\n"
            f"[User]: {it['article']}\n[Assistant]:")

CFG = {"NLU": ("nyu-mll/glue", "sst2"), "QA": ("rajpurkar/squad_v2", None), "Summarization": ("abisee/cnn_dailymail", "3.0.0")}
TOK = {  # 데이터의 모델명 -> HF 토크나이저 ID
 "DeepSeek-R1-Distill-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
 "EXAONE-3.5-2.4B-Instruct": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
 "Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
 "Midm-2.0-Mini-Instruct": "K-intelligence/Midm-2.0-Mini-Instruct",
 "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
 "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
 "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
 "gemma-2b-it": "unsloth/gemma-2b-it",
 "opt-2.7b": "facebook/opt-2.7b"}

# 1) 프롬프트 복원 (모델과 무관)
prompts = []
for task, (name, sub) in CFG.items():
    ds = load_dataset(name, sub, split="validation") if sub else load_dataset(name, split="validation")
    ds = ds.shuffle(seed=42).select(range(100))
    for i, it in enumerate(ds):
        prompts.append(dict(Task=task, Sample=i + 1, prompt=create_prompt(task, it),
                            has_answer=(bool(it["answers"]["text"]) if task == "QA" else np.nan)))
P = pd.DataFrame(prompts)
P["in_chars"] = P.prompt.str.len()

# 2) 모델별 입력 토큰 수
rows, failed = [], []
for m, tid in TOK.items():
    try:
        tk = AutoTokenizer.from_pretrained(tid, trust_remote_code=True)
        n = [len(tk(p).input_ids) for p in P.prompt]
        rows.append(P.drop(columns="prompt").assign(Model=m, in_tok=n))
        print("OK  ", m, "mean in_tok =", round(np.mean(n), 1))
    except Exception as e:
        failed.append((m, str(e)[:100])); print("FAIL", m, "->", str(e)[:100])
if rows:
    pd.concat(rows).to_csv("input_features.csv", index=False)
    print("\nsaved input_features.csv  (models ok:", len(rows), "/ failed:", len(failed), ")")
P[["Task","Sample","in_chars"]].to_csv("input_chars_only.csv", index=False)
