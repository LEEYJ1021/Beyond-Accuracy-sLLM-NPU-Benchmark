import pandas as pd, numpy as np, os
os.makedirs("data", exist_ok=True)

# ---- trials.csv
t = pd.read_csv("Data_시계열.csv", encoding_errors="replace")
t = t.rename(columns={"Model":"model","Task":"task","Sample":"sample_idx"})
t["latency_ms"] = pd.to_numeric(t["latency_ms"], errors="coerce")
t["generated_tokens"] = pd.to_numeric(t["Tokens Generated"], errors="coerce")
def tk(x):
    s = str(x).strip().lower()
    return "NLU" if s.startswith("nlu") else "QA" if s.startswith("qa") else "Summarization" if s.startswith("summ") or s.startswith("sum") else x
t["task"] = t["task"].map(tk)
f = pd.read_csv("input_features.csv", encoding_errors="replace").rename(columns={"Model":"model","Task":"task","Sample":"sample_idx","in_tok":"input_tokens"})
f["task"] = f["task"].map(tk)
m = t.merge(f[["model","task","sample_idx","input_tokens"]], on=["model","task","sample_idx"], how="left", validate="one_to_one")
print("tasks:", m.task.unique().tolist())
print("latency/gen 결측(오류 시행):", m.latency_ms.isna().sum(), m.generated_tokens.isna().sum(),
      "| input_tokens 결측:", m.input_tokens.isna().sum())
print(m[m.latency_ms.isna() | m.generated_tokens.isna()][["model","task","sample_idx"]].to_string())
m = m.dropna(subset=["latency_ms","generated_tokens"])
print("유효 시행:", len(m), "(논문 2,696)")
m[["model","task","sample_idx","latency_ms","generated_tokens","input_tokens"]].to_csv("data/trials.csv", index=False)

# ---- models.csv
j = pd.read_csv("Data_정제.csv", encoding_errors="replace")
g = j.groupby("Model").agg(parameter_b=("파라미터_b","first"), peak_npu_mem_gb=("peak_npu_mem_휴","max")).reset_index()
print("peak mem가 과제마다 다른 모델:", j.groupby("Model")["peak_npu_mem_휴"].nunique().loc[lambda s: s>1].to_dict())
h = pd.read_csv("Data_허깅페이스.csv", encoding_errors="replace").set_index("항목")
ctx = h.loc["Context Length"].str.replace(",","").astype(int)
g["context_length"] = g.Model.map(ctx)
g = g.rename(columns={"Model":"model"})[["model","parameter_b","context_length","peak_npu_mem_gb"]]
print(g.to_string(index=False)); g.to_csv("data/models.csv", index=False)
