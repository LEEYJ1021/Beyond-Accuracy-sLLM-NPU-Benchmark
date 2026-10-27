import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
OUT = "revision_out/figures/"

# ---------- Fig 3 ----------
T = pd.read_csv("revision_out/new_TableA2.csv", index_col=0, encoding="utf-8-sig").set_index("short")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.15]})
t = T.sort_values("OPS")
ax[0].barh(t.index, t["OPS"], color=["#c0504d" if v > 0.6 else "#4f81bd" for v in t["OPS"]])
ax[0].axvline(0.6, ls="--", c="r", lw=1)
ax[0].set_xlabel("OPS (descriptive; higher = less stable)")
ax[0].set_title("(a) OPS ranking", fontsize=10)

off = {"Midm-2.0": (-8, -12, "right"), "gemma-2b": (-8, 4, "right"), "EXAONE-2.4B": (-6, 6, "right"),
       "OPT-2.7B": (6, 6, "left"), "Qwen-0.5B": (6, -12, "left"), "Qwen-1.5B": (4, -13, "left"),
       "DeepSeek-1.5B": (-4, -14, "right"), "Llama-3B": (4, 9, "left"), "Qwen-3B": (14, -14, "left")}
sc = ax[1].scatter(T["cv"], T["tail"], c=T["OPS"], s=70, cmap="viridis", vmin=0.15, vmax=0.75, edgecolor="k", lw=0.4)
for m, r in T.iterrows():
    dx, dy, ha = off.get(m, (5, 5, "left"))
    ax[1].annotate(m, (r["cv"], r["tail"]), xytext=(dx, dy), textcoords="offset points",
                   fontsize=7.5, ha=ha, arrowprops=dict(arrowstyle="-", lw=0.4, color="gray"))
ax[1].set_xlim(0.1, 0.95); ax[1].set_ylim(1.1, 3.05)
ax[1].set_xlabel("CV"); ax[1].set_ylabel("P95/P50")
ax[1].set_title("(b) Distributional cost map", fontsize=10)
cb = fig.colorbar(sc, ax=ax[1], pad=0.02); cb.set_label("OPS")
plt.tight_layout(); plt.savefig(OUT + "Fig3_ops.png", dpi=250); plt.close()

# ---------- Fig 10 ----------
d = pd.read_csv("Data_정제.csv", encoding="utf-8-sig")
g = d.groupby("Model").agg(thr=("throughput_tokens_per_sec", "mean"),
                           mem=("peak_npu_mem_휴", "first"), par=("파라미터_b", "first"))
short = {"DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-1.5B", "EXAONE-3.5-2.4B-Instruct": "EXAONE-2.4B",
         "gemma-2b-it": "gemma-2b", "Llama-3.2-3B-Instruct": "Llama-3B", "Midm-2.0-Mini-Instruct": "Midm-2.0",
         "opt-2.7b": "OPT-2.7B", "Qwen2.5-0.5B-Instruct": "Qwen-0.5B", "Qwen2.5-1.5B-Instruct": "Qwen-1.5B",
         "Qwen2.5-3B-Instruct": "Qwen-3B"}
g.index = [short.get(i, i) for i in g.index]
rho = stats.spearmanr(g["mem"], g["thr"])[0]
g2 = g.drop("Llama-3B"); rho2 = stats.spearmanr(g2["mem"], g2["thr"])[0]
print(f"rho all = {rho:.3f}; rho without Llama = {rho2:.3f}")
imp = {"Peak memory": 0.61, "Parameters": 0.22, "Context": 0.0}
try:
    k = pd.read_csv("revision_out/tables/K2_H4_grouped_rf.csv", encoding="utf-8-sig"); print(k.to_string())
except Exception as e:
    print("K2 not read:", e)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cols = ["#d62728" if m == "Llama-3B" else "#1f77b4" for m in g.index]
ax[0].scatter(g["mem"], g["thr"], c=cols, s=55)
offs = {"Qwen-0.5B": (8, -4), "Llama-3B": (-8, 8), "Qwen-3B": (8, 10), "Midm-2.0": (-10, -14),
        "EXAONE-2.4B": (10, -6), "Qwen-1.5B": (-14, -24), "DeepSeek-1.5B": (4, 26), "gemma-2b": (34, -30),
        "OPT-2.7B": (12, 14)}
pos = {"Qwen-0.5B": (3.6, 77), "Llama-3B": (13.2, 24.5), "Qwen-3B": (9.6, 24.5), "Midm-2.0": (5.4, 17.5),
       "EXAONE-2.4B": (8.4, 29.0), "Qwen-1.5B": (3.8, 28.5), "DeepSeek-1.5B": (4.0, 45.0),
       "gemma-2b": (6.3, 24.5), "OPT-2.7B": (8.2, 44.0)}
for m, r in g.iterrows():
    tx, ty = pos.get(m, (r["mem"] + 0.3, r["thr"] + 2))
    ax[0].annotate(m, (r["mem"], r["thr"]), xytext=(tx, ty), textcoords="data", fontsize=7.5,
                   ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color="gray"))
ax[0].set_ylim(14, 84); ax[0].set_xlabel("Peak NPU memory (GB)"); ax[0].set_ylabel("Throughput (tokens/s)")
ax[0].set_title(f"(a) Spearman ρ = {rho:.2f} (without Llama, red: {rho2:.2f})", fontsize=10)
ax[1].bar(list(imp), list(imp.values()), color="#1f77b4")
ax[1].text(2, 0.02, "≈ 0", ha="center", fontsize=9)
ax[1].set_ylabel("Permutation importance")
ax[1].set_title("(b) Grouped importance; leave-one-model-out CV R² = 0.13", fontsize=10)
plt.tight_layout(); plt.savefig(OUT + "Fig10_memory_throughput.png", dpi=250); plt.close()

# ---------- Fig 1 ----------
fig, ax = plt.subplots(figsize=(11, 1.9)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
steps = ["① Measure\nprompt, latency,\ngenerated tokens", "② Decompose\noutput length,\ninput length, task",
         "③ Characterize\nCV, P95/P50, OPS,\ntail shape", "④ Decide\ntask budgets, output cap,\npreemption quantum,\nlatency bound"]
w, gap = 0.215, 0.035
for i, s in enumerate(steps):
    x0 = 0.02 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x0, 0.1), w, 0.8, boxstyle="round,pad=0.005,rounding_size=0.02", fc="#dbe5f1", ec="k", lw=1))
    ax.text(x0 + w / 2, 0.5, s, ha="center", va="center", fontsize=8.5)
    if i < 3:
        ax.annotate("", xy=(x0 + w + gap - 0.006, 0.5), xytext=(x0 + w + 0.006, 0.5), arrowprops=dict(arrowstyle="->", lw=1.3))
plt.savefig(OUT + "Fig1_workflow.png", dpi=250, bbox_inches="tight"); plt.close()
print("done")
