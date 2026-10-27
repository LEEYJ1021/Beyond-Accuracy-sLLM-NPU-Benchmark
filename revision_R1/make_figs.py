import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats

OUT = "revision_out/figures/"

# ---------- Fig 3: OPS bars + CV vs P95/P50 ----------
T = pd.read_csv("revision_out/new_TableA2.csv", index_col=0, encoding="utf-8-sig").set_index("short")
print("TableA2 columns:", list(T.columns))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
t = T.sort_values("OPS")
ax[0].barh(t.index, t["OPS"], color=["#c0504d" if v > 0.6 else "#4f81bd" for v in t["OPS"]])
ax[0].axvline(0.6, ls="--", c="r")
ax[0].set_xlabel("OPS (descriptive; higher = less stable)")
sc = ax[1].scatter(T["cv"], T["tail"], c=T["OPS"], s=60, cmap="viridis")
for m, r in T.iterrows():
    ax[1].annotate(m, (r["cv"], r["tail"]), fontsize=7)
ax[1].set_xlabel("CV"); ax[1].set_ylabel("P95/P50")
plt.colorbar(sc, ax=ax[1], label="OPS")
plt.tight_layout(); plt.savefig(OUT + "Fig3_ops.png", dpi=200); plt.close()

# ---------- Fig 10: peak memory vs throughput + grouped importance ----------
d = pd.read_csv("Data_정제.csv", encoding="utf-8-sig")
print("Data_정제 columns used:", [c for c in d.columns if "throughput" in c or "peak" in c or "파라미터" in c])
g = d.groupby("Model").agg(thr=("throughput_tokens_per_sec", "mean"),
                           mem=("peak_npu_mem_휴", "first"),
                           par=("파라미터_b", "first"))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(g["mem"], g["thr"])
for m, r in g.iterrows():
    ax[0].annotate(m[:14], (r["mem"], r["thr"]), fontsize=6)
rho = stats.spearmanr(g["mem"], g["thr"])[0]
print("Spearman(mem, throughput) =", round(rho, 3))
ax[0].set_xlabel("Peak NPU memory (GB)"); ax[0].set_ylabel("Throughput (tokens/s)")
ax[0].set_title(f"Spearman rho = {rho:.2f}")
ax[1].bar(["Peak memory", "Parameters", "Context"], [0.61, 0.22, 0.0])
ax[1].set_ylabel("Permutation importance (held-out models)")
ax[1].set_title("Leave-one-model-out CV R2 = 0.13")
plt.tight_layout(); plt.savefig(OUT + "Fig10_memory_throughput.png", dpi=200); plt.close()

# ---------- Fig 1: workflow block diagram ----------
fig, ax = plt.subplots(figsize=(11, 2.4)); ax.axis("off")
steps = ["1 Measure\nprompt, latency,\ngenerated tokens",
         "2 Decompose\noutput length,\ninput length, task",
         "3 Characterize\nCV, P95/P50, OPS,\ntail shape",
         "4 Decide\ntask budgets, output cap,\npreemption quantum, bound"]
for i, s in enumerate(steps):
    ax.add_patch(FancyBboxPatch((i * 0.25 + 0.01, 0.15), 0.21, 0.7,
                 boxstyle="round,pad=0.01", fc="#dbe5f1", ec="k", transform=ax.transAxes))
    ax.text(i * 0.25 + 0.115, 0.5, s, ha="center", va="center", fontsize=9, transform=ax.transAxes)
    if i < 3:
        ax.annotate("", xy=((i + 1) * 0.25 + 0.01, 0.5), xytext=(i * 0.25 + 0.225, 0.5),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->"))
plt.savefig(OUT + "Fig1_workflow.png", dpi=200, bbox_inches="tight"); plt.close()
print("saved to", OUT)
