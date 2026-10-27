# figlib.py
import os, itertools, numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("figs", exist_ok=True); os.makedirs("out", exist_ok=True)
_f = open("out/numbers.txt", "a", encoding="utf-8")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); _f.write(s + "\n"); _f.flush()

CFG = dict(trials="data/trials.csv", models="data/models.csv",
           model="model", task="task", pid="sample_idx",
           lat="latency_ms", gen="generated_tokens", inp="input_tokens",
           params="parameter_b", ctx="context_length", mem="peak_npu_mem_gb")
ALIAS = {"Qwen2.5-0.5B-Instruct":"Qwen-0.5B", "DeepSeek-R1-Distill-Qwen-1.5B":"DeepSeek-1.5B",
         "Qwen2.5-1.5B-Instruct":"Qwen-1.5B", "gemma-2b-it":"gemma-2b",
         "EXAONE-3.5-2.4B-Instruct":"EXAONE-2.4B", "Midm-2.0-Mini-Instruct":"Midm-2.0",
         "opt-2.7b":"OPT-2.7B", "OPT-2.7B":"OPT-2.7B", "Qwen2.5-3B-Instruct":"Qwen-3B",
         "Llama-3.2-3B-Instruct":"Llama-3B"}
ORDER = ["Qwen-0.5B","DeepSeek-1.5B","Qwen-1.5B","gemma-2b","EXAONE-2.4B","Midm-2.0","OPT-2.7B","Qwen-3B","Llama-3B"]
TASKS = ["NLU","QA","Summarization"]
COL = dict(zip(ORDER, plt.cm.tab10.colors[:9]))

def load():
    c = CFG
    df = pd.read_csv(c["trials"])
    need = [c[k] for k in ("model","task","lat","gen","inp")]
    miss = [x for x in need if x not in df.columns]
    assert not miss, f"trials.csv에 없는 열: {miss} (CFG 수정 필요). 현재 열: {list(df.columns)}"
    ren = {c[k]: k for k in ("model","task","lat","gen","inp")}
    if c["pid"] in df.columns: ren[c["pid"]] = "pid"
    df = df.rename(columns=ren)
    df["model"] = df["model"].map(ALIAS).fillna(df["model"])
    df["task"] = df["task"].replace({"nlu":"NLU","qa":"QA","summarization":"Summarization","summ":"Summarization"})
    df = df.dropna(subset=["lat","gen"]).copy()
    if "pid" not in df.columns: df["pid"] = df.groupby(["model","task"]).cumcount()
    if len(df) != 2696: print(f"[경고] 유효 시행 {len(df)}건 (논문: 2,696)")
    assert set(df.model) == set(ORDER), f"모델명 불일치: {set(df.model) ^ set(ORDER)}"
    return df

def load_models(df):
    c = CFG
    mi = pd.read_csv(c["models"]).rename(columns={c["model"]:"model", c["params"]:"params", c["ctx"]:"ctx", c["mem"]:"mem"})
    mi["model"] = mi["model"].map(ALIAS).fillna(mi["model"])
    thr = df.groupby(["model","task"]).apply(lambda g: g.gen.sum() / (g.lat.sum() / 1000.0), include_groups=False).groupby("model").mean().rename("thr")
    return mi.merge(thr.reset_index(), on="model")

def style():
    mpl.rcParams.update({"font.size":8, "axes.titlesize":8.5, "axes.labelsize":8, "xtick.labelsize":7,
        "ytick.labelsize":7, "legend.fontsize":7, "axes.spines.top":False, "axes.spines.right":False,
        "axes.grid":True, "grid.alpha":.25, "savefig.dpi":300, "savefig.bbox":"tight"})

def panel(ax, s): ax.text(-0.13, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold")
def save(fig, name):
    fig.savefig(f"figs/{name}.png"); fig.savefig(f"figs/{name}.pdf"); plt.close(fig); log(f"[saved] figs/{name}.png")

def r2_ols(X, y):
    X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - (y - X @ b).var() / y.var()

def per_model_slope(df):
    return pd.Series({m: (stats.linregress(g.gen, g.lat).slope if g.gen.nunique() > 2 else np.nan)
                      for m, g in df.groupby("model")})

# Table A2 (논문 값) — OPS 재계산과 verify 단계용
A2 = pd.DataFrame({
 "model":["Qwen-0.5B","DeepSeek-1.5B","Llama-3B","Qwen-1.5B","OPT-2.7B","Qwen-3B","EXAONE-2.4B","gemma-2b","Midm-2.0"],
 "mean":[611.2,1205.3,1866.9,1025.9,1462.5,1996.3,932.1,753.2,1207.8],
 "cv":[.177,.433,.474,.624,.154,.459,.745,.816,.830], "skew":[-.04,-1.02,-1.00,-.70,1.76,-1.27,-.11,.29,.34],
 "kurt":[4.09,-.31,-.42,-1.28,3.00,.22,-1.49,-1.31,-1.30], "tail":[1.33,1.25,1.26,1.21,1.42,1.25,1.51,2.47,2.90],
 "ops":[.186,.270,.287,.306,.315,.318,.375,.631,.731]}).set_index("model")