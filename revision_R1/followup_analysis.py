import heapq, numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import additional_analysis as aa

ST = aa.Ctx(); ST.base = aa.DEFAULT_BASE; ST.out = ST.base + "/revision_out"
aa.load(ST); d = ST.d; rng = np.random.default_rng(42)
T = ["NLU", "QA", "Summarization"]
pd.set_option("display.width", 220, "display.max_columns", 30)
def out(name, df):
    df.to_csv(f"{ST.out}/tables/{name}.csv", encoding="utf-8-sig")
    print(f"\n=== {name} ==="); print(df.to_string())

# L0. 셀 구조 점검 (Sample 중복 여부, 셀당 n)
g = d.groupby(["Model", "Task"]).Sample.agg(n="size", n_unique="nunique", smin="min", smax="max")
out("L0_cell_structure", g)

# L1. 모델 내 CV (H3의 올바른 단위) + Friedman / Wilcoxon
cv = d.groupby(["Model", "Task"]).lat.agg(lambda s: s.std(ddof=1) / s.mean()).unstack()[T]
out("L1_within_model_cv", cv.round(3))
print("Friedman:", stats.friedmanchisquare(*[cv[t] for t in T]))
pw = []
for a, b in [("NLU", "QA"), ("NLU", "Summarization"), ("QA", "Summarization")]:
    w = stats.wilcoxon(cv[a], cv[b]); pw.append({"pair": f"{a} vs {b}", "W": w.statistic, "p": w.pvalue,
                                                 "median_diff": float((cv[a] - cv[b]).median())})
pw = pd.DataFrame(pw); pw["p_BH"] = aa.bh(pw.p); out("L1b_wilcoxon_cv", pw.round(4).set_index("pair"))

# L2. 모델 내 tail ratio (과제별 / tok=50 고정)
tr = lambda x: x.quantile(.95) / x.median()
t_all = d.groupby(["Model", "Task"]).lat.apply(tr).unstack()[T]
t50 = d[d.tok == 50].groupby(["Model", "Task"]).lat.apply(tr).unstack().reindex(columns=T)
out("L2_tail_ratio_by_task", pd.concat({"all": t_all, "tok50": t50}, axis=1).round(2))

# L3. 라벨-only 비율 (NLU/QA에서 tok<=5 비중) + 이원성 dip test
rows = []
try:
    from diptest import diptest
except Exception:
    diptest = None
for (m, t), x in ST.cells.items():
    rows.append({"Model": m, "Task": t, "frac_tok_le5": (x.tok <= 5).mean(), "frac_tok_eq50": (x.tok >= 50).mean(),
                 "dip_p": diptest(x.lat.values)[1] if diptest else np.nan})
out("L3_bimodality_by_cell", pd.DataFrame(rows).round(3).set_index(["Model", "Task"]))

# L4. H1의 모델 수준 검정 (n=9): (Summ-NLU) 평균 지연차 ~ size, ms/token 버전 포함
m = d.groupby(["Model", "Task"]).agg(lat=("lat", "mean"), mspt=("mspt", "mean")).unstack()
r = pd.DataFrame({"size": m.index.map(aa.PARAMS)}, index=m.index)
r["d_lat"] = m[("lat", "Summarization")] - m[("lat", "NLU")]
r["d_mspt"] = m[("mspt", "Summarization")] - m[("mspt", "NLU")]
r["ratio_lat"] = m[("lat", "Summarization")] / m[("lat", "NLU")]
out("L4a_model_level_H1", r.round(2))
for y in ["d_lat", "d_mspt"]:
    rho, p = stats.spearmanr(r["size"], r[y]); f = smf.ols(f"{y} ~ size", r).fit()
    print(f"{y}: spearman={rho:.3f} (p={p:.3f}), OLS slope={f.params['size']:.1f}, p={f.pvalues['size']:.3f}")

# L5. 토큰 수 통제 후 과제 효과 (prefill 대리) + 모델별 ms/token
rows = []
for mod in ST.models:
    g = d[d.Model == mod]
    f = smf.ols("lat ~ tok + C(Task, Treatment('NLU'))", g).fit()
    rows.append({"Model": mod, "ms_per_tok": f.params["tok"], "R2": f.rsquared,
                 "QA_vs_NLU_ms": f.params.get("C(Task, Treatment('NLU'))[T.QA]", np.nan),
                 "Summ_vs_NLU_ms": f.params.get("C(Task, Treatment('NLU'))[T.Summarization]", np.nan)})
out("L5_token_controlled_task_effect", pd.DataFrame(rows).round(2).set_index("Model"))

# L6. 고정 주기/데드라인 시뮬레이션 (모델 무관 요구사항), 비선점 vs 이상적 선점 EDF
def sim_fixed(arrs, P, H, preempt):
    jobs = []
    for k, (a, p) in enumerate(zip(arrs, P)):
        n = int(H // p); jobs += [(i * p, (i + 1) * p, k, float(e)) for i, e in enumerate(rng.choice(a, n))]
    jobs.sort(); t, i, rd = 0.0, 0, []; miss, tot = np.zeros(len(P)), np.zeros(len(P))
    while i < len(jobs) or rd:
        if not rd and t < jobs[i][0]: t = jobs[i][0]
        while i < len(jobs) and jobs[i][0] <= t:
            r_, dl, k, e = jobs[i]; heapq.heappush(rd, [dl, r_, k, e]); i += 1
        j = rd[0]
        if preempt:
            nxt = jobs[i][0] if i < len(jobs) else float("inf")
            run = min(j[3], nxt - t); t += run; j[3] -= run
            if j[3] <= 1e-9:
                heapq.heappop(rd); tot[j[2]] += 1; miss[j[2]] += t > j[0]
        else:
            heapq.heappop(rd); t += j[3]; tot[j[2]] += 1; miss[j[2]] += t > j[0]
    return miss / tot, miss.sum() / tot.sum()

PERIOD_SETS = {"tight": [1000, 4000, 8000], "relaxed": [1500, 6000, 12000]}   # ms, 필요 시 수정
rows = []
for mod in ST.models:
    arrs = [ST.cells[(mod, t)].lat.values for t in T]
    for name, P in PERIOD_SETS.items():
        util = sum(a.mean() / p for a, p in zip(arrs, P))
        for pol, pre in [("EDF-nonpreempt", False), ("EDF-preempt(ideal)", True)]:
            pm, ov = sim_fixed(arrs, P, 2_000_000, pre)
            rows.append({"Model": mod, "periods": name, "util_mean": round(util, 2), "policy": pol,
                         **{f"miss_{t}": v for t, v in zip(T, pm)}, "miss_all": ov})
out("L6_fixed_period_sim", pd.DataFrame(rows).round(3).set_index(["Model", "periods", "policy"]))

# L7. 정적 상한(WCET proxy) = 관측 max vs prefill + 50*ms/token 모델
rows = []
for mod in ST.models:
    g = d[d.Model == mod]; f = smf.ols("lat ~ tok", g).fit()
    rows.append({"Model": mod, "obs_max": g.lat.max(), "obs_p99": g.lat.quantile(.99),
                 "fit_at_50tok": f.params["Intercept"] + 50 * f.params["tok"],
                 "intercept_ms": f.params["Intercept"], "ms_per_tok": f.params["tok"]})
out("L7_bound_vs_observed", pd.DataFrame(rows).round(1).set_index("Model"))
print("\nDone. tables ->", ST.out + "/tables/L*.csv")
