import warnings; warnings.filterwarnings("ignore")
# run_all.py
import sys, os
if os.path.exists("out/numbers.txt") and len(sys.argv) == 1: os.remove("out/numbers.txt")
from figlib import *
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

df = load(); mi = load_models(df); STEPS = {}
def step(f): STEPS[f.__name__] = f; return f

# ---------------------------------------------------------------- verify
@step
def verify():
    g = df.groupby("model").lat
    T = pd.DataFrame({"mean": g.mean(), "cv": g.std()/g.mean(), "p95p50": g.quantile(.95)/g.median()}).loc[ORDER]
    ref = A2.loc[ORDER]
    D = pd.DataFrame({"mean_diff": T["mean"]-ref["mean"], "cv_diff": T.cv-ref.cv, "p95p50_diff": T.p95p50-ref["tail"]})
    log("== verify: 데이터 대 Table A2 (차이가 mean±1 ms, cv±.005, P95/P50±.02 안이어야 함)"); log(D.round(3).to_string())
    log("n per model:", df.groupby("model").size().to_dict())

# ---------------------------------------------------------------- OPS (Table C1, OPS-core, Fig 3c)
@step
def ops():
    mm = lambda x: (x - x.min()) / (x.max() - x.min())
    Z = pd.DataFrame({"cv": mm(A2.cv), "tail": mm(A2["tail"]), "skew": mm(A2["skew"].abs()), "kurt": mm(A2["kurt"])})
    S = {"baseline":(.35,.35,.15,.15), "equal":(.25,.25,.25,.25), "cv_heavy":(.45,.25,.15,.15),
         "tail_heavy":(.25,.45,.15,.15), "core":(.5,.5,0,0)}
    sc = pd.DataFrame({k: Z.values @ np.array(w) for k, w in S.items()}, index=Z.index)
    rk = sc.rank(method="min").astype(int)
    log("== Table C1 점수\n", sc.round(3).to_string(), "\n== 순위(1=가장 안정)\n", rk.to_string())
    log("baseline vs A2 OPS max abs diff:", round((sc.baseline - A2.ops).abs().max(), 4), "(≤0.001이어야 함)")
    log("Spearman(OPS, OPS-core) =", round(stats.spearmanr(sc.baseline, sc.core)[0], 2))
    sc.round(3).to_csv("out/ops_scores.csv")
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for m in sc.index:
        ax.plot([0, 1], [rk.loc[m, "baseline"], rk.loc[m, "core"]], "-o", color=COL[m], lw=1.3)
        ax.text(-.05, rk.loc[m, "baseline"], m, ha="right", va="center", fontsize=6.5)
        ax.text(1.05, rk.loc[m, "core"], m, ha="left", va="center", fontsize=6.5)
    ax.set_xlim(-1.9, 2.9); ax.set_xticks([0, 1]); ax.set_xticklabels(["OPS", "OPS-core"]); ax.set_ylabel("Rank (1 = most stable)")
    ax.invert_yaxis(); ax.grid(False); save(fig, "fig03c")

# ---------------------------------------------------------------- Fig 4 (d), (e)
@step
def fig04de():
    style(); SH = 5; rng = np.random.default_rng(0)
    def curve(g, grid, B=20000):
        s, l = g[g.gen <= SH].lat.values, g[g.gen > SH].lat.values; out = []
        for pi in grid:
            k = rng.binomial(B, pi); x = np.concatenate([rng.choice(s, k), rng.choice(l, B - k)])
            out.append(np.percentile(x, 95) / np.percentile(x, 50))
        return np.array(out)
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.7))
    g = df[df.model == "Midm-2.0"]; bins = np.logspace(np.log10(g.lat.min()), np.log10(g.lat.max()), 40)
    ax[0].hist(g[g.gen <= SH].lat, bins, color="#4C78A8", label="label-only (≤5 tokens)")
    ax[0].hist(g[g.gen > SH].lat, bins, color="#F58518", label="free-running")
    for q, ls in ((50, "-"), (95, "--")):
        v = np.percentile(g.lat, q); ax[0].axvline(v, color="k", ls=ls, lw=1); ax[0].text(v, ax[0].get_ylim()[1]*.92, f" P{q}", fontsize=7)
    ax[0].set_xscale("log"); ax[0].set_xlabel("Latency (ms), Midm-2.0, all tasks"); ax[0].set_ylabel("Trials"); ax[0].legend(frameon=False, fontsize=5.2, handlelength=1, columnspacing=.8, loc="lower left", bbox_to_anchor=(0.1, 1.0), ncol=2)
    grid = np.linspace(.02, .98, 49); used, skipped = [], []
    for m in ORDER:
        g = df[df.model == m]
        if (g.gen <= SH).sum() < 20 or (g.gen > SH).sum() < 20: skipped.append(m); continue
        ax[1].plot(grid, curve(g, grid), color=COL[m], lw=1.1, label=m); used.append(m)
        ax[1].scatter((g.gen <= SH).mean(), np.percentile(g.lat, 95)/np.percentile(g.lat, 50), color=COL[m], edgecolor="k", zorder=3, s=26)
    ax[1].set_xlabel("Share of label-only outputs, π"); ax[1].set_ylabel("Pooled P95/P50"); ax[1].legend(frameon=False, fontsize=6)
    panel(ax[0], "(d)"); panel(ax[1], "(e)"); log("fig04e 제외 모델(캡션에 기재):", skipped); save(fig, "fig04de")

# ---------------------------------------------------------------- Fig 5 + 문맥 길이 ρ, p
@step
def fig05():
    style(); sl = per_model_slope(df).rename("slope").reset_index().rename(columns={"index":"model"}).merge(mi, on="model")
    ok = sl.dropna(subset=["slope"]); rho, p = stats.spearmanr(ok.ctx, ok.slope)
    log(f"== §4.3 문맥 길이: n={len(ok)} distinct ctx={ok.ctx.nunique()}  Spearman rho={rho:.2f}, p={p:.3f}   ← 본문 [ρ], [p]")
    log("slopes (Table A5와 ±0.2 ms 이내여야 함):", ok.set_index("model").slope.round(1).to_dict())
    fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.0), gridspec_kw={"width_ratios":[1,1,1.15], "wspace":.45})
    for a, x, xl in ((ax[0], "ctx", "Model-card context length"), (ax[1], "params", "Parameters (B)")):
        for _, r in ok.iterrows():
            a.scatter(r[x], r.slope, color=COL[r.model], s=22, zorder=3, label=r.model)
        a.set_xlabel(xl); a.set_ylabel("Per-token cost (ms/token)")
    ax[0].set_xscale("log"); h_, l_ = ax[1].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=6, frameon=False, loc="lower center", ncol=8, bbox_to_anchor=(0.4, -0.16))
    H = df.pivot_table(index="model", columns="task", values="gen", aggfunc="mean").loc[ORDER, TASKS]
    ax[2].imshow(H.values, cmap="viridis", vmin=0, vmax=50, aspect="auto")
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(TASKS); ax[2].set_yticks(range(9)); ax[2].set_yticklabels(ORDER); ax[2].grid(False)
    for i in range(9):
        for j in range(3): ax[2].text(j, i, f"{H.values[i,j]:.0f}", ha="center", va="center", color="w" if H.values[i,j] < 30 else "k", fontsize=7)
    for a, s in zip(ax, "abc"): panel(a, f"({s})")
    save(fig, "fig05")

# ---------------------------------------------------------------- Fig 7 하단 패널 + 크기 그룹 정의
@step
def fig07():
    style(); P = mi.set_index("model").params.loc[ORDER]; srt = P.sort_values(kind="stable")
    grp = {m: ("Small", "Medium", "Large")[i // 3] for i, m in enumerate(srt.index)}   # 파라미터 3분위(3·3·3)
    for gname in ("Small", "Medium", "Large"):
        ms = [m for m in srt.index if grp[m] == gname]; log(f"== Fig.7 {gname}: {ms}  ({P[ms].min()}–{P[ms].max()}B)  ← 캡션 [..]B")
    d = df.assign(group=df.model.map(grp)); fig, ax = plt.subplots(figsize=(4.2, 3.2)); cl = {"Small":"#3B4C8C", "Medium":"#1F9A8A", "Large":"#5DC863"}
    for gname in cl:
        q = [d[(d.group == gname) & (d.task == t)].lat.quantile([.25, .5, .75]).values for t in TASKS]
        med = np.array([x[1] for x in q]); lo = med - np.array([x[0] for x in q]); hi = np.array([x[2] for x in q]) - med
        ax.errorbar(np.arange(3) + {"Small": -.07, "Medium": 0, "Large": .07}[gname], med, yerr=[lo, hi], marker="o", capsize=4, color=cl[gname], label=gname)
    ax.set_xticks(range(3)); ax.set_xticklabels(TASKS); ax.set_ylabel("Latency (ms), median and IQR"); ax.legend(frameon=False, title="Model size group", loc="center left", bbox_to_anchor=(1.02, .5))
    save(fig, "fig07_bottom")   # 상단 3개 산점 패널은 기존 이미지 유지 후 합성

# ---------------------------------------------------------------- Fig 8c
@step
def fig08c():
    style(); rows = []
    for (m, t), g in df.groupby(["model", "task"]):
        y = g.lat.values.astype(float)
        X = g[["gen", "inp"]].values.astype(float) if g.gen.nunique() > 1 else g[["inp"]].values.astype(float)
        g50 = g[g.gen == 50]
        r50 = r2_ols(g50[["inp"]].values.astype(float), g50.lat.values.astype(float)) if len(g50) > 10 else np.nan
        rows.append((m, t, r2_ols(X, y), r50, len(g50)))
    R = pd.DataFrame(rows, columns=["model", "task", "r2_all", "r2_50tok_inp_only", "n50"])
    log("== Fig.8c (요약의 r2_50tok_inp_only가 Fig.9의 0.96–0.97과 같아야 함)\n", R.round(2).to_string())
    fig, ax = plt.subplots(figsize=(3.5, 2.6)); rng = np.random.default_rng(0)
    for i, t in enumerate(TASKS):
        s = R[R.task == t]; ax.scatter(np.full(len(s), i) + rng.uniform(-.12, .12, len(s)), s.r2_all, c=[COL[m] for m in s.model], s=20, zorder=3)
        ax.hlines(s.r2_all.median(), i-.3, i+.3, color="k", lw=1.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(TASKS); ax.set_ylabel("R² (latency ~ generated + input tokens)"); ax.set_ylim(0, 1.05); ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COL[m], label=m, markersize=4) for m in ORDER], fontsize=5.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, .5)); save(fig, "fig08c")

# ---------------------------------------------------------------- Fig 9
@step
def fig09():
    style(); S = df[(df.task == "Summarization") & (df.gen == 50)]
    def cv_rmse(x, y, kind, k=10):
        idx = np.random.default_rng(0).permutation(len(x)); err = []
        f = "y ~ x" if kind == "lin" else f"y ~ cr(x, df=4, lower_bound={x.min()}, upper_bound={x.max()}, constraints='center')"
        for te in np.array_split(idx, k):
            tr = np.setdiff1d(idx, te); fit = smf.ols(f, pd.DataFrame({"x": x[tr], "y": y[tr]})).fit()
            err.append((y[te] - fit.predict(pd.DataFrame({"x": x[te]})).values) ** 2)
        return np.sqrt(np.concatenate(err).mean())
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.8), gridspec_kw={"hspace": .65, "wspace": .3}); rows = []
    for ax, m in zip(axes.flat, ORDER):
        g = S[S.model == m]; x = g.inp.values.astype(float); y = g.lat.values.astype(float)
        lr = stats.linregress(x, y); gain = (cv_rmse(x, y, "lin") - cv_rmse(x, y, "spl")) / cv_rmse(x, y, "lin") * 100
        xs = np.linspace(x.min(), x.max(), 100)
        sp = smf.ols(f"y ~ cr(x, df=4, lower_bound={x.min()}, upper_bound={x.max()}, constraints='center')", pd.DataFrame({"x": x, "y": y})).fit()
        ax.scatter(x, y, s=6, alpha=.6, color=COL[m]); ax.plot(xs, lr.intercept + lr.slope*xs, color="crimson", lw=1.2)
        ax.plot(xs, sp.predict(pd.DataFrame({"x": xs})), "k--", lw=.9); ax.set_title(m, color=COL[m], fontweight="bold", pad=12)
        ax.text(.5, 1.0, f"n={len(g)}  R²={lr.rvalue**2:.2f}  {lr.slope:.3f} ms/token  spline {gain:+.1f}%", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.5)
        rows.append((m, len(g), lr.rvalue**2, lr.slope, gain, x.min(), x.max()))
    for ax in axes[-1]: ax.set_xlabel("Input tokens")
    for ax in axes[:, 0]: ax.set_ylabel("Latency (ms)")
    T = pd.DataFrame(rows, columns=["model", "n", "R2", "ms_per_in_tok", "spline_gain_%", "inp_min", "inp_max"])
    log("== Fig.9 / §4.3 / §5.4 / 초록 (R², 기울기, 스플라인 이득 범위)\n", T.round(3).to_string())
    log(f"R2 range {T.R2.min():.2f}–{T.R2.max():.2f}; slope range {T.ms_per_in_tok.min():.2f}–{T.ms_per_in_tok.max():.2f}; gain range {T['spline_gain_%'].min():+.1f} to {T['spline_gain_%'].max():+.1f}")
    log("OPT-2.7B 요약 입력 최대 토큰:", int(T.set_index('model').loc['OPT-2.7B', 'inp_max']), "(컴파일 최대 시퀀스 길이/2,048과 비교 → 절단 여부 확인)")
    log("1 생성 토큰 / 1 입력 토큰 비율 범위:", round(per_model_slope(df).min() / T.ms_per_in_tok.max(), 0), "–", round(per_model_slope(df).max() / T.ms_per_in_tok.min(), 0))
    save(fig, "fig09")

# ---------------------------------------------------------------- Fig 10
@step
def fig10():
    style()
    def shap(g):
        y = g.lat.values.astype(float)
        cols = {"gen": g.gen.values[:, None].astype(float), "inp": g.inp.values[:, None].astype(float), "task": pd.get_dummies(g.task, drop_first=True).values.astype(float)}
        v = lambda S: 0.0 if not S else r2_ols(np.column_stack([cols[s] for s in S]), y)
        phi = dict.fromkeys(cols, 0.0); perms = list(itertools.permutations(cols))
        for p in perms:
            S = []
            for n in p: phi[n] += (v(S + [n]) - v(S)) / len(perms); S.append(n)
        return phi, v(list(cols))
    res = {m: shap(g) for m, g in df.groupby("model")}
    A = pd.DataFrame({m: {**p, "R2": r} for m, (p, r) in res.items()}).T.loc[ORDER]; A["unexpl"] = 1 - A.R2
    N = A[["gen", "inp", "task"]].div(A.R2, axis=0)
    log("== Fig.10 Shapley (절대 몫)\n", A.round(3).to_string()); log("정규화 몫 (OPT 제외 중앙값):", N.drop("OPT-2.7B").median().round(3).to_dict(),
        "| gen 범위:", round(N.drop("OPT-2.7B").gen.min(), 2), "–", round(N.drop("OPT-2.7B").gen.max(), 2), "| input+task:", round(N.drop("OPT-2.7B")[["inp", "task"]].sum(axis=1).median(), 2))
    log("OPT gen 몫(0이어야 함):", round(A.loc["OPT-2.7B", "gen"], 4))
    order = A.drop("OPT-2.7B").gen.sort_values().index.tolist() + ["OPT-2.7B"]
    cl = {"gen": "#4C78A8", "inp": "#F58518", "task": "#54A24B", "unexpl": "#BAB0AC"}; lab = {"gen": "Generated tokens", "inp": "Input tokens", "task": "Task", "unexpl": "Unexplained"}
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
    for a, norm in zip(ax, (False, True)):
        left = np.zeros(len(order))
        for k in ("gen", "inp", "task") + (() if norm else ("unexpl",)):
            v = (N.loc[order, k] if norm else A.loc[order, k]).values; a.barh(order, v, left=left, color=cl[k], label=lab[k]); left += v
        a.set_xlim(0, 1); a.set_xlabel("Share of explained variance" if norm else "Share of total variance")
    for i, m in enumerate(order): ax[0].text(1.01, i, f"R²={A.loc[m, 'R2']:.2f}", va="center", fontsize=6.5)
    ax[0].legend(loc="lower center", bbox_to_anchor=(1.0, -.32), ncol=4, frameon=False); panel(ax[0], "(a)"); panel(ax[1], "(b)"); save(fig, "fig10")

# ---------------------------------------------------------------- Fig 11
@step
def fig11():
    style(); B = df.groupby("model").lat.quantile(.95); d = df.assign(ex=df.lat > df.model.map(B))
    EX = d.groupby(["model", "task"]).ex.mean().unstack()[TASKS].loc[ORDER] * 100
    P95 = d.groupby(["model", "task"]).lat.quantile(.95).unstack()[TASKS].loc[ORDER]; RT = pd.DataFrame({t: B.loc[ORDER] / P95[t] for t in TASKS})
    cv = d.groupby("task").lat.agg(lambda x: x.std() / x.mean())
    log("== Fig.11 균일 P95 예산 초과율(%)\n", EX.round(1).to_string(), "\n예산/과제 P95 (배)\n", RT.round(1).to_string(), "\npooled CV:", cv.round(2).to_dict())
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.2, 1.4, 1.4], "wspace": .55})
    ax[0].boxplot([d[d.task == t].lat for t in TASKS], showfliers=False); ax[0].set_xticklabels(TASKS)
    ax[0].set_xticklabels([f"{t}\nCV={cv[t]:.2f}" for t in TASKS], fontsize=6.5)
    ax[0].set_ylabel("Latency (ms)"); w = .27; tc = ["#4C78A8", "#F58518", "#54A24B"]
    for j, t in enumerate(TASKS):
        ax[1].bar(np.arange(9) + (j - 1) * w, EX[t], w, color=tc[j], label=t); ax[2].bar(np.arange(9) + (j - 1) * w, RT[t], w, color=tc[j])
    ax[2].set_yscale("log"); ax[2].axhline(1, color="k", lw=.8)
    for a in ax[1:]: a.set_xticks(range(9)); a.set_xticklabels(ORDER, rotation=60, ha="right")
    ax[1].set_ylabel("Requests above uniform P95 budget (%)"); ax[2].set_ylabel("Budget / task P95 (×)"); ax[1].set_ylim(0, 22); ax[1].legend(frameon=False, fontsize=6, loc="upper left", ncol=3)
    for a, s in zip(ax, "abc"): panel(a, f"({s})")
    save(fig, "fig11")

# ---------------------------------------------------------------- Fig 14 (Cantelli)
@step
def fig14():
    style(); mu = df.groupby("model").lat.mean(); cv = df.groupby("model").lat.std() / mu
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.6), sharey=True); gr = np.linspace(0, 1, 200)
    for a, k in zip(ax, (1.5, 2, 3)):
        miss = (df.lat > k * df.model.map(mu)).groupby(df.model).mean(); bound = cv**2 / (cv**2 + (k - 1)**2)
        a.plot(gr, gr**2 / (gr**2 + (k - 1)**2), "k--", lw=1)
        for m in ORDER: a.scatter(cv[m], miss[m], color=COL[m], s=24, zorder=3, label=m)
        a.set_title(f"D = {k}× mean"); a.set_xlabel("CV"); log(f"k={k} 미스율:", miss.round(3).to_dict(), "| 모든 점이 상한 아래:", bool((miss <= bound + 1e-9).all()))
    log("Midm k=2 상한:", round(float(cv['Midm-2.0']**2 / (cv['Midm-2.0']**2 + 1)), 2), "(본문 0.41)")
    ax[0].set_ylabel("Miss rate"); ax[2].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1)); save(fig, "fig14")

# ---------------------------------------------------------------- Table 5 (BH q, 부트스트랩 CI, †)
@step
def table5():
    sc = pd.read_csv("out/ops_scores.csv", index_col=0)["baseline"] if os.path.exists("out/ops_scores.csv") else A2.ops
    g = df.groupby("model").lat; mu = g.mean()
    ind = pd.DataFrame({"mean": mu, "cv": g.std() / mu, "p95p50": g.quantile(.95) / g.median(), "ops": sc}).loc[ORDER]
    miss = {f"{k}x": (df.lat > k * df.model.map(mu)).groupby(df.model).mean() for k in (1.5, 2, 3)}
    miss |= {f"{d}ms": (df.lat > d).groupby(df.model).mean() for d in (500, 1000, 2000, 3000)}; miss = pd.DataFrame(miss).loc[ORDER]
    rng = np.random.default_rng(0); rows = []
    for i in ind:
        for d in miss:
            x, y = ind[i].values, miss[d].values
            rho, p = (np.nan, 1.0) if np.ptp(y) == 0 else stats.spearmanr(x, y); bs = []
            for _ in range(5000):
                ix = rng.integers(0, 9, 9)
                if np.ptp(y[ix]) > 0 and np.ptp(x[ix]) > 0: bs.append(stats.spearmanr(x[ix], y[ix])[0])
            rows.append((i, d, rho, p, *(np.nanpercentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))))
    T = pd.DataFrame(rows, columns=["indicator", "deadline", "rho", "p", "ci_lo", "ci_hi"]); T["q_BH"] = multipletests(T.p, method="fdr_bh")[1]; T["dagger"] = np.where(T.q_BH < .05, "†", "")
    log("== Table 5 전체 (rho가 논문 Table 5와 같아야 함; † = q<.05)\n", T.round(3).to_string()); log("† 셀:", T[T.dagger == "†"][["indicator", "deadline"]].values.tolist())
    log("== Table 5 피벗(rho)\n", T.pivot(index="indicator", columns="deadline", values="rho").round(2).to_string()); T.to_csv("out/table5_full.csv", index=False)

# ---------------------------------------------------------------- Table A6
@step
def tableA6():
    rows = []
    for m in ORDER:
        g = df[df.model == m].sort_values(["task", "pid"]); f = "lat ~ C(task) + inp" if g.gen.nunique() < 3 else "lat ~ C(task) + gen + inp"
        fit = smf.ols(f, data=g).fit(); g = g.assign(res=fit.resid, fit=fit.fittedvalues); ratio = g.lat / g.fit
        ac = max(g[g.task == t].res.reset_index(drop=True).autocorr(1) for t in TASKS)
        rows.append((m, g.res.std() / g.lat.mean() * 100, np.percentile(ratio, 99) / np.percentile(ratio, 50), ac))
    T = pd.DataFrame(rows, columns=["model", "resid_SD_%mean", "resid_P99/P50", "lag1_acf_max"]).round(2)
    log("== Table A6 (OPT는 gen 항 없이 적합)\n", T.to_string(index=False))
    log(f"범위: SD {T['resid_SD_%mean'].min()}–{T['resid_SD_%mean'].max()}%, P99/P50 {T['resid_P99/P50'].min()}–{T['resid_P99/P50'].max()}  (본문 4–15%, 1.12–1.43와 대조)")

# ---------------------------------------------------------------- H4 (§4.5 p값)
@step
def h4():
    d = mi.set_index("model"); log("== H4 처리량(Eq.4 계산, Fig.12와 대조):", d.thr.round(1).to_dict())
    def psp(x, y, z):
        rx, ry, rz = map(stats.rankdata, (x, y, z)); ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        r = np.corrcoef(ex, ey)[0, 1]; k = len(x) - 3; t = r * np.sqrt(k / (1 - r**2)); return r, 2 * stats.t.sf(abs(t), k), k
    r, p = stats.spearmanr(d.mem, d.thr); log(f"Spearman(mem, thr) = {r:.2f}, p = {p:.3f}   (본문 −0.77, p=.015)")
    r2, p2 = stats.spearmanr(d.params, d.thr); log(f"Spearman(params, thr) = {r2:.2f}, p = {p2:.3f}   (본문 −0.56)")
    pr, pp, k = psp(d.thr, d.mem, d.params); log(f"partial rho = {pr:.2f}, df = {k}, p = {pp:.3f}   (본문 −0.78, df=6, p=.022)")
    for drop in ("Llama-3B", "Qwen-0.5B"):
        s = d.drop(drop); log(f"without {drop}: rho = {stats.spearmanr(s.mem, s.thr)[0]:.2f}")
    log("Spearman(mem, params) =", round(stats.spearmanr(d.mem, d.params)[0], 2), "(본문 0.92)")

# ---------------------------------------------------------------- Table 6 blocking ratio + Fig 15(d)
@step
def sim():
    style()
    R = [("P1","EXAONE-2.4B",11.8,0.0),("P1","Midm-2.0",20.1,0.0),("P1","Qwen-0.5B",7.6,0.2),("P1","Qwen-1.5B",26.3,13.4),("P1","gemma-2b",5.1,0.2),
         ("P2","DeepSeek-1.5B",4.2,0.0),("P2","EXAONE-2.4B",0.1,0.0),("P2","Midm-2.0",2.3,0.0),("P2","Qwen-0.5B",0.0,0.0),("P2","Qwen-1.5B",1.6,0.0),("P2","gemma-2b",0.0,0.0),
         ("P3","Llama-3B",12.1,1.1),("P3","Qwen-3B",12.0,1.0),("P3","OPT-2.7B",0.5,0.0),("P3","DeepSeek-1.5B",0.1,0.0),("P3","Midm-2.0",0.1,0.0),
         ("P3","EXAONE-2.4B",0.0,0.0),("P3","Qwen-0.5B",0.0,0.0),("P3","Qwen-1.5B",0.0,0.0),("P3","gemma-2b",0.0,0.0)]
    S = pd.DataFrame(R, columns=["period_set", "model", "np_miss", "pre_miss_max"]); S["p_min"] = S.period_set.map({"P1": 1000, "P2": 2000, "P3": 3000})
    p99 = df[df.task == "Summarization"].groupby("model").lat.quantile(.99); nlu_med = df[df.task == "NLU"].groupby("model").lat.median()
    S["B"] = S.model.map(p99) / S.p_min; S["thr_absent"] = 1 - S.model.map(nlu_med) / S.p_min
    S["pred_miss"] = S.B > 1; S["pred_absent"] = S.B < S.thr_absent
    log("== Table 6 B 열 (B>1이면 비선점 미스 예측; B<thr_absent이면 미스 없음 예측)\n", S.round(2).to_string(index=False))
    log("Spearman(B, NP miss) =", tuple(round(x, 3) for x in stats.spearmanr(S.B, S.np_miss)), "→ 약하면 Fig.15(d)와 N5 두 번째 문장 삭제")
    fig, ax = plt.subplots(figsize=(3.5, 2.8)); mk = {"P1": "o", "P2": "s", "P3": "^"}
    for _, r in S.iterrows():
        ax.scatter(r.B, r.np_miss, marker=mk[r.period_set], color=COL[r.model], s=30, zorder=3)
        ax.scatter(r.B, r.pre_miss_max, marker=mk[r.period_set], facecolors="none", edgecolors=COL[r.model], s=30)
    ax.axvline(1, color="k", ls="--", lw=.8); ax.set_xlabel("Blocking ratio B = P99(summarization) / shortest period"); ax.set_ylabel("Miss rate (%)")
    h = [plt.Line2D([], [], marker=mk[k], ls="", color="gray", label=k, markersize=4) for k in mk]
    h += [plt.Line2D([], [], marker="o", ls="", color="gray", label="non-preemptive", markersize=4), plt.Line2D([], [], marker="o", ls="", mfc="none", mec="gray", label="preemptive (max)", markersize=4)]
    ax.legend(handles=h, fontsize=6, frameon=False, loc="center right"); save(fig, "fig15d")

# ---------------------------------------------------------------- Fig 17 + N_cap
@step
def fig17():
    style(); N = np.arange(1, 51); caps = []; fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True)
    for a, task in zip(ax, ("Summarization", "QA")):
        for m in ORDER:
            g = df[df.model == m]
            if g.gen.nunique() < 3: continue
            fit = smf.ols("lat ~ C(task) + gen + inp", data=g).fit(); d99 = np.percentile(fit.resid, 99); Lmax = g[g.task == task].inp.max()
            pred = fit.predict(pd.DataFrame({"task": [task] * 50, "gen": N, "inp": [Lmax] * 50})).values + d99
            a.plot(N, pred, color=COL[m], lw=1.2, label=m)
            for D in (1000, 2000, 3000): caps.append((task, m, D, int(N[pred <= D].max()) if (pred <= D).any() else 0))
        for D in (1000, 2000, 3000): a.axhline(D, color="k", ls=":", lw=.8)
        a.set_title(task); a.set_xlabel("Cap on generated tokens, N")
    ax[0].set_ylabel("Predicted 99th-percentile latency bound (ms)"); ax[1].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1))
    C = pd.DataFrame(caps, columns=["task", "model", "deadline_ms", "N_cap"]).pivot_table(index=["task", "model"], columns="deadline_ms", values="N_cap")
    log("== Fig.17 N_cap (0 = 캡 1토큰으로도 불충족, 50 = 캡 없이 충족)\n", C.to_string()); log("§5.5 예시 [value]: Llama-3B, Summarization, 3 s →", C.loc[("Summarization", "Llama-3B"), 3000])
    save(fig, "fig17")

if __name__ == "__main__":
    todo = sys.argv[1:] or ["verify", "ops", "fig04de", "fig05", "fig07", "fig08c", "fig09", "fig10", "fig11", "fig14", "h4", "table5", "tableA6", "sim", "fig17"]
    for s in todo:
        log(f"\n######## {s}"); STEPS[s]()