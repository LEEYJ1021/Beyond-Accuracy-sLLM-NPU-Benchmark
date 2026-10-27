#!/usr/bin/env python3
"""
Within-model observational decomposition of sLLM execution time (NPU trial log).

Input : trial-level CSV (one row per model x task x prompt).
Output: CSV tables, PNG figures and summary.txt in --out.

Nothing about models, tasks, token ranges, break points or thresholds is hard-coded:
columns are auto-detected (override with --*-col), and every cut-off is derived from the data
or exposed as a CLI option.

Analyses
  0 data overview (token-range quantiles per model x task)
  1 within-model input-length curve at (near-)fixed output length: linear / piecewise / spline / staircase
  2 label-only (very short output) trials as a near-direct prefill proxy
  3 variance decomposition (Shapley share of R^2): generated tokens, input tokens, task
  4 residual jitter upper bound (prompt effect shared across models vs. unshared remainder)
  5 temporal structure of residuals (rolling-median outliers, runs, Ljung-Box)
  6 decode cost vs. input length (g x in interaction)
  7 per-token cost next to compile settings (optional --config-csv)
  8 own-tokenizer input counts (optional --prompts-csv + --hf-map)
"""
import argparse
import json
import math
import re
import sys
import warnings
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

PATTERNS = {
    "model": [r"^model(_name|_id)?$", r"model"],
    "task": [r"^task(_type|_name)?$", r"task", r"dataset"],
    "latency": [r"^latency", r"latency", r"exec.*time", r"elapsed", r"wall"],
    "gen": [r"gen.*tok", r"(output|new|completion).*tok", r"tokens?[\s_]*(out|generated)"],
    "inp": [r"(^|_)(input|prompt|in)_?tok", r"input.*len"],
    "sample": [r"sample.*(id|idx|index)", r"(prompt|item).*(id|idx|index)", r"^idx$", r"sample"],
    "order": [r"order", r"trial", r"run", r"seq", r"time_?stamp"],
}


# ----------------------------------------------------------------------------- helpers
def detect(df, key, override=None):
    if override:
        if override not in df.columns:
            sys.exit(f"--{key}-col '{override}' not in columns: {list(df.columns)}")
        return override
    for pat in PATTERNS[key]:
        hits = [c for c in df.columns if re.search(pat, c, re.I)]
        if hits:
            return hits[0]
    return None


def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    m = ~np.isnan(p)
    if m.sum():
        out[m] = multipletests(p[m], method="fdr_bh")[1]
    return out


def cv_rmse(formula, data, folds, rng):
    idx = np.arange(len(data))
    rng.shuffle(idx)
    parts = np.array_split(idx, folds)
    errs = []
    for k in range(folds):
        te = data.iloc[parts[k]]
        tr = data.iloc[np.concatenate([parts[j] for j in range(folds) if j != k])]
        try:
            m = smf.ols(formula, tr).fit()
            errs.append((te["lat"] - m.predict(te)).values)
        except Exception:
            return np.nan
    e = np.concatenate(errs)
    return float(np.sqrt(np.nanmean(e ** 2)))


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def short(name):
    return re.sub(r"(?i)-?(instruct|it|distill)", "", str(name)).strip("-_ ")[:22]


# ----------------------------------------------------------------------------- data
def load(args):
    raw = pd.read_csv(args.csv, encoding="utf-8-sig")
    cols = {k: detect(raw, k, getattr(args, f"{k}_col", None)) for k in PATTERNS}
    print("Detected columns:", {k: v for k, v in cols.items()})
    for k in ("model", "task", "latency", "gen"):
        if cols[k] is None:
            sys.exit(f"Could not detect '{k}' column. Columns: {list(raw.columns)}. Use --{k}-col.")
    d = pd.DataFrame({"model": raw[cols["model"]].astype(str),
                      "task": raw[cols["task"]].astype(str),
                      "lat": pd.to_numeric(raw[cols["latency"]], errors="coerce"),
                      "g": pd.to_numeric(raw[cols["gen"]], errors="coerce")})
    lname = cols["latency"].lower()
    if args.latency_unit == "s" or (args.latency_unit == "auto" and re.search(r"(_s|sec|seconds)$", lname)):
        d["lat"] = d["lat"] * 1000.0
        print("Latency interpreted as seconds -> converted to ms.")
    if cols["inp"] is not None:
        d["x"] = pd.to_numeric(raw[cols["inp"]], errors="coerce")
    else:
        d["x"] = np.nan
    if cols["sample"] is not None:
        d["s"] = raw[cols["sample"]].astype(str)
    else:
        d["s"] = d.groupby(["model", "task"]).cumcount().astype(str)
        print("WARNING: no sample-index column; using row order within model x task as prompt key "
              "(valid only if all models ran the same prompts in the same order).")
    if cols["order"] is not None and pd.api.types.is_numeric_dtype(raw[cols["order"]]):
        d["o"] = pd.to_numeric(raw[cols["order"]], errors="coerce")
    else:
        d["o"] = d.groupby(["model", "task"]).cumcount().astype(float)
        print("Trial order = row order within model x task.")
    return d, raw, cols


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def attach_features(d, args):
    """Merge precomputed input-token counts (e.g. input_features.csv) on (model,) task, sample."""
    if not args.features_csv:
        return d
    f = pd.read_csv(args.features_csv, encoding="utf-8-sig")
    ft, fs, fm = detect(f, "task"), detect(f, "sample"), detect(f, "model")
    if ft is None or fs is None:
        sys.exit(f"--features-csv needs task and sample columns; found {list(f.columns)}")
    if args.features_input_col:
        xc = args.features_input_col
    else:
        cand = [c for c in f.columns if re.search(r"tok", c, re.I) and pd.api.types.is_numeric_dtype(f[c])]
        if len(cand) != 1:
            sys.exit(f"Specify --features-input-col; numeric token-like columns found: {cand} (all: {list(f.columns)})")
        xc = cand[0]
    g = pd.DataFrame({"task": f[ft].astype(str), "s": f[fs].astype(str),
                      "x_feat": pd.to_numeric(f[xc], errors="coerce")})
    keys = ["task", "s"]
    if fm is not None and f[fm].nunique() > 1:
        fmodels = f[fm].astype(str).unique()
        mp = {}
        for dm in d["model"].unique():
            c = [x for x in fmodels if _norm(dm) in _norm(x) or _norm(x) in _norm(dm)]
            c = [x for x in c if _norm(x) == _norm(dm)] or c
            if len(c) == 1:
                mp[dm] = c[0]
            else:
                print(f"WARNING: cannot map model '{dm}' to features models {list(fmodels)} (candidates: {c})")
        d["fm"] = d["model"].map(mp)
        g["fm"] = f[fm].astype(str)
        keys = ["fm"] + keys
        print("Model mapping (csv -> features):", mp)
    g = g.drop_duplicates(keys)
    d = d.merge(g, on=keys, how="left")
    d["x"] = d["x_feat"]
    cov = d["x"].notna().mean()
    print(f"Input-token merge coverage: {cov:.1%} using column '{xc}' (keys: {keys})")
    if cov < 0.9:
        print("WARNING: low coverage - check task/sample/model naming between the two CSVs.")
    return d


def attach_tokens(d, args):
    """Optional: count input tokens with one tokenizer (--tokenizer) or per-model (--hf-map)."""
    if args.prompts_csv is None:
        return d
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("pip install transformers to use --prompts-csv")
    p = pd.read_csv(args.prompts_csv)
    tcol = detect(p, "task"); scol = detect(p, "sample")
    xcol = next((c for c in p.columns if re.search(r"text|prompt|input|article|sentence|question", c, re.I)
                 and c not in (tcol, scol)), None)
    if None in (tcol, scol, xcol):
        sys.exit(f"--prompts-csv needs task, sample and text columns; found {list(p.columns)}")
    p = p.rename(columns={tcol: "task", scol: "s", xcol: "text"})[["task", "s", "text"]]
    p["task"] = p["task"].astype(str); p["s"] = p["s"].astype(str)
    d = d.merge(p, on=["task", "s"], how="left")
    if args.hf_map:
        hf = json.load(open(args.hf_map))
        d["x_own"] = np.nan
        for m, repo in hf.items():
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            sel = d["model"] == m
            d.loc[sel, "x_own"] = [len(tok(str(t))["input_ids"]) if isinstance(t, str) else np.nan
                                   for t in d.loc[sel, "text"]]
    if args.tokenizer:
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        cnt = {t: len(tok(str(t))["input_ids"]) for t in p["text"].unique()}
        d["x_single"] = d["text"].map(cnt)
        if d["x"].isna().all():
            d["x"] = d["x_single"]
    return d


# ----------------------------------------------------------------------------- analyses
def overview(d, gfull, out):
    rows = []
    for (m, t), g in d.groupby(["model", "task"]):
        q = g["x"].quantile([0, .25, .5, .75, 1]).values if g["x"].notna().any() else [np.nan] * 5
        rows.append(dict(model=m, task=t, n=len(g), gen_mean=g["g"].mean(), n_full=int((g["g"] >= gfull).sum()),
                         x_min=q[0], x_q25=q[1], x_med=q[2], x_q75=q[3], x_max=q[4], lat_med=g["lat"].median()))
    t = pd.DataFrame(rows)
    t.to_csv(out / "T0_data_overview.csv", index=False)
    print("\n[0] Data overview (use this to judge which analyses are informative)")
    print(t.round(1).to_string(index=False))
    return t


def curve_table(sub, dfs, folds, rng, nbreak):
    sub = sub.copy()
    te = " + C(task)" if sub["task"].nunique() > 1 else ""
    forms = {"linear": "lat ~ x" + te}
    for k in dfs:
        if sub["x"].nunique() > k + 2:
            forms[f"spline_df{k}"] = f"lat ~ cr(x, df={k})" + te
    best = None
    for q in np.linspace(0.15, 0.85, nbreak):
        bp = sub["x"].quantile(q)
        sub["h"] = np.maximum(sub["x"] - bp, 0)
        if (sub["h"] > 0).sum() < 5 or (sub["h"] == 0).sum() < 5:
            continue
        a = smf.ols("lat ~ x + h" + te, sub).fit().aic
        if best is None or a < best[0]:
            best = (a, bp)
    if best:
        sub["h"] = np.maximum(sub["x"] - best[1], 0)
        forms["piecewise"] = "lat ~ x + h" + te
    nb = min(8, sub["x"].nunique() // 5)
    if nb >= 3:
        sub["bin"] = pd.qcut(sub["x"], nb, duplicates="drop").cat.codes
        forms["staircase"] = "lat ~ C(bin)" + te
    rows, fits = [], {}
    for name, f in forms.items():
        m = smf.ols(f, sub).fit()
        fits[name] = m
        rows.append(dict(curve=name, aic=m.aic, r2=m.rsquared, rmse_cv=cv_rmse(f, sub, folds, rng),
                         n_par=int(m.df_model) + 1, breakpoint=best[1] if (name == "piecewise" and best) else np.nan))
    return pd.DataFrame(rows), fits, sub


def a1_curves(d, gfull, args, out, rng, log):
    print("\n[1] Within-model input-length curves (output length fixed at the cap)")
    full = d[(d["g"] >= gfull) & d["x"].notna()]
    rows, plots = [], {}
    for m, gm in full.groupby("model"):
        subsets = [("pooled", gm)] + [(t, gt) for t, gt in gm.groupby("task")]
        for name, sub in subsets:
            if len(sub) < args.min_n or sub["x"].nunique() < args.min_unique or sub["x"].std() == 0:
                continue
            tab, fits, sub2 = curve_table(sub, args.spline_df, args.folds, rng, args.n_break)
            lin = tab.loc[tab.curve == "linear", "rmse_cv"].iloc[0]
            bestc = tab.sort_values("rmse_cv").iloc[0]
            gain = (lin - bestc.rmse_cv) / lin if lin > 0 else np.nan
            rows.append(dict(model=m, subset=name, n=len(sub), x_min=sub.x.min(), x_max=sub.x.max(),
                             lin_r2=tab.loc[tab.curve == "linear", "r2"].iloc[0],
                             lin_slope_ms_per_tok=smf.ols("lat ~ x" + (" + C(task)" if name == "pooled" and sub.task.nunique() > 1 else ""), sub).fit().params["x"],
                             best_curve=bestc.curve, best_rmse_cv=bestc.rmse_cv, lin_rmse_cv=lin,
                             rmse_gain_vs_linear=gain,
                             d_aic_best_vs_linear=tab.loc[tab.curve == "linear", "aic"].iloc[0] - tab.aic.min(),
                             breakpoint=tab.loc[tab.curve == "piecewise", "breakpoint"].iloc[0] if (tab.curve == "piecewise").any() else np.nan))
            if name != "pooled":
                bm = fits[bestc.curve]
                o = np.argsort(sub2["x"].values)
                plots[(m, name)] = (sub2["x"].values[o], bm.fittedvalues.values[o], bestc.curve)
    t = pd.DataFrame(rows)
    t.to_csv(out / "T1_input_length_curves.csv", index=False)
    if t.empty:
        print("  no subset had enough full-length trials with varying input length; see T0.")
        log.append("A1: no eligible subset (insufficient full-length trials / input variation).")
        return t
    print(t.round(3).to_string(index=False))
    models = sorted(full["model"].unique())
    nc = min(3, len(models)); nr = math.ceil(len(models) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 3.6 * nr), squeeze=False)
    tasks = sorted(full["task"].unique()); cmap = dict(zip(tasks, plt.cm.tab10.colors))
    for ax, m in zip(axs.ravel(), models):
        g = full[full["model"] == m]
        for t_, gt in g.groupby("task"):
            ax.scatter(gt["x"], gt["lat"], s=6, alpha=.4, color=cmap[t_], label=t_)
            if (m, t_) in plots:
                xs, ys, cn = plots[(m, t_)]
                ax.plot(xs, ys, color=cmap[t_], lw=1.6)
        ax.set_title(short(m), fontsize=10); ax.set_xlabel("input tokens"); ax.set_ylabel("latency (ms)")
        ax.grid(alpha=.3)
    for ax in axs.ravel()[len(models):]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=7)
    savefig(fig, out / "F1_input_length_curves.png")
    sel = t[t.subset != "pooled"]
    nl = (sel.rmse_gain_vs_linear > args.gain_thr).sum()
    log.append(f"A1: {len(sel)} model-task subsets with full-length outputs; best curve is non-linear with >"
               f"{args.gain_thr:.0%} CV-RMSE gain in {nl}; median linear R2={sel.lin_r2.median():.2f}; "
               f"median slope={sel.lin_slope_ms_per_tok.median():.3f} ms/input token.")
    return t


def a2_label_only(d, args, out, log):
    print(f"\n[2] Label-only trials (generated tokens <= {args.short_gen}) as a prefill proxy")
    sub = d[(d["g"] <= args.short_gen) & d["x"].notna()]
    rows = []
    for (m, t), g in sub.groupby(["model", "task"]):
        if len(g) < args.min_n or g["x"].nunique() < args.min_unique:
            continue
        r = smf.ols("lat ~ x", g).fit()
        rows.append(dict(model=m, task=t, n=len(g), x_min=g.x.min(), x_max=g.x.max(), lat_median_ms=g.lat.median(),
                         slope_ms_per_input_tok=r.params["x"], intercept_ms=r.params["Intercept"], r2=r.rsquared,
                         p=r.pvalues["x"]))
    t = pd.DataFrame(rows)
    if t.empty:
        print("  no model-task cell has enough label-only trials.")
        log.append("A2: no cell with enough label-only trials.")
        t.to_csv(out / "T2_label_only_prefill.csv", index=False)
        return t
    t["p_bh"] = bh(t["p"])
    t.to_csv(out / "T2_label_only_prefill.csv", index=False)
    print(t.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    for (m, tk), g in sub.groupby(["model", "task"]):
        if ((t.model == m) & (t.task == tk)).any():
            ax.scatter(g["x"], g["lat"], s=8, alpha=.5, label=f"{short(m)}/{tk}")
    ax.set_xlabel("input tokens"); ax.set_ylabel("latency (ms)"); ax.legend(fontsize=6); ax.grid(alpha=.3)
    savefig(fig, out / "F2_label_only_prefill.png")
    log.append(f"A2: {len(t)} cells; latency {t.lat_median_ms.min():.0f}-{t.lat_median_ms.max():.0f} ms (median per cell); "
               f"slope range {t.slope_ms_per_input_tok.min():.3f}-{t.slope_ms_per_input_tok.max():.3f} ms/input token.")
    return t


def shapley(sub, factors):
    names = list(factors)
    cache = {}

    def v(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = smf.ols("lat ~ " + " + ".join(factors[s] for s in S), sub).fit().rsquared if S else 0.0
        return cache[key]
    phi = {}
    for i in names:
        tot = 0.0
        for perm in permutations(names):
            pre = list(perm[:perm.index(i)])
            tot += v(pre + [i]) - v(pre)
        phi[i] = tot / math.factorial(len(names))
    return phi, v(names)


def a3_variance(d, out, log):
    print("\n[3] Variance decomposition (Shapley share of R^2)")
    rows = []
    for m, g in d.dropna(subset=["x"]).groupby("model"):
        fac = {}
        if g["g"].nunique() > 1: fac["generated_tokens"] = "g"
        if g["x"].nunique() > 1: fac["input_tokens"] = "x"
        if g["task"].nunique() > 1: fac["task"] = "C(task)"
        if not fac:
            continue
        phi, r2 = shapley(g, fac)
        row = dict(model=m, n=len(g), r2_total=r2, residual=1 - r2)
        row.update({k: phi.get(k, 0.0) for k in ("generated_tokens", "input_tokens", "task")})
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out / "T3_variance_decomposition.csv", index=False)
    print(t.round(3).to_string(index=False))
    if t.empty:
        return t
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bot = np.zeros(len(t)); cols = {"generated_tokens": "#4c78a8", "input_tokens": "#f58518", "task": "#54a24b", "residual": "#bab0ac"}
    for k in ("generated_tokens", "input_tokens", "task", "residual"):
        ax.bar([short(m) for m in t.model], t[k].values, bottom=bot, label=k.replace("_", " "), color=cols[k]); bot += t[k].values
    ax.set_ylabel("share of latency variance"); ax.legend(fontsize=7); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    savefig(fig, out / "F3_variance_decomposition.png")
    log.append("A3: median shares across models - " + ", ".join(f"{k}={t[k].median():.2f}" for k in ("generated_tokens", "input_tokens", "task", "residual")) + ".")
    return t


def residuals(d):
    rows = []
    for (m, tk), g in d.groupby(["model", "task"]):
        g = g.sort_values("o")
        terms = [v for v, col in (("g", "g"), ("x", "x")) if g[col].notna().all() and g[col].nunique() > 1]
        if len(g) < 8:
            continue
        if terms:
            fit = smf.ols("lat ~ " + " + ".join(terms), g).fit().fittedvalues
        else:
            fit = pd.Series(g["lat"].mean(), index=g.index)
        ok = fit > 0
        e = ((g["lat"] - fit) / fit).where(ok)
        rows.append(pd.DataFrame(dict(model=m, task=tk, s=g["s"], o=g["o"], e=e)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def a4_jitter(res, out, log):
    print("\n[4] Residual jitter upper bound (prompt effect shared across models removed)")
    if res.empty:
        return pd.DataFrame()
    res = res.copy(); res["key"] = res["task"] + "|" + res["s"]
    piv = res.pivot_table(index="key", columns="model", values="e")
    rows = []
    for m in piv.columns:
        others = piv.drop(columns=m)
        u = others.mean(axis=1, skipna=True).where(others.notna().sum(axis=1) >= 2)
        z = pd.DataFrame({"e": piv[m], "u": u}).dropna()
        if len(z) < 20 or z["u"].std() == 0:
            continue
        b, a = np.polyfit(z["u"], z["e"], 1)
        noise = z["e"] - (a + b * z["u"])
        rows.append(dict(model=m, n=len(z), resid_sd=z["e"].std(), noise_sd=noise.std(),
                         shared_share=1 - noise.var() / z["e"].var(),
                         noise_p95_abs=np.quantile(np.abs(noise), .95), noise_p99_abs=np.quantile(np.abs(noise), .99),
                         p99_over_p50=np.quantile(1 + noise, .99) / np.median(1 + noise)))
    t = pd.DataFrame(rows)
    t.to_csv(out / "T4_jitter_upper_bound.csv", index=False)
    print(t.round(4).to_string(index=False))
    if not t.empty:
        log.append(f"A4: relative noise SD (upper bound on unshared jitter) {t.noise_sd.min():.3f}-{t.noise_sd.max():.3f}; "
                   f"P99/P50 of (1+noise) {t.p99_over_p50.min():.2f}-{t.p99_over_p50.max():.2f}; "
                   f"median share of residual variance explained by the cross-model prompt effect = {t.shared_share.median():.2f}.")
    return t


def a5_temporal(res, args, out, log):
    print("\n[5] Temporal structure of residuals")
    rows = []
    for (m, tk), g in res.dropna(subset=["e"]).groupby(["model", "task"]):
        g = g.sort_values("o"); e = g["e"].reset_index(drop=True); n = len(e)
        if n < 20:
            continue
        w = max(5, n // 10) | 1
        roll = e.rolling(w, center=True, min_periods=1).median()
        dev = e - roll
        scale = max(1.4826 * np.median(np.abs(dev - np.median(dev))), args.min_scale)
        flag = (np.abs(dev) / scale) > args.z_thr
        runs = (flag.astype(int).groupby((flag != flag.shift()).cumsum()).sum()).max() if flag.any() else 0
        L = max(1, min(10, n // 5))
        lb = acorr_ljungbox(e, lags=[L], return_df=True)["lb_pvalue"].iloc[0]
        rows.append(dict(model=m, task=tk, n=n, window=w, flagged_frac=flag.mean(), max_run=int(runs),
                         lag1_autocorr=float(e.autocorr(1)), ljung_box_lags=L, lb_p=lb))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["lb_p_bh"] = bh(t["lb_p"])
    t.to_csv(out / "T5_temporal_structure.csv", index=False)
    print(t.round(4).to_string(index=False))
    models = sorted(res["model"].unique())
    nc = min(3, len(models)); nr = math.ceil(len(models) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(5 * nc, 3 * nr), squeeze=False)
    tasks = sorted(res["task"].unique()); cmap = dict(zip(tasks, plt.cm.tab10.colors))
    for ax, m in zip(axs.ravel(), models):
        for tk, g in res[res.model == m].groupby("task"):
            g = g.sort_values("o"); ax.plot(np.arange(len(g)), g["e"], lw=.7, color=cmap[tk], label=tk)
        ax.set_title(short(m), fontsize=10); ax.set_xlabel("trial order within task"); ax.set_ylabel("relative residual"); ax.grid(alpha=.3)
    for ax in axs.ravel()[len(models):]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=7)
    savefig(fig, out / "F5_residual_time.png")
    sig = int((t.lb_p_bh < 0.05).sum())
    log.append(f"A5: {len(t)} series; flagged-outlier fraction median={t.flagged_frac.median():.3f} (max {t.flagged_frac.max():.3f}); "
               f"longest run={t.max_run.max()}; Ljung-Box significant after BH in {sig} series.")
    return t


def a6_decode_vs_input(d, args, out, log):
    print("\n[6] Decode cost vs input length (g x in interaction, exploratory)")
    rows = []
    for m, g in d.dropna(subset=["x"]).groupby("model"):
        if g["g"].nunique() < 3 or g["x"].nunique() < 3:
            continue
        for name, sub in [("pooled", g)] + [(t, gt) for t, gt in g.groupby("task")]:
            if len(sub) < args.min_n or sub["g"].nunique() < 3 or sub["x"].nunique() < 3:
                continue
            s = sub.copy(); s["gc"] = s["g"] - s["g"].mean(); s["xc"] = s["x"] - s["x"].mean()
            te = " + C(task)" if name == "pooled" and s["task"].nunique() > 1 else ""
            r = smf.ols("lat ~ gc * xc" + te, s).fit(cov_type="HC3")
            ci = r.conf_int().loc["gc:xc"]
            rows.append(dict(model=m, subset=name, n=len(s), corr_g_x=s[["g", "x"]].corr().iloc[0, 1],
                             d_ms_per_gen_per_input=r.params["gc:xc"], ci_lo=ci[0], ci_hi=ci[1], p=r.pvalues["gc:xc"]))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["p_bh"] = bh(t["p"])
    t.to_csv(out / "T6_decode_vs_input.csv", index=False)
    print(t.round(5).to_string(index=False))
    log.append(f"A6: {len(t)} fits; interaction significant after BH in {int((t.p_bh < .05).sum())}; "
               f"median |corr(g,x)| = {t.corr_g_x.abs().median():.2f} (high values mean weak identification).")
    return t


def a7_config(d, args, out, log):
    slopes = []
    for m, g in d.groupby("model"):
        if g["g"].nunique() > 2:
            r = smf.ols("lat ~ g", g).fit()
            slopes.append(dict(model=m, ms_per_generated_token=r.params["g"], r2=r.rsquared))
    t = pd.DataFrame(slopes)
    if args.config_csv:
        c = pd.read_csv(args.config_csv)
        mc = detect(c, "model")
        norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
        c["_k"] = c[mc].map(norm); t["_k"] = t["model"].map(norm)
        t = t.merge(c.drop(columns=[mc]), on="_k", how="left").drop(columns="_k")
    t.to_csv(out / "T7_per_token_cost_and_config.csv", index=False)
    print("\n[7] Per-token cost" + (" with compile settings" if args.config_csv else " (add --config-csv to attach settings)"))
    print(t.round(3).to_string(index=False))
    log.append(f"A7: per-token cost {t.ms_per_generated_token.min():.1f}-{t.ms_per_generated_token.max():.1f} ms/token across {len(t)} models.")
    return t


def a8_tokenizer_sensitivity(d, out, log):
    cols = [c for c in ("x_own", "x_single") if c in d.columns and d[c].notna().any()]
    if "x_own" not in cols or d["x"].isna().all():
        return
    rows = []
    for m, g in d.dropna(subset=["x", "x_own"]).groupby("model"):
        rows.append(dict(model=m, n=len(g), spearman=stats.spearmanr(g["x"], g["x_own"])[0],
                         ratio_median=(g["x_own"] / g["x"]).median()))
    t = pd.DataFrame(rows); t.to_csv(out / "T8_tokenizer_sensitivity.csv", index=False)
    print("\n[8] Own-tokenizer vs reference input counts"); print(t.round(3).to_string(index=False))
    log.append(f"A8: Spearman between own and reference input counts {t.spearman.min():.3f}-{t.spearman.max():.3f}.")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="trial-level CSV")
    ap.add_argument("--out", default="within_model_out")
    for k in PATTERNS:
        ap.add_argument(f"--{k}-col", default=None)
    ap.add_argument("--latency-unit", choices=["auto", "ms", "s"], default="auto")
    ap.add_argument("--full-gen", type=float, default=None, help="generated tokens counted as 'full length' (default: data max)")
    ap.add_argument("--short-gen", type=float, default=3, help="max generated tokens for label-only trials")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--min-unique", type=int, default=10)
    ap.add_argument("--spline-df", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--n-break", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gain-thr", type=float, default=0.05)
    ap.add_argument("--z-thr", type=float, default=3.0)
    ap.add_argument("--min-scale", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config-csv", default=None)
    ap.add_argument("--features-csv", default=None, help="precomputed input-token CSV (e.g. input_features.csv)")
    ap.add_argument("--features-input-col", default=None, help="input-token column in --features-csv")
    ap.add_argument("--prompts-csv", default=None, help="task, sample, text columns (to count input tokens)")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer for a single reference count")
    ap.add_argument("--hf-map", default=None, help="JSON {model_name_in_csv: hf_repo} for own-tokenizer counts")
    ap.add_argument("--use-own-tokens", action="store_true", help="use x_own as the input-length variable")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    d, raw, cols = load(args)
    d = attach_features(d, args)
    d = attach_tokens(d, args)
    if args.use_own_tokens and "x_own" in d.columns:
        d["x"] = d["x_own"]
    d = d.dropna(subset=["lat", "g"]); d = d[d["lat"] > 0].reset_index(drop=True)
    if d["x"].isna().all():
        sys.exit("No input-token column found. Provide --input-col or --prompts-csv with --tokenizer (and/or --hf-map).")
    gfull = args.full_gen if args.full_gen is not None else d["g"].max()
    print(f"\nModels: {sorted(d.model.unique())}\nTasks : {sorted(d.task.unique())}\nFull-length = generated tokens >= {gfull}")
    log = []
    overview(d, gfull, out)
    a1_curves(d, gfull, args, out, rng, log)
    a2_label_only(d, args, out, log)
    a3_variance(d, out, log)
    res = residuals(d)
    if not res.empty:
        res.to_csv(out / "residuals.csv", index=False)
    a4_jitter(res, out, log)
    a5_temporal(res, args, out, log)
    a6_decode_vs_input(d, args, out, log)
    a7_config(d, args, out, log)
    a8_tokenizer_sensitivity(d, out, log)
    txt = "\n".join(log)
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print("\n===== SUMMARY (paste-ready numbers) =====\n" + txt + f"\n\nFiles written to: {out.resolve()}")


if __name__ == "__main__":
    main()