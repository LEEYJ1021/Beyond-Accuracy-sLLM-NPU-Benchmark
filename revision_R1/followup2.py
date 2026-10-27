#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
followup2.py -- (1) 요구사항 기반 고정 주기 시뮬레이션 (비선점 / 제한적 선점 / 출력 길이 제한 what-if)
                (2) 과제별 지연 상한 모델(prefill + 50*ms/token + 여유율) 홀드아웃 검증
입력: Data_시계열.csv 만 사용
사용: python followup2.py [--quick] [--base "<folder>"] [--reps 20] [--H 600000] [--maxutil 0.8]
                         [--nlu_cap 8] [--periods "1000,4000,8000;2000,8000,16000;3000,12000,24000"]
"""
import argparse, heapq, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
DEFAULT_BASE = "/home/yjlee/Research/Microprocessors and Microsystems_R1"
TASKS = ["NLU", "QA", "Summarization"]
SHORT = {"DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeek-1.5B", "EXAONE-3.5-2.4B-Instruct": "EXAONE-2.4B",
         "Llama-3.2-3B-Instruct": "Llama-3B", "Midm-2.0-Mini-Instruct": "Midm-2.0",
         "Qwen2.5-0.5B-Instruct": "Qwen-0.5B", "Qwen2.5-1.5B-Instruct": "Qwen-1.5B",
         "Qwen2.5-3B-Instruct": "Qwen-3B", "gemma-2b-it": "gemma-2b", "opt-2.7b": "OPT-2.7B"}
POLICIES = [("NP", np.inf, 0.0), ("Q500", 500.0, 0.0), ("Q100", 100.0, 0.0),
            ("Q25", 25.0, 0.0), ("Q25+ovh2ms", 25.0, 2.0)]   # (이름, 선점 가능 간격 Q[ms], 재개 오버헤드[ms])
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)


def short(m):
    return SHORT.get(m, m)


def save(df, name):
    df.to_csv(f"{OUT}/tables/{name}.csv", encoding="utf-8-sig")
    return df


# ------------------------------------------------------------------ 데이터
def load(base):
    d = pd.read_csv(f"{base}/Data_시계열.csv")
    d.columns = ["Model", "Task", "Sample", "lat", "tok", "tps"]
    for c in ["lat", "tok", "tps"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["lat", "tok"]).reset_index(drop=True)
    return d


def fit_model(df):
    """모델 하나의 lat ~ 0 + C(Task) + tok. 반환: (과제별 절편 dict, 토큰 기울기 b, 기울기 식별 여부)"""
    X = np.column_stack([(df.Task == t).astype(float).values for t in TASKS] + [df.tok.values.astype(float)])
    beta = np.linalg.lstsq(X, df.lat.values, rcond=None)[0]
    ok = df.tok.std() >= 1.0          # 토큰이 거의 상수(예: OPT)면 기울기 식별 불가
    return {t: beta[i] for i, t in enumerate(TASKS)}, float(beta[3]), bool(ok)


# ------------------------------------------------------------------ (1) 시뮬레이션
def simulate(arrs, periods, H, Q, ovh, rng):
    """단일 NPU, EDF(deadline=period), 위상 무작위. Q=inf -> 비선점, Q 유한 -> Q[ms]마다 선점 가능(제한적 선점)."""
    jobs = []
    for k, (a, p) in enumerate(zip(arrs, periods)):
        off = rng.uniform(0, p)
        m = int((H - off) // p)
        ex = rng.choice(a, m)
        jobs += [(off + i * p, off + (i + 1) * p, k, float(e)) for i, e in enumerate(ex)]
    jobs.sort()
    t, i, ready, seq, last = 0.0, 0, [], 0, -1
    miss, tot = np.zeros(len(arrs)), np.zeros(len(arrs))
    while i < len(jobs) or ready:
        if not ready and t < jobs[i][0]:
            t = jobs[i][0]
        while i < len(jobs) and jobs[i][0] <= t:
            r, dl, k, e = jobs[i]
            heapq.heappush(ready, [dl, seq, k, e, False]); seq += 1; i += 1
        job = heapq.heappop(ready)
        dl, sid, k, rem, started = job
        if started and last != sid:
            rem += ovh                      # 문맥 전환 후 재개 오버헤드
        run = min(rem, Q)
        t += run; rem -= run; last = sid
        if rem <= 1e-9:
            tot[k] += 1; miss[k] += t > dl + 1e-9
        else:
            job[3], job[4] = rem, True
            heapq.heappush(ready, job)
    with np.errstate(invalid="ignore", divide="ignore"):
        return miss / tot, miss.sum() / tot.sum()


def build_arrays(d, models, nlu_cap, fits):
    base, capd = {}, {}
    for m in models:
        base[m] = [d[(d.Model == m) & (d.Task == t)].lat.values for t in TASKS]
        a, b, ok = fits[m]
        arr = []
        for t in TASKS:
            g = d[(d.Model == m) & (d.Task == t)]
            if t == "NLU" and ok:
                arr.append(np.maximum(g.lat.values - b * np.maximum(g.tok.values - nlu_cap, 0), 0.2 * g.lat.values))
            else:
                arr.append(g.lat.values)
        capd[m] = arr if ok else None      # 기울기 식별 불가 모델(OPT)은 what-if 제외
    return base, capd


def run_sim(d, models, fits, args):
    rng = np.random.default_rng(42)
    base, capd = build_arrays(d, models, args.nlu_cap, fits)
    rows, blk = [], []
    for pname, P in args.period_sets.items():
        for m in models:
            util = sum(a.mean() / p for a, p in zip(base[m], P))
            # 블로킹/자체초과 진단 (시뮬레이션과 무관한 결정적 지표)
            for k, t in enumerate(TASKS):
                longer = [j for j in range(3) if P[j] > P[k]]
                blocker = max([base[m][j].max() for j in longer], default=np.nan)
                blk.append({"Model": m, "periods": pname, "task": t, "D_ms": P[k], "util_mean": round(util, 2),
                            "P(exec>D)_self": float((base[m][k] > P[k]).mean()),
                            "blocker_max_ms": blocker,
                            "blocking_violation(NP)": bool(blocker > P[k]) if not np.isnan(blocker) else False})
            if util > args.maxutil:
                continue
            for scen, arrs in [("baseline", base[m]), (f"NLU_cap{args.nlu_cap}tok(what-if)", capd[m])]:
                if arrs is None:
                    continue
                for pol, Q, ovh in POLICIES:
                    res = [simulate(arrs, P, args.H, Q, ovh, rng) for _ in range(args.reps)]
                    pt = np.array([r[0] for r in res]); ov = np.array([r[1] for r in res])
                    rows.append({"Model": m, "periods": pname, "util_mean": round(util, 2), "scenario": scen, "policy": pol,
                                 **{f"miss_{t}": np.nanmean(pt[:, k]) for k, t in enumerate(TASKS)},
                                 "miss_all": ov.mean(), "all_lo": np.percentile(ov, 2.5), "all_hi": np.percentile(ov, 97.5)})
        print(f"  periods {pname} {P} done", flush=True)
    sim = pd.DataFrame(rows)
    if not sim.empty:
        save(sim.round(4).set_index(["Model", "periods", "scenario", "policy"]), "M1_sim_fixed_period")
    blk = pd.DataFrame(blk).round(3)
    save(blk.set_index(["Model", "periods", "task"]), "M2_blocking_and_self_exceed")
    return sim, blk


def print_sim(sim, blk):
    if sim.empty:
        print("유효(util<=maxutil) 조건이 없습니다. --periods 를 늘리거나 --maxutil 을 올리세요."); return
    sim = sim.copy()
    sim["cell"] = sim.apply(lambda r: f"{100*r.miss_all:5.1f} [{100*r.all_lo:4.1f}-{100*r.all_hi:4.1f}]", axis=1)
    sim["Model"] = sim.Model.map(short)
    pv = sim.pivot_table(index=["periods", "Model", "util_mean", "scenario"], columns="policy", values="cell", aggfunc="first")
    pv = pv[[p[0] for p in POLICIES if p[0] in pv.columns]]
    print("\n=== M1 전체 miss율 % [95% 시뮬레이션 변동 구간] ===")
    print(pv.to_string())
    nl = sim[(sim.scenario == "baseline")].copy()
    nl["NLU_miss%"] = (100 * nl.miss_NLU).round(1)
    pv2 = nl.pivot_table(index=["periods", "Model"], columns="policy", values="NLU_miss%", aggfunc="first")
    pv2 = pv2[[p[0] for p in POLICIES if p[0] in pv2.columns]]
    print("\n=== M1b baseline NLU 과제 miss율 % (유효 조건) ===")
    print(pv2.to_string())
    b = blk[(blk.task != "Summarization")].copy(); b["Model"] = b.Model.map(short)
    b = b[b["util_mean"] <= args_global.maxutil]
    print("\n=== M2 결정적 진단: 자체 초과 P(exec>D) 와 비선점 블로킹 위반 (유효 조건, NLU/QA) ===")
    print(b[["periods", "Model", "task", "D_ms", "P(exec>D)_self", "blocker_max_ms", "blocking_violation(NP)"]].to_string(index=False))


# ------------------------------------------------------------------ (2) 상한 모델
def bound_validation(d, models, args):
    rng = np.random.default_rng(7)
    meths = ["emp_max", "emp_max x1.1", "emp_P99", "P50x1.5", "model(a_t+50b)*(1+margin)", "model x1.1"]
    acc = {(m, t, k): {"viol": [], "any": [], "tight": []} for m in models for t in TASKS for k in meths}
    cells = {(m, t): d[(d.Model == m) & (d.Task == t)].reset_index(drop=True) for m in models for t in TASKS}
    okm = {m: d[d.Model == m].tok.std() >= 1.0 for m in models}
    for _ in range(args.splits):
        tr_idx, te_idx = {}, {}
        for key, g in cells.items():
            p = rng.permutation(len(g)); h = len(g) // 2
            tr_idx[key], te_idx[key] = p[:h], p[h:]
        for m in models:
            tr = pd.concat([cells[(m, t)].iloc[tr_idx[(m, t)]] for t in TASKS])
            a, b, _ok = fit_model(tr)
            for t in TASKS:
                g = cells[(m, t)]; trn, ten = g.iloc[tr_idx[(m, t)]], g.iloc[te_idx[(m, t)]]
                B = {"emp_max": trn.lat.max(), "emp_max x1.1": 1.1 * trn.lat.max(),
                     "emp_P99": np.percentile(trn.lat, 99), "P50x1.5": 1.5 * trn.lat.median()}
                if okm[m]:
                    pred = np.maximum(a[t] + b * trn.tok.values, 1.0)
                    mar = max(float((trn.lat.values / pred).max()) - 1.0, 0.0)
                    B["model(a_t+50b)*(1+margin)"] = (a[t] + 50 * b) * (1 + mar)
                    B["model x1.1"] = 1.1 * B["model(a_t+50b)*(1+margin)"]
                for k in meths:
                    if k not in B:
                        continue
                    v = (ten.lat.values > B[k])
                    acc[(m, t, k)]["viol"].append(v.mean()); acc[(m, t, k)]["any"].append(v.any())
                    acc[(m, t, k)]["tight"].append(B[k] / ten.lat.max())
    rows = []
    for (m, t, k), r in acc.items():
        if r["viol"]:
            rows.append({"Model": m, "Task": t, "method": k, "viol_rate": np.mean(r["viol"]),
                         "P(any_viol)": np.mean(r["any"]), "tightness_med": np.median(r["tight"])})
    res = pd.DataFrame(rows)
    save(res.round(4).set_index(["Model", "Task", "method"]), "M3_bound_holdout")
    full = []
    for m in models:
        a, b, ok = fit_model(d[d.Model == m])
        for t in TASKS:
            g = cells[(m, t)]
            pred = np.maximum(a[t] + b * g.tok.values, 1.0)
            mar = max(float((g.lat.values / pred).max()) - 1.0, 0.0)
            full.append({"Model": m, "Task": t, "slope_identified": ok, "ms_per_tok": b, "a_task_ms": a[t],
                         "pred_at_50tok": a[t] + 50 * b, "margin": mar,
                         "bound_model": (a[t] + 50 * b) * (1 + mar), "obs_median": g.lat.median(),
                         "obs_P99": np.percentile(g.lat, 99), "obs_max": g.lat.max(),
                         "bound/obs_max": (a[t] + 50 * b) * (1 + mar) / g.lat.max(),
                         "bound/obs_median": (a[t] + 50 * b) * (1 + mar) / g.lat.median()})
    fu = pd.DataFrame(full).round(3)
    save(fu.set_index(["Model", "Task"]), "M4_bound_full_fit")
    return res, fu


def print_bound(res, fu):
    r = res.copy(); r["Model"] = r.Model.map(short)
    summ = r.groupby(["method", "Task"]).agg(viol_rate=("viol_rate", "mean"), P_any=("P(any_viol)", "mean"),
                                             tight_med=("tightness_med", "median")).round(3)
    print("\n=== M3 홀드아웃 검증: 방법 x 과제 (모델 평균). viol_rate = 검증 절반에서 상한 초과 비율 ===")
    print(summ.to_string())
    allm = r.groupby("method").agg(viol_rate=("viol_rate", "mean"), P_any=("P(any_viol)", "mean"),
                                   tight_med=("tightness_med", "median")).round(3)
    print("\n=== M3b 방법별 전체 요약 (tight = 상한/검증 최댓값, 클수록 보수적) ===")
    print(allm.to_string())
    w = r[r.method.isin(["emp_max", "emp_max x1.1", "model(a_t+50b)*(1+margin)", "model x1.1"])].pivot_table(
        index=["Model", "Task"], columns="method", values=["viol_rate", "tightness_med"]).round(3)
    print("\n=== M3c 셀별: 경험적 최댓값 vs 모델 기반 상한 ===")
    print(w.to_string())
    f = fu.copy(); f["Model"] = f.Model.map(short)
    print("\n=== M4 전체 데이터 적합 (in-sample이라 bound/obs_max=1.0은 정의상 나옴. 의미는 NLU 라벨-only 셀의 과다 할당 배수) ===")
    print(f[["Model", "Task", "slope_identified", "ms_per_tok", "pred_at_50tok", "margin", "bound_model",
             "obs_max", "bound/obs_max"]].to_string(index=False))


# ------------------------------------------------------------------ main
args_global = None


def main():
    global OUT, args_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--H", type=float, default=None, help="시뮬레이션 시간 [ms]")
    ap.add_argument("--splits", type=int, default=None)
    ap.add_argument("--maxutil", type=float, default=0.8)
    ap.add_argument("--nlu_cap", type=int, default=8)
    ap.add_argument("--periods", default="1000,4000,8000;2000,8000,16000;3000,12000,24000",
                    help="세트는 ';'로, 주기(NLU,QA,Summ)는 ','로 구분 [ms]")
    args = ap.parse_args()
    args.reps = args.reps or (6 if args.quick else 20)
    args.H = args.H or (200_000 if args.quick else 600_000)
    args.splits = args.splits or (40 if args.quick else 200)
    args.period_sets = {f"P{i+1}": [float(x) for x in s.split(",")] for i, s in enumerate(args.periods.split(";"))}
    args_global = args
    OUT = args.out or os.path.join(args.base, "revision_out")
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    d = load(args.base)
    models = sorted(d.Model.unique())
    fits = {m: fit_model(d[d.Model == m]) for m in models}
    print("기울기(ms/token), 식별 여부:", {short(m): (round(f[1], 1), f[2]) for m, f in fits.items()})
    print(f"설정: reps={args.reps} H={args.H:.0f}ms maxutil={args.maxutil} nlu_cap={args.nlu_cap} periods={args.period_sets}")
    print("\n[1] 고정 주기 시뮬레이션 ...", flush=True)
    sim, blk = run_sim(d, models, fits, args)
    print_sim(sim, blk)
    print("\n[2] 지연 상한 모델 홀드아웃 검증 ...", flush=True)
    res, fu = bound_validation(d, models, args)
    print_bound(res, fu)
    print(f"\nDone. tables -> {OUT}/tables/M*.csv")


if __name__ == "__main__":
    main()
