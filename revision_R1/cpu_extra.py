import sys, numpy as np, pandas as pd
from scipy import stats
import additional_analysis as aa
OUT = "revision_out/cpu_extra/"; rng = np.random.default_rng(0)
RUN_SIM = "--sim" in sys.argv; B = 500

ST = aa.Ctx(); ST.base = aa.DEFAULT_BASE; aa.load(ST)
df = ST.d.rename(columns={"Model":"model","Task":"task","lat":"lat","tok":"tok"}).copy()
df["idx"] = df.Sample.astype(int) - 1
df["model"] = df.model.map(aa.SHORT)
PAR = {aa.SHORT[m]: aa.PARAMS[m] for m in aa.PARAMS}
CTX = {aa.SHORT[m]: v for m, v in ST.hf_ctx.items()}        # 모델카드 값 (컴파일 max_seq_len 아님)
print(len(df), "valid trials"); print(df.groupby(["model","task"]).size().unstack())

def ind(x):
    p50,p95,p99 = np.percentile(x,[50,95,99])
    return dict(mean=x.mean(), cv=x.std(ddof=1)/x.mean(), tailr=p95/p50, p99r=p99/p50, maxr=x.max()/p50,
                skew=stats.skew(x), kurt=stats.kurtosis(x))
def add_ops(t):
    mm = lambda s:(s-s.min())/(s.max()-s.min()+1e-12)
    t["ops"] = .35*mm(t["cv"])+.35*mm(t["tailr"])+.15*mm(t["skew"].abs())+.15*mm(t["kurt"]); return t
def r2(X,y):
    X = np.column_stack([np.ones(len(y)),X]); b = np.linalg.lstsq(X,y,rcond=None)[0]
    return 1-(y-X@b).var()/y.var(), b

# ---------- A. 입력 토큰 수 회귀 (prefill 대리) ----------
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    sst = load_dataset("nyu-mll/glue","sst2",split="validation").shuffle(seed=42).select(range(100))
    qa  = load_dataset("rajpurkar/squad_v2",split="validation").shuffle(seed=42).select(range(100))
    cn  = load_dataset("abisee/cnn_dailymail","3.0.0",split="validation").shuffle(seed=42).select(range(100))
    txt = {"NLU":sst["sentence"], "QA":[c+"\n"+q for c,q in zip(qa["context"],qa["question"])], "Summarization":cn["article"]}
    IN = {t:np.array([len(tk(s).input_ids) for s in v]) for t,v in txt.items()}
    rows = []
    for (m,t),g in df.groupby(["model","task"]):
        xi = IN[t][g.idx.values]; y = g.lat.values; k = g.tok.values
        var = k.std() > 0
        r_tok = r2(k,y)[0] if var else 0.0
        r_full,b = r2(np.column_stack([k,xi]) if var else xi, y)
        s = k == 50
        rows.append(dict(model=m, task=t, n=len(g), R2_tok=r_tok, R2_tok_in=r_full,
                         partial_R2_in=(r_full-r_tok)/(1-r_tok+1e-12), ms_per_in_tok=b[-1],
                         rho_in_vs_lat_tok50=stats.spearmanr(xi[s],y[s])[0] if s.sum()>10 else np.nan, n50=int(s.sum())))
    A = pd.DataFrame(rows); A.to_csv(OUT+"A_input_token_regression.csv", index=False, encoding="utf-8-sig")
    print("\n[A] input-token regression\n", A.round(3).to_string())
    print("\n[A-check] 요약 tok50 rho가 모델 간 모두 양(+)으로 크면 정렬 OK:\n",
          A[A.task=="Summarization"][["model","rho_in_vs_lat_tok50","n50"]].round(2).to_string(index=False))
except Exception as e:
    print("[A] skipped:", repr(e))

# ---------- B. 절대 마감 miss vs 지표 (+ 부트스트랩, split-half) ----------
Ds = [500,1000,2000,3000]
def table(d):
    r = pd.DataFrame({m:ind(g.lat.values) for m,g in d.groupby("model")}).T.astype(float)
    for D in Ds: r[f"miss{D}"] = d.groupby("model").lat.apply(lambda s:(s>D).mean())
    return add_ops(r)
full = table(df); full.round(4).to_csv(OUT+"B_stats_pooled.csv", encoding="utf-8-sig")
IND = ["mean","cv","tailr","ops"]
boots = {(i,D):[] for i in IND for D in Ds}
for _ in range(B):
    d = pd.concat([g.sample(len(g),replace=True,random_state=int(rng.integers(1e9))) for _,g in df.groupby("model")])
    t = table(d)
    for i in IND:
        for D in Ds: boots[(i,D)].append(stats.spearmanr(t[i],t[f"miss{D}"])[0])
te, to = table(df[df.idx%2==0]), table(df[df.idx%2==1])
rows = []
for i in IND:
    for D in Ds:
        bs = np.array(boots[(i,D)])
        rows.append(dict(indicator=i, deadline_ms=D, rho=stats.spearmanr(full[i],full[f"miss{D}"])[0],
                         ci_lo=np.nanpercentile(bs,2.5), ci_hi=np.nanpercentile(bs,97.5),
                         rho_splithalf=stats.spearmanr(te[i],to[f"miss{D}"])[0]))
Bt = pd.DataFrame(rows); Bt.to_csv(OUT+"B_indicator_vs_absolute_miss.csv", index=False, encoding="utf-8-sig")
print("\n[B] indicator vs absolute-deadline miss\n", Bt.round(2).to_string())

# ---------- C. 한계비용(기울기)·절편, 컨텍스트 재검정, 출력 cap what-if ----------
rows = []
for m,g in df.groupby("model"):
    if g.tok.std() > 0:
        sl,ic,*_ = stats.linregress(g.tok,g.lat)
    else: sl,ic = np.nan,np.nan
    rows.append(dict(model=m, slope_ms_per_tok=sl, intercept_ms=ic,
                     ratio_ms_per_tok=(g.lat/g.tok.clip(lower=1)).mean(), params=PAR[m], ctx=CTX[m]))
C = pd.DataFrame(rows).set_index("model"); C.round(2).to_csv(OUT+"C_marginal_cost.csv", encoding="utf-8-sig"); print("\n[C]\n", C.round(2))
Cc = C.dropna()
for col in ["slope_ms_per_tok","ratio_ms_per_tok"]:
    rk = lambda s:s.rank().values
    res = lambda y,x: y-np.polyval(np.polyfit(x,y,1),x)
    rho,p = stats.spearmanr(Cc.ctx,Cc[col])
    pr,pp = stats.pearsonr(res(rk(Cc.ctx),rk(Cc.params)), res(rk(Cc[col]),rk(Cc.params)))
    print(f"ctx vs {col}: rho={rho:.2f} p={p:.2f} | partial(params) r={pr:.2f} p={pp:.2f} (n={len(Cc)})")
rows = []
for cap in [8,16,32]:
    for (m,t),g in df.groupby(["model","task"]):
        b = C.loc[m,"slope_ms_per_tok"]; b = 0 if np.isnan(b) else b
        l2 = np.where(g.tok>cap, g.lat-b*(g.tok-cap), g.lat)
        for D in [1000,2000]:
            rows.append(dict(cap=cap, model=m, task=t, D=D, miss_orig=(g.lat>D).mean(), miss_cap=(l2>D).mean()))
pd.DataFrame(rows).to_csv(OUT+"C_output_cap_whatif.csv", index=False, encoding="utf-8-sig")

# ---------- D. 과제 층화 꼬리 ----------
rows = []
for (m,t),g in df.groupby(["model","task"]):
    for nm,s in [("all",g),("tok50",g[g.tok==50])]:
        if len(s) >= 20: rows.append(dict(model=m, task=t, subset=nm, n=len(s), **ind(s.lat.values)))
D_ = pd.DataFrame(rows); D_.round(3).to_csv(OUT+"D_stratified_tails.csv", index=False, encoding="utf-8-sig")
print("\n[D] 과제별 tok50 꼬리(중앙값)\n", D_[D_.subset=="tok50"].groupby("task")[["cv","tailr","p99r","maxr"]].median().round(2))

# ---------- E. 선점 시뮬레이션 (양자=토큰시간 배수, 재개 오버헤드 민감도) ----------
def sim(smp, per, Q, ovh, H):
    jobs = []
    for t,P in per.items():
        ph = rng.uniform(0,P); n = int(H//P); ex = rng.choice(smp[t], n)
        jobs += [[ph+k*P, ph+(k+1)*P, ex[k], t, False] for k in range(n)]
    jobs.sort(key=lambda j:j[0]); now=0.; i=0; ready=[]; last=None; miss={t:[0,0] for t in per}
    while i < len(jobs) or ready:
        while i < len(jobs) and jobs[i][0] <= now: ready.append(jobs[i]); i += 1
        if not ready: now = jobs[i][0]; continue
        j = min(ready, key=lambda x:x[1])
        if j[4] and last is not j: now += ovh
        run = j[2] if Q is None else min(j[2],Q); now += run; j[2] -= run; j[4] = True
        if j[2] <= 1e-9:
            ready.remove(j); miss[j[3]][1] += 1; miss[j[3]][0] += now > j[1]; last = None
        else: last = j
    n = sum(v[1] for v in miss.values()); return sum(v[0] for v in miss.values())/max(n,1)
if RUN_SIM:
    PER = {"P1":dict(NLU=1e3,QA=4e3,Summarization=8e3), "P2":dict(NLU=2e3,QA=8e3,Summarization=16e3), "P3":dict(NLU=3e3,QA=12e3,Summarization=24e3)}
    rows = []
    for m in df.model.unique():
        smp = {t:g.lat.values for t,g in df[df.model==m].groupby("task")}
        tc = C.loc[m,"slope_ms_per_tok"]; tc = 30 if np.isnan(tc) else tc
        for pn,per in PER.items():
            U = sum(smp[t].mean()/per[t] for t in per)
            pols = [("NP",None,0)] + [(f"Q{k}tok_ovh{o}", k*tc, o) for k in (1,4,16) for o in (0,20,50)]
            for nm,Q,o in pols:
                r = [sim(smp,per,Q,o,6e5) for _ in range(20)]
                rows.append(dict(model=m, periods=pn, U=U, policy=nm, miss=np.mean(r), sd=np.std(r)))
    S = pd.DataFrame(rows); S.to_csv(OUT+"E_sim_preemption_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\n[E] P1\n", S[S.periods=="P1"].pivot(index="model",columns="policy",values="miss").round(3).to_string())
