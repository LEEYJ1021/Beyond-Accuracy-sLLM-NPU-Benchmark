import numpy as np, pandas as pd
from scipy import stats
import additional_analysis as aa
ST = aa.Ctx(); ST.base = aa.DEFAULT_BASE; aa.load(ST)
df = ST.d.copy(); df["model"] = df.Model.map(aa.SHORT)
rows = {}
for m,g in df.groupby("model"):
    x = g.lat.values; p50,p95 = np.percentile(x,[50,95])
    rows[m] = dict(mean=x.mean(), cv=x.std(ddof=1)/x.mean(), tailr=p95/p50, skew=stats.skew(x), kurt=stats.kurtosis(x))
t = pd.DataFrame(rows).T.astype(float)
mm = lambda s:(s-s.min())/(s.max()-s.min()+1e-12)
t["ops"] = .35*mm(t["cv"])+.35*mm(t["tailr"])+.15*mm(t["skew"].abs())+.15*mm(t["kurt"])
out = []
for k in (1.5,2,3):
    miss = df.groupby("model").lat.apply(lambda s:(s>k*s.mean()).mean())
    t[f"miss_{k}xmean"] = miss
    for i in ["mean","cv","tailr","ops"]:
        r,p = stats.spearmanr(t[i], miss); out.append(dict(k=k, indicator=i, rho=round(r,2), p=round(p,3)))
print(pd.DataFrame(out).to_string(index=False))
print(t.round(3).to_string())
t.round(4).to_csv("revision_out/cpu_extra/F_relative_deadline.csv", encoding="utf-8-sig")
