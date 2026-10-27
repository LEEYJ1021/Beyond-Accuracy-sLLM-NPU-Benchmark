import re
s = open("run_all.py", encoding="utf-8").read()
res = []
def sub(name, pat, new, flags=0):
    global s
    s2, n = re.subn(pat, lambda m: new, s, count=1, flags=flags)
    res.append((name, "ok" if n else "SKIP"))
    if n: s = s2
sub("fig04d legend", r'ax\[0\]\.set_ylabel\("Trials"\); ax\[0\]\.legend\(.*',
    'ax[0].set_ylabel("Trials"); ax[0].legend(frameon=False, fontsize=5.2, handlelength=1, columnspacing=.8, loc="lower left", bbox_to_anchor=(0.1, 1.0), ncol=2)')
sub("fig05 wspace", r'"wspace":\s*\.8\b', '"wspace":.45')
sub("fig05 legend", r'ax\[1\]\.legend\(fontsize=5\.5.*',
    'h_, l_ = ax[1].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=6, frameon=False, loc="lower center", ncol=8, bbox_to_anchor=(0.4, -0.06))')
old = 'ax.errorbar(range(3), med,'
if old in s:
    s = s.replace(old, 'ax.errorbar(np.arange(3) + {"Small": -.07, "Medium": 0, "Large": .07}[gname], med,', 1); res.append(("fig07 offset", "ok"))
else:
    res.append(("fig07 offset", "SKIP"))
sub("fig09 hspace", r'"hspace":\s*[\d.]+', '"hspace": .65')
sub("fig09 title pad", r'ax\.set_title\(m, color=COL\[m\], fontweight="bold"[^)]*\)', 'ax.set_title(m, color=COL[m], fontweight="bold", pad=12)')
sub("fig09 stats line", r'ax\.text\(\.97, \.04, f"n = .*?alpha=\.85\)\)',
    'ax.text(.5, 1.0, f"n={len(g)}  R²={lr.rvalue**2:.2f}  {lr.slope:.3f} ms/token  spline {gain:+.1f}%", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.5)', re.S)
open("run_all.py", "w", encoding="utf-8").write(s)
for r in res: print(r)
