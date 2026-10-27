s = open("run_all.py", encoding="utf-8").read()
P = [
# fig03c
(r'fig, ax = plt.subplots(figsize=(3.0, 3.2))', r'fig, ax = plt.subplots(figsize=(4.4, 3.2))'),
(r'ax.set_xlim(-.9, 1.9)', r'ax.set_xlim(-1.9, 2.9)'),
# fig04d 범례
(r'ax[0].legend(frameon=False, fontsize=6)', r'ax[0].legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=2)'),
# fig05
(r'fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.7), gridspec_kw={"width_ratios":[1,1,1.15]})',
 r'fig, ax = plt.subplots(1, 3, figsize=(9.5, 3.0), gridspec_kw={"width_ratios":[1,1,1.15], "wspace":.8})'),
(r'a.scatter(r[x], r.slope, color=COL[r.model], s=22, zorder=3); a.annotate(r.model, (r[x], r.slope), fontsize=6, xytext=(3, 3), textcoords="offset points")',
 r'a.scatter(r[x], r.slope, color=COL[r.model], s=22, zorder=3, label=r.model)'),
('ax[0].set_xscale("log")\n    H = df.pivot_table',
 'ax[0].set_xscale("log"); ax[1].legend(fontsize=5.5, frameon=False, loc="lower right", ncol=2)\n    H = df.pivot_table'),
# fig07 범례
(r'ax.legend(frameon=False, title="Model size group")', r'ax.legend(frameon=False, title="Model size group", loc="center left", bbox_to_anchor=(1.02, .5))'),
# fig08c 범례
(r'ax.set_ylim(0, 1.05); save(fig, "fig08c")',
 r'ax.set_ylim(0, 1.05); ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=COL[m], label=m, markersize=4) for m in ORDER], fontsize=5.5, frameon=False, loc="center left", bbox_to_anchor=(1.02, .5)); save(fig, "fig08c")'),
# fig09
(r'fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.0)); rows = []',
 r'fig, axes = plt.subplots(3, 3, figsize=(7.5, 7.8), gridspec_kw={"hspace": .5, "wspace": .3}); rows = []'),
(r'ax.text(.03, .97, f"n = ', r'ax.text(.97, .04, f"n = '),
(r'transform=ax.transAxes, va="top", fontsize=6.5, bbox=dict(fc="w", ec="0.7"))',
 r'transform=ax.transAxes, ha="right", va="bottom", fontsize=6, bbox=dict(fc="w", ec="0.7", alpha=.85))'),
(r'ms/token\nspline gain', r'ms/input token\nspline gain'),
# fig11
(r'figsize=(7.5, 2.9), gridspec_kw={"width_ratios": [.8, 1.4, 1.4]}', r'figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.2, 1.4, 1.4], "wspace": .55}'),
(r'for i, t in enumerate(TASKS): ax[0].text(i + 1, d.lat.max() * 1.02, f"CV={cv[t]:.2f}", ha="center", fontsize=6.5)',
 r'ax[0].set_xticklabels([f"{t}\nCV={cv[t]:.2f}" for t in TASKS], fontsize=6.5)'),
(r'ax[1].legend(frameon=False)', r'ax[1].set_ylim(0, 22); ax[1].legend(frameon=False, fontsize=6, loc="upper left", ncol=3)'),
# fig15d 범례
(r'ax.set_ylabel("Miss rate (%)"); save(fig, "fig15d")',
 'ax.set_ylabel("Miss rate (%)")\n'
 '    h = [plt.Line2D([], [], marker=mk[k], ls="", color="gray", label=k, markersize=4) for k in mk]\n'
 '    h += [plt.Line2D([], [], marker="o", ls="", color="gray", label="non-preemptive", markersize=4), plt.Line2D([], [], marker="o", ls="", mfc="none", mec="gray", label="preemptive (max)", markersize=4)]\n'
 '    ax.legend(handles=h, fontsize=6, frameon=False, loc="center right"); save(fig, "fig15d")'),
]
for old, new in P:
    assert s.count(old) == 1, f"찾지 못했거나 중복({s.count(old)}회): {old[:70]}"
    s = s.replace(old, new)
open("run_all.py", "w", encoding="utf-8").write(s); print("patched", len(P))
