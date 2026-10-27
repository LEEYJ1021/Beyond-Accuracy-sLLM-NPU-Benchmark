import re
s = open("run_all.py", encoding="utf-8").read()
P = [
# fig04d: 범례를 (e) 라벨 앞에서 끝나도록 축소
('ax[0].legend(frameon=False, fontsize=6, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=2)',
 'ax[0].legend(frameon=False, fontsize=5.2, handlelength=1, columnspacing=.8, loc="lower left", bbox_to_anchor=(0.1, 1.0), ncol=2)'),
# fig05: 여백 축소 + 범례를 그림 아래로
('"wspace":.8})', '"wspace":.45})'),
('ax[1].legend(fontsize=5.5, frameon=False, loc="lower right", ncol=2)',
 'h_, l_ = ax[1].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=6, frameon=False, loc="lower center", ncol=8, bbox_to_anchor=(0.4, -0.06))'),
# fig07: Medium/Large 오차막대 겹침 방지
('ax.errorbar(range(3), med,', 'ax.errorbar(np.arange(3) + {"Small": -.07, "Medium": 0, "Large": .07}[gname], med,'),
# fig09: 제목 여백과 간격
('{"hspace": .6, "wspace": .3}' if '{"hspace": .6, "wspace": .3}' in s else '{"hspace": .5, "wspace": .3}', '{"hspace": .65, "wspace": .3}'),
('ax.set_title(m, color=COL[m], fontweight="bold")', 'ax.set_title(m, color=COL[m], fontweight="bold", pad=12)'),
]
for old, new in P:
    assert s.count(old) == 1, f"찾지 못했거나 중복({s.count(old)}회): {old[:70]}"
    s = s.replace(old, new)
# fig09: 통계 박스 → 패널 제목 아래 한 줄
pat = r'ax\.text\(\.97, \.04, f".*?bbox=dict\(fc="w", ec="0\.7", alpha=\.85\)\)'
new = 'ax.text(.5, 1.0, f"n = {len(g)}, R² = {lr.rvalue**2:.2f}, {lr.slope:.3f} ms/input token, spline gain {gain:+.1f}%", transform=ax.transAxes, ha="center", va="bottom", fontsize=5.2)'
s, n = re.subn(pat, lambda m: new, s, count=1, flags=re.S)
assert n == 1, "fig09 통계 박스 줄을 찾지 못함"
open("run_all.py", "w", encoding="utf-8").write(s); print("patched")
