"""Figure 1 of the paper: acceleration factor against 30 C / 65 % RH.

Run from the project folder:  python analysis/make_figure.py
Reads results/acceleration_factors.csv, writes paper/fig_af.pdf and .png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
a = pd.read_csv(ROOT / "results" / "acceleration_factors.csv")
a = a[a["reference"] == "30C/65%RH"]

# top to bottom: highlands first, then lowlands by humidity
ORDER = ["Quetzaltenango", "Huehuetenango", "Guatemala La Aurora", "Coban",
         "Zacapa", "Retalhuleu", "San Jose", "Mundo Maya (Flores)", "Puerto Barrios"]
LABEL = {"Guatemala La Aurora": "Guatemala City", "Mundo Maya (Flores)": "Flores",
         "Coban": "Cobán", "San Jose": "San José"}
NIGHT_FILLED = set(a.loc[a["method"].str.contains("reanalysis"), "station"])

# one hue, light to dark, because B is an ordered quantity; shapes repeat the
# encoding so the figure survives greyscale printing
SERIES = [(0.00, "#6baed6", "o", "B = 0 (no humidity effect)"),
          (0.04, "#2a78d6", "^", "B = 0.04"),
          (0.08, "#0b3d91", "s", "B = 0.08")]

plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.45, 3.3))
ax.axvline(1.0, color="#333333", lw=0.9, zorder=1)

for i, st in enumerate(ORDER):
    y0 = len(ORDER) - 1 - i
    for j, (b, color, marker, _) in enumerate(SERIES):
        y = y0 + (1 - j) * 0.24
        d = a[(a["station"] == st) & (a["B_per_pctRH"] == b)]
        ax.plot([d["AF"].min(), d["AF"].max()], [y, y], color=color, lw=1.1,
                solid_capstyle="round", zorder=2)
        mid = d.loc[d["Ea_kJ_mol"] == 83.144, "AF"].iloc[0]
        ax.plot(mid, y, marker=marker, ms=4.6, color=color, mec="#1a1a1a",
                mew=0.5, ls="none", zorder=3)

for i in range(len(ORDER) - 1):
    ax.axhline(i + 0.5, color="#dddddd", lw=0.5, zorder=0)

ax.set_xscale("log")
ax.set_xlim(0.06, 11)
ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
ax.minorticks_off()
ax.set_ylim(-0.6, len(ORDER) - 0.4)
ax.set_yticks(range(len(ORDER)))
ax.set_yticklabels([LABEL.get(s, s) + ("*" if s in NIGHT_FILLED else "")
                    for s in reversed(ORDER)])
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Acceleration factor vs. 30 °C / 65% RH", x=0.36)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

handles = [plt.Line2D([], [], marker=m, color=c, mec="#1a1a1a", mew=0.5, ms=4.6,
                      lw=1.1, label=lab) for _, c, m, lab in SERIES]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.36, 1.17),
          ncol=3, frameon=False, fontsize=6.6, handlelength=1.6,
          columnspacing=0.9, handletextpad=0.4)
ax.text(0.93, len(ORDER) - 0.42, "milder", ha="right", va="bottom", fontsize=6.5, color="#555555")
ax.text(1.08, len(ORDER) - 0.42, "harsher than test", ha="left", va="bottom", fontsize=6.5, color="#555555")

fig.tight_layout(pad=0.3)
(ROOT / "paper").mkdir(exist_ok=True)
fig.savefig(ROOT / "paper" / "fig_af.pdf")
fig.savefig(ROOT / "paper" / "fig_af.png", dpi=300)
print("written paper/fig_af.pdf and paper/fig_af.png")
