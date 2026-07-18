"""E040 @0.12 result figure: cross rate + V(spawn) vs spawn momentum, 3 critics.
Parses a probe_*.out file (the value-ordering probe output) -> two-panel PNG.

  python tools_e040/make_figure.py tools_e040/probe_final.out tools_e040/e040_w012.png
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "tools_e040/probe_final.out"
out = sys.argv[2] if len(sys.argv) > 2 else "tools_e040/e040_w012.png"

# parse: sections "===== <name> ..." then rows "vx~ X.X (s=..): V(spawn) +v cross c"
data, cur = {}, None
for ln in open(src):
  m = re.search(r"=====\s*([\w-]+)", ln)
  if m:
    cur = m.group(1); data[cur] = {"vx": [], "V": [], "cross": []}
    continue
  r = re.search(r"vx~\s*([\d.]+).*V\(spawn\)\s*([+-][\d.]+)\s+cross\s+([\d.]+)", ln)
  if r and cur:
    data[cur]["vx"].append(float(r.group(1)))
    data[cur]["V"].append(float(r.group(2)))
    data[cur]["cross"].append(float(r.group(3)))

STYLE = {"avoid": ("#888888", "o", "avoid (E025)"),
         "buggy-RA": ("#C44", "s", "buggy-RA (E025, g-anchor)"),
         "corr-RA": ("#E77500", "D", "corr-RA (E040, min(l,g))")}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
for name, (c, mk, lab) in STYLE.items():
  key = next((k for k in data if k.startswith(name)), None)
  if not key:
    continue
  d = data[key]
  a1.plot(d["vx"], d["cross"], marker=mk, color=c, lw=2.2, ms=7, label=lab)
  a2.plot(d["vx"], d["V"], marker=mk, color=c, lw=2.2, ms=7, label=lab)

a1.axvspan(0.3, 0.9, color="#E77500", alpha=0.07)
a1.set_title("Crossing rate from near-edge spawn", fontsize=12, fontweight="bold")
a1.set_xlabel("spawn momentum  vx (m/s)"); a1.set_ylabel("cross rate")
a1.set_ylim(-0.03, 1.05); a1.invert_xaxis(); a1.grid(alpha=0.3)
a1.legend(fontsize=9, loc="lower right")
a1.annotate("standstill:\ncorr-RA commits (1.00),\navoid/buggy stall (~0.1)",
            xy=(0.4, 1.0), xytext=(1.1, 0.55), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#E77500"))

a2.set_title("Critic value V(spawn)", fontsize=12, fontweight="bold")
a2.set_xlabel("spawn momentum  vx (m/s)"); a2.set_ylabel("V(spawn)")
a2.invert_xaxis(); a2.grid(alpha=0.3); a2.legend(fontsize=9, loc="best")
a2.annotate("buggy over-certifies\nthe loiter state\n(max V, 7% cross)",
            xy=(0.4, 1.069), xytext=(1.3, 1.15), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color="#C44"))

fig.suptitle("E040 split test @ gap 0.12 — corrected reach-avoid initiates from standstill "
             "with a sound certificate", fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
