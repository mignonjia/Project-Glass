"""Access-pattern figure (VSA-style only), larger fonts.
Curves were digitized from the original plot_report.py SVG (server worktree exp/realtime-profile);
data in access_pattern_data.json next to this script."""
import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BG, INK, AX, GRID, BLUE, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc", "#2a78d6", "#8a887f"
d = json.load(open(__file__.replace("plot_access_pattern.py", "access_pattern_data.json")))
plt.rcParams.update({"font.size": 16, "text.color": INK, "axes.labelcolor": AX, "xtick.color": AX,
                     "ytick.color": AX, "axes.edgecolor": AX})
panels = [  # (data key, title, definition, ylim, end label, PAN value)
    ("1", "(a) Read per forward",
     "blocks picked by the 48 query blocks\n÷ all older blocks", 50, "~40%", 21.8),
    ("2", "(b) New between forwards",
     "share of a forward's blocks that the\nprevious forward did not read", 20, "~7%", 6.7),
    ("0", "(c) Shared with previous chunk",
     "share of this chunk's blocks that the\nprevious chunk also read", 100, "~74%", 65.0),
]
fig, axs = plt.subplots(1, 3, figsize=(16, 6.2), facecolor=BG)
for ax, (k, title, defn, ylim, endlab, pan) in zip(axs, panels):
    x, y = zip(*d[k]["vsa"])
    ax.set_facecolor(BG)
    ax.plot(x, y, color=BLUE, lw=2.6)
    ax.axhline(pan, color=REF, ls="--", lw=1.6)
    ax.text(124, pan - ylim * 0.03, f"PAN, trained VSA ({pan:g}%)", ha="right", va="top", fontsize=13, color=AX)
    ax.text(x[-1], max(y[-12:]) + ylim * 0.04, endlab, ha="right", va="bottom", fontsize=17, fontweight="bold", color=BLUE)
    ax.set_title(title, loc="left", fontsize=16.5, fontweight="bold", pad=52)
    ax.text(0, 1.03, defn, transform=ax.transAxes, fontsize=13.5, color=AX, va="bottom")
    ax.set_xlim(0, 125); ax.set_ylim(0, ylim); ax.set_xlabel("history (s)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.grid(color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.text(0.01, 0.975, "Which older KV blocks are read (VSA-style top-5% selection)", fontsize=20, fontweight="bold", va="top")
fig.text(0.01, 0.905, "(a) sets the fetch per forward; (b) how much of it changes across the 5 forwards of a chunk; "
         "(c) how much the next chunk can reuse", fontsize=14, color=AX, va="top")
fig.text(0.01, 0.015, "LingBot 14B, SP8, full-history KV, 512x768. Older = beyond the 12 most recent latent frames. "
         "Per layer and head, averaged.", fontsize=12, color=AX)
fig.subplots_adjust(left=0.05, right=0.985, top=0.70, bottom=0.14, wspace=0.28)
for ext in ("png", "svg"):
    fig.savefig(__file__.replace("plot_access_pattern.py", f"figures/access_pattern.{ext}"), dpi=180, facecolor=BG)
