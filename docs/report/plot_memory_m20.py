import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BG, INK, AX, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
BASE = 1016  # ms per chunk, 12-frame window, no memory (realtime_profile.md §6.1)
rows = [  # label, fetch, write-back, recompute  (M = 20, ms per chunk)
    ("Stored KV in HBM",       18,  0,   0),
    ("Stored KV in host DRAM", 109, 28,  0),
    ("Recompute from latents", 0,   0, 757),
]
plt.rcParams.update({"font.size": 12, "text.color": INK, "axes.labelcolor": AX,
                     "xtick.color": AX, "ytick.color": AX, "axes.edgecolor": AX})
fig, ax = plt.subplots(figsize=(10, 3.9), facecolor=BG)
ax.set_facecolor(BG)
for i, (lab, f, w, r) in enumerate(rows):
    y = len(rows) - 1 - i; x = 0
    for v, c, name in ((f, BLUE, "fetch stored KV"), (w, GREEN, "write KV to bank"), (r, ORANGE, "recompute memory KV")):
        if v:
            ax.barh(y, v, left=x, height=0.55, color=c, label=name if i == [0, 1, 2][[0, 1, 2].index(i)] else None)
            x += v
    ax.text(x + 12, y, f"+{x} ms  ({x / BASE:.0%})", va="center", fontsize=12, fontweight="bold", color=INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlim(0, 1000); ax.set_xlabel("extra ms per chunk (1 s of video) to bring in 20 memory frames")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
h, l = ax.get_legend_handles_labels(); seen = {}
for hh, ll in zip(h, l): seen.setdefault(ll, hh)
ax.legend(seen.values(), seen.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
fig.text(0.01, 0.955, "Bringing 20 memory frames into attention: fetching stored KV is cheap, recomputing is not",
         fontweight="bold", fontsize=13.5, va="top")
fig.text(0.01, 0.885, f"Extra time per chunk, and % of the fixed ~{BASE:,} ms per chunk (linear layers, window attention, all-to-all)",
         fontsize=11.5, color=AX, va="top")
fig.text(0.01, 0.015,
         "LingBot 14B, SP8, 8xH200, 512x768, 12-frame window + 20 memory frames, 60 s rollout, mean of last 10 chunks.\n"
         "Host-DRAM run also slows the rest of the chunk by ~330 ms (cause not yet measured; not shown).",
         fontsize=9, color=AX)
fig.subplots_adjust(left=0.22, right=0.98, top=0.80, bottom=0.36)
for ext in ("png", "svg"):
    fig.savefig(f"memory_time.{ext}", dpi=260, facecolor=BG)
