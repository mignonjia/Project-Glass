import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import FancyBboxPatch

out_dir = Path("journal/20260519_glass_motivation/glass_motivation_figure")
out_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(15.5, 8.8))

# Sweet-spot regions
x1 = np.linspace(0.8, 4.1, 140)
y1_low = 0.55 + 0.06 * x1
y1_high = 2.5 + 0.14 * x1
ax.fill_between(x1, y1_low, y1_high, alpha=0.22)

x2 = np.linspace(2.9, 7.2, 160)
y2_low = 1.95 + 0.13 * x2
y2_high = 4.95 + 0.23 * x2
ax.fill_between(x2, y2_low, y2_high, alpha=0.22)

x3 = np.linspace(5.9, 9.7, 140)
y3_low = 4.9 + 0.18 * x3
y3_high = 7.9 + 0.16 * x3
ax.fill_between(x3, y3_low, y3_high, alpha=0.22)

# In-region labels
ax.text(1.9, 1.7, "GPU / NVL72\nsweet spot", fontsize=19, ha="center", va="center", weight="bold")
ax.text(5.0, 4.55, "Dual-panel glass\nsweet spot", fontsize=19, ha="center", va="center", weight="bold")
ax.text(7.85, 7.8, "Multi-panel glass\nsweet spot", fontsize=19, ha="center", va="center", weight="bold")

# Main arrow
ax.text(
    1.2, 9.18,
    "Scaling trend:",
    fontsize=18,
    ha="left",
    va="center",
    weight="bold",
)
ax.text(
    2.9, 9.18,
    "longer horizon + lower latency + larger state",
    fontsize=18,
    ha="left",
    va="center",
)
ax.annotate(
    "",
    xy=(6.7, 8.1),
    xytext=(4.5, 8.8),
    arrowprops=dict(arrowstyle="->", linewidth=2.6),
)

# Representative points with colors
# points = [
#     ("1", 3.05, 2.85, "Wan / Hunyuan", "(2024–2025)"),
#     ("2", 4.25, 4.00, "NVIDIA Cosmos", "(2025)"),
#     ("3", 6.35, 6.10, "Helios / LongCat", "(2025–2026)"),
#     ("4", 7.85, 7.15, "LongLive", "(2025–2026)"),
# ]
points = [
    ("1", 2.85, 2.55, "Wan / Hunyuan", "(2024–2025)"),
    ("2", 4.3, 3.35, "Cosmos-Predict", "(2025)"),
    ("3", 6.2, 5.25, "Helios / LongCat", "(2025–2026)"),
    ("4", 8, 7.2, "LongLive", "(2025–2026)"),
    ("5", 9.25, 8, "StreamDiffusionV2", "(2026)"),
]

pt_colors = {
    "1": "#d62728",  # red
    "2": "#9467bd",  # purple
    "3": "#8c564b",  # brown
    "4": "#e377c2",  # pink
    "5": "#58a7e3",  # blue
}

for num, px, py, _, _ in points:
    ax.scatter([px], [py], s=260, c=pt_colors[num], edgecolors="black", linewidths=0.8, zorder=6)
    ax.text(px, py, num, fontsize=13, ha="center", va="center", weight="bold", zorder=7, color="white")

# Right-side model callout box with matching colored circles
box_x, box_y_top = 6.9, 3.15
box_width = 2.95
box_height = 2.8
box_pad_x = 0.15
box_pad_top = 0.14
callout_bg = FancyBboxPatch(
    (box_x - box_pad_x, box_y_top - box_height),
    box_width,
    box_height + box_pad_top,
    boxstyle="round,pad=0.08",
    facecolor="#d9e6eb",
    edgecolor="#9fb0b8",
    linewidth=0.9,
    alpha=0.72,
    zorder=4,
)
ax.add_patch(callout_bg)
ax.text(
    box_x, box_y_top, "Representative video models",
    fontsize=14.5, ha="left", va="top",
    zorder=9,
)

entry_start_y = box_y_top - 0.68
entry_gap = 0.45
circle_x = box_x + 0.10
text_x = box_x + 0.28

for i, (num, _, _, model_name, year) in enumerate(points):
    y = entry_start_y - i * entry_gap
    ax.scatter([circle_x], [y], s=160, c=pt_colors[num], edgecolors="black", linewidths=0.8, zorder=8)
    ax.text(circle_x, y, num, fontsize=11.5, ha="center", va="center", weight="bold", color="white", zorder=9)
    ax.text(text_x, y, f"{model_name}  {year}", fontsize=13, ha="left", va="center", zorder=9)

# Axes
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("Workload scale: short clips  →  long clips  →  real-time / RL", fontsize=20, labelpad=12)
ax.set_ylabel("Demand of Memory Capacity", fontsize=20, labelpad=12)
ax.set_title("Open-Source Video Models Move Beyond the GPU Memory Sweet Spot", fontsize=24, pad=18, weight="bold")

ax.set_xticks([1.5, 4.3, 7.2, 9.2])
ax.set_xticklabels(["seconds", "10–20s", "minute-scale", "real-time / RL"], fontsize=18)

# y-axis with 3 labels
ax.set_yticks([1.5, 5.0, 8.7])
ax.set_yticklabels(["HBM Fits", "Dual-Panel Helps", "Multi-Panel Helps"], fontsize=18)

ax.grid(True, alpha=0.25)

fig.tight_layout()

png_path = out_dir / "glass_video_world_model_sweet_spot.png"
svg_path = out_dir / "glass_video_world_model_sweet_spot.svg"
fig.savefig(png_path, dpi=280, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
plt.close(fig)

print(png_path)
print(svg_path)
