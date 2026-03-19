"""Generate 560x280 thumbnail for Kaggle submission."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BG = "#0d1117"
TEXT = "#e6edf3"
ACCENT = "#4ec9b0"
COLORS = {"Qwen2.5-72B": "#4ec9b0", "Llama3.1-8B": "#5b9bd5", "Phi-3-mini": "#f0c060"}

# 560x280 px at 200 DPI = 2.8 x 1.4 inches
fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
fig.set_facecolor(BG)

# ── Left side: Title text ──────────────────────────────────────────────────
ax_text = fig.add_axes([0.02, 0.05, 0.48, 0.9])
ax_text.set_facecolor(BG)
ax_text.set_xlim(0, 1)
ax_text.set_ylim(0, 1)
ax_text.axis("off")

ax_text.text(0.05, 0.78, "COG", fontsize=30, fontweight="bold", color=ACCENT,
             fontfamily="monospace", va="top")
ax_text.text(0.05, 0.53, "Attention", fontsize=22, fontweight="bold", color=TEXT,
             fontfamily="sans-serif", va="top")

ax_text.text(0.05, 0.32, "8 tasks  |  5 abilities  |  256 items",
             fontsize=8, color="#8b949e", fontfamily="sans-serif", va="top")

# Small CAS bars
models = ["Qwen-72B", "Llama-8B", "Phi-3"]
scores = [0.81, 0.73, 0.57]
colors = ["#4ec9b0", "#5b9bd5", "#f0c060"]
bar_y = [0.18, 0.11, 0.04]
for m, s, c, y in zip(models, scores, colors, bar_y):
    ax_text.barh(y, s * 0.55, height=0.05, left=0.05, color=c, zorder=3)
    ax_text.text(0.05 + s * 0.55 + 0.02, y, f"{m} {s:.2f}",
                 fontsize=5.5, color=TEXT, va="center")

# ── Right side: Mini radar chart ───────────────────────────────────────────
ax_radar = fig.add_axes([0.52, 0.05, 0.46, 0.9], polar=True)
ax_radar.set_facecolor(BG)

abilities = ["Capacity", "Sustained", "Selective", "Shifting", "Stim-Driven"]
N = len(abilities)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

ax_radar.set_theta_offset(np.pi / 2)
ax_radar.set_theta_direction(-1)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(abilities, fontsize=6, color="#8b949e")
ax_radar.set_ylim(0, 1.05)
ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_radar.set_yticklabels([], fontsize=0)
ax_radar.yaxis.grid(color="#21262d", linewidth=0.4)
ax_radar.xaxis.grid(color="#21262d", linewidth=0.4)
ax_radar.spines["polar"].set_color("#21262d")

# Model profiles (from results)
profiles = {
    "Qwen2.5-72B": [0.83, 0.96, 0.84, 0.91, 0.43],
    "Llama3.1-8B": [0.59, 0.90, 0.87, 0.69, 0.50],
    "Phi-3-mini":  [0.49, 0.65, 0.72, 0.69, 0.15],
}
for model, vals in profiles.items():
    v = vals + vals[:1]
    ax_radar.plot(angles, v, linewidth=1.5, color=COLORS[model], zorder=3)
    ax_radar.fill(angles, v, alpha=0.10, color=COLORS[model])

fig.savefig("/data/home/rnaa/cognitive_ability/figures/thumbnail.png",
            dpi=100, bbox_inches="tight", facecolor=BG, pad_inches=0.05)
plt.close(fig)
print("[OK] thumbnail.png")
