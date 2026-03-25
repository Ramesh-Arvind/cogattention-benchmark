"""Phase 5: Visualization & Analysis — publication-quality figures for the CAS benchmark."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────
BG       = "#0d1117"
GRID_CLR = "#21262d"
TEXT_CLR  = "#e6edf3"
COLORS   = {"Qwen2.5-72B": "#4ec9b0", "Llama3.1-8B": "#5b9bd5", "Phi-3-mini": "#f0c060"}
MODEL_ORDER = ["Qwen2.5-72B", "Llama3.1-8B", "Phi-3-mini"]
MODEL_KEYS  = {"Qwen2.5-72B": "qwen72b", "Llama3.1-8B": "llama8b", "Phi-3-mini": "phi3"}
TASK_ORDER  = ["sustained", "selective", "shifting", "capacity", "interference",
               "stroop", "stream_segregation", "anomaly"]
TASK_LABELS = {
    "sustained": "Sustained",
    "selective": "Selective",
    "shifting": "Shifting",
    "capacity": "Capacity",
    "interference": "Interference",
    "stroop": "Stroop",
    "stream_segregation": "Stream Seg.",
    "anomaly": "Anomaly",
}
DIFF_ORDER = ["Easy", "Medium", "Hard", "Expert"]


def _apply_dark_style(ax, title=""):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.tick_params(colors=TEXT_CLR, which="both")
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_CLR, pad=12)


def load_cas(model_key):
    path = os.path.join(RESULTS, f"pilot_{model_key}_cas.json")
    with open(path) as f:
        return json.load(f)


def load_metrics():
    with open(os.path.join(RESULTS, "pilot_metrics.json")) as f:
        return json.load(f)


# ── Figure 1: CAS Score Bar Chart ─────────────────────────────────────────────
def fig_cas_scores():
    data = {name: load_cas(key)["cas_score"] for name, key in MODEL_KEYS.items()}
    # Sort descending
    sorted_models = sorted(data.keys(), key=lambda m: data[m])

    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_dark_style(ax, "Cognitive Attention Score (CAS) by Model")

    y_pos = np.arange(len(sorted_models))
    scores = [data[m] for m in sorted_models]
    bars = ax.barh(y_pos, scores, height=0.55,
                   color=[COLORS[m] for m in sorted_models],
                   edgecolor="none", zorder=3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                f"{score:.2f}", va="center", ha="left",
                fontsize=13, fontweight="bold", color=TEXT_CLR)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models, fontsize=12, color=TEXT_CLR)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("CAS Score", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.5, zorder=0)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "cas_scores.png"), dpi=300, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("[OK] cas_scores.png")


# ── Figure 2: Heatmap ─────────────────────────────────────────────────────────
def fig_task_heatmap():
    cas = {name: load_cas(key) for name, key in MODEL_KEYS.items()}
    # Build matrix: rows=models, cols=tasks
    matrix = np.zeros((len(MODEL_ORDER), len(TASK_ORDER)))
    for i, model in enumerate(MODEL_ORDER):
        for j, task in enumerate(TASK_ORDER):
            matrix[i, j] = cas[model]["per_task"][task]["mean"]

    fig, ax = plt.subplots(figsize=(10, 4))
    _apply_dark_style(ax, "Per-Task Performance Across Models")

    # Custom red-to-green colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rg", ["#d73027", "#fee08b", "#1a9850"])
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], fontsize=10, color=TEXT_CLR,
                       rotation=30, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=11, color=TEXT_CLR)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_CLR)
    cbar.outline.set_edgecolor(GRID_CLR)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "task_heatmap.png"), dpi=300, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("[OK] task_heatmap.png")


# ── Figure 3: Difficulty Degradation Curves ────────────────────────────────────
def fig_difficulty_curves():
    cas = {name: load_cas(key) for name, key in MODEL_KEYS.items()}
    tasks_to_plot = ["sustained", "shifting", "stroop", "interference"]
    task_titles = {"sustained": "Sustained Attention", "shifting": "Attention Shifting",
                   "stroop": "Stroop Interference", "interference": "Proactive Interference"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.set_facecolor(BG)
    fig.suptitle("Difficulty Degradation Curves", fontsize=15, fontweight="bold",
                 color=TEXT_CLR, y=0.98)

    for idx, (ax, task) in enumerate(zip(axes.flat, tasks_to_plot)):
        _apply_dark_style(ax, task_titles[task])
        x = np.arange(len(DIFF_ORDER))

        for model in MODEL_ORDER:
            by_diff = cas[model]["per_task"][task]["by_difficulty"]
            values = [by_diff[d] for d in DIFF_ORDER]
            ax.plot(x, values, marker="o", markersize=6, linewidth=2.2,
                    color=COLORS[model], label=model, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(DIFF_ORDER, fontsize=9, color=TEXT_CLR)
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("Score", fontsize=10, color=TEXT_CLR)
        ax.grid(color=GRID_CLR, linewidth=0.5, zorder=0)

        if idx == 0:
            leg = ax.legend(fontsize=8, loc="lower left", facecolor=BG,
                           edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES, "difficulty_curves.png"), dpi=300, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("[OK] difficulty_curves.png")


# ── Figure 4: Radar / Cognitive Profile (with Human Baseline + Gemini) ────────
def fig_cognitive_profile():
    cas = {name: load_cas(key) for name, key in MODEL_KEYS.items()}

    abilities = ["Capacity", "Sustained", "Selective", "Shifting", "Stimulus-Driven"]

    def compute_axes(c):
        pt = c["per_task"]
        return [
            np.mean([pt["capacity"]["mean"], pt["interference"]["mean"]]),    # Capacity
            np.mean([pt["sustained"]["mean"], pt["stream_segregation"]["mean"]]),  # Sustained
            np.mean([pt["selective"]["mean"], pt["stroop"]["mean"]]),          # Selective
            pt["shifting"]["mean"],                                            # Shifting
            pt["anomaly"]["mean"],                                             # Stimulus-Driven
        ]

    # Human baseline data (n=25, from human_baseline/results/human_baselines.json)
    human_baseline_path = os.path.join(ROOT, "human_baseline", "results", "human_baselines.json")
    human_vals = None
    if os.path.exists(human_baseline_path):
        with open(human_baseline_path) as f:
            hb = json.load(f)
        hp = hb["per_task"]
        human_vals = [
            np.mean([hp["capacity"]["mean"], hp["interference"]["mean"]]),     # Capacity
            np.mean([hp["sustained"]["mean"], hp["stream_segregation"]["mean"]]),  # Sustained
            np.mean([hp["selective"]["mean"], hp["stroop"]["mean"]]),           # Selective
            hp["shifting"]["mean"],                                             # Shifting
            hp["anomaly"]["mean"],                                              # Stimulus-Driven
        ]

    # Gemini 2.5 Flash (from Kaggle benchmark runs: 440 items)
    gemini_vals = [
        0.992,   # Capacity: 119/120
        0.938,   # Sustained: 75/80
        0.992,   # Selective: 119/120
        1.000,   # Shifting: 80/80
        0.575,   # Stimulus-Driven: 23/40
    ]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.set_facecolor(BG)
    ax.set_facecolor(BG)

    N = len(abilities)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(abilities, fontsize=12, fontweight="bold", color=TEXT_CLR)

    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color=TEXT_CLR)
    ax.yaxis.grid(color=GRID_CLR, linewidth=0.5)
    ax.xaxis.grid(color=GRID_CLR, linewidth=0.5)
    ax.spines["polar"].set_color(GRID_CLR)

    # Plot human baseline first (dashed white line, prominent)
    if human_vals is not None:
        hv = human_vals + human_vals[:1]
        ax.plot(angles, hv, linewidth=2.5, color="#ffffff", label="Human (n=25)",
                linestyle="--", zorder=5, marker="s", markersize=5)
        ax.fill(angles, hv, alpha=0.06, color="#ffffff")

    # Plot Gemini 2.5 Flash
    gv = gemini_vals + gemini_vals[:1]
    ax.plot(angles, gv, linewidth=2.2, color="#e84855", label="Gemini 2.5 Flash",
            zorder=4, marker="^", markersize=5)
    ax.fill(angles, gv, alpha=0.10, color="#e84855")

    # Plot local models
    for model in MODEL_ORDER:
        vals = compute_axes(cas[model])
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2.0, color=COLORS[model], label=model, zorder=3)
        ax.fill(angles, vals, alpha=0.08, color=COLORS[model])

    ax.set_title("Cognitive Attention Profile: Models vs Human Baseline",
                 fontsize=14, fontweight="bold", color=TEXT_CLR, pad=24)
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.08),
                    fontsize=9, facecolor=BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    fig.savefig(os.path.join(FIGURES, "cognitive_profile.png"), dpi=300, bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("[OK] cognitive_profile.png (with human baseline + Gemini 2.5 Flash)")


# ── Figure 5: Degradation Power-Law (log-log scatter + fit) ──────────────────
def fig_degradation_power_law(degradation_data=None):
    """Scatter + fit line: noise_ratio vs error_rate on log-log scale.

    Args:
        degradation_data: Dict keyed by model name, each containing:
            noise_ratios: list of floats
            error_rates: list of floats
            delta: float (power-law exponent)
            a: float (intercept)
        If None, generates a demo with synthetic data.
    """
    if degradation_data is None:
        # Demo data
        degradation_data = {}
        for model, delta, a in [("Qwen2.5-72B", 1.2, 0.05), ("Llama3.1-8B", 1.8, 0.10), ("Phi-3-mini", 2.5, 0.15)]:
            nrs = [0.1, 0.2, 0.3, 0.5, 0.7]
            ers = [a * (nr ** delta) for nr in nrs]
            degradation_data[model] = {"noise_ratios": nrs, "error_rates": ers, "delta": delta, "a": a}

    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_dark_style(ax, "Selective Attention Degradation (Power-Law Fit)")

    for model in MODEL_ORDER:
        if model not in degradation_data:
            continue
        d = degradation_data[model]
        nrs = np.array(d["noise_ratios"])
        ers = np.array(d["error_rates"])

        ax.scatter(nrs, ers, color=COLORS[model], s=60, zorder=4, label=f"{model} (δ={d['delta']:.2f})")

        # Fit line
        x_fit = np.linspace(nrs.min(), nrs.max(), 100)
        y_fit = d["a"] * (x_fit ** d["delta"])
        ax.plot(x_fit, y_fit, color=COLORS[model], linewidth=1.8, alpha=0.7, linestyle="--", zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noise Ratio", fontsize=11)
    ax.set_ylabel("Error Rate", fontsize=11)
    ax.grid(color=GRID_CLR, linewidth=0.5, zorder=0)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "degradation_power_law.png"), dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] degradation_power_law.png")


# ── Figure 6: Arithmetic vs Geometric CAS ────────────────────────────────────
def fig_arithmetic_vs_geometric(cas_data=None):
    """Grouped bar chart comparing arithmetic and geometric CAS.

    Args:
        cas_data: Dict keyed by model name, each with 'cas_score' and 'cas_geometric'.
            If None, uses demo data.
    """
    if cas_data is None:
        cas_data = {
            "Qwen2.5-72B": {"cas_score": 0.81, "cas_geometric": 0.78},
            "Llama3.1-8B": {"cas_score": 0.73, "cas_geometric": 0.65},
            "Phi-3-mini": {"cas_score": 0.57, "cas_geometric": 0.42},
        }

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_style(ax, "CAS: Arithmetic vs Geometric Mean")

    models = [m for m in MODEL_ORDER if m in cas_data]
    x = np.arange(len(models))
    width = 0.32

    arith = [cas_data[m]["cas_score"] for m in models]
    geo = [cas_data[m]["cas_geometric"] for m in models]

    bars1 = ax.bar(x - width / 2, arith, width, label="Arithmetic (compensatory)",
                   color=[COLORS[m] for m in models], edgecolor="none", zorder=3)
    bars2 = ax.bar(x + width / 2, geo, width, label="Geometric (non-compensatory)",
                   color=[COLORS[m] for m in models], edgecolor="none", alpha=0.5, zorder=3,
                   hatch="//")

    for bar, val in zip(bars1, arith):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, color=TEXT_CLR)
    for bar, val in zip(bars2, geo):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, color=TEXT_CLR)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, color=TEXT_CLR)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("CAS Score", fontsize=11)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.5, zorder=0)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "arithmetic_vs_geometric.png"), dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] arithmetic_vs_geometric.png")


# ── Figure 7: Shifting Error Breakdown (Stacked Bar) ─────────────────────────
def fig_shifting_error_breakdown(error_data=None):
    """Stacked bar: perseveration vs residue vs random errors per model.

    Args:
        error_data: Dict keyed by model name, each with perseveration, residue, random counts.
            If None, uses demo data.
    """
    if error_data is None:
        error_data = {
            "Qwen2.5-72B": {"perseveration": 2, "residue": 1, "random": 1},
            "Llama3.1-8B": {"perseveration": 5, "residue": 3, "random": 2},
            "Phi-3-mini": {"perseveration": 8, "residue": 5, "random": 4},
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_style(ax, "Shifting Task: Error Type Breakdown")

    models = [m for m in MODEL_ORDER if m in error_data]
    x = np.arange(len(models))
    width = 0.5

    persev = [error_data[m]["perseveration"] for m in models]
    residue = [error_data[m]["residue"] for m in models]
    random_ = [error_data[m]["random"] for m in models]

    ax.bar(x, persev, width, label="Perseveration", color="#e74c3c", zorder=3)
    ax.bar(x, residue, width, bottom=persev, label="Attentional Residue", color="#f39c12", zorder=3)
    bottoms = [p + r for p, r in zip(persev, residue)]
    ax.bar(x, random_, width, bottom=bottoms, label="Random Error", color="#95a5a6", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, color=TEXT_CLR)
    ax.set_ylabel("Error Count", fontsize=11)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.5, zorder=0)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "shifting_error_breakdown.png"), dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] shifting_error_breakdown.png")


# ── Figure 8: Position Bias U-Curve ──────────────────────────────────────────
def fig_position_bias(bias_data=None):
    """Accuracy vs position-in-context showing U-shaped attention.

    Args:
        bias_data: Dict keyed by model name, each with 'bins' list of
            {bin_label, accuracy_mean}. If None, uses demo U-shape data.
    """
    if bias_data is None:
        labels = ["0%-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%"]
        bias_data = {
            "Qwen2.5-72B": {"bins": [{"bin_label": l, "accuracy_mean": v} for l, v in
                zip(labels, [0.92, 0.85, 0.80, 0.83, 0.90])]},
            "Llama3.1-8B": {"bins": [{"bin_label": l, "accuracy_mean": v} for l, v in
                zip(labels, [0.88, 0.75, 0.65, 0.72, 0.85])]},
            "Phi-3-mini": {"bins": [{"bin_label": l, "accuracy_mean": v} for l, v in
                zip(labels, [0.78, 0.55, 0.42, 0.50, 0.70])]},
        }

    fig, ax = plt.subplots(figsize=(8, 5))
    _apply_dark_style(ax, "Position Bias: U-Shaped Attention Curve")

    for model in MODEL_ORDER:
        if model not in bias_data:
            continue
        bins = bias_data[model]["bins"]
        labels = [b["bin_label"] for b in bins]
        vals = [b["accuracy_mean"] for b in bins]
        x = np.arange(len(vals))
        ax.plot(x, vals, marker="o", markersize=7, linewidth=2.2,
                color=COLORS[model], label=model, zorder=3)

    if bias_data:
        first_model = next(m for m in MODEL_ORDER if m in bias_data)
        labels = [b["bin_label"] for b in bias_data[first_model]["bins"]]
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, color=TEXT_CLR)

    ax.set_xlabel("Position in Context", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(color=GRID_CLR, linewidth=0.5, zorder=0)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "position_bias.png"), dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] position_bias.png")


# ── Figure 9: Selectivity Frontier (Recall vs Intrusion Trade-Off) ───────────
def fig_selectivity_frontier(selectivity_data=None):
    """Scatter plot of recall vs intrusion rate — the selectivity frontier.

    Models near top-left (high recall, low intrusion) are best.

    Args:
        selectivity_data: Dict keyed by model name, each with lists of
            'recall' and 'intrusion' values. If None, uses demo data.
    """
    if selectivity_data is None:
        np.random.seed(42)
        selectivity_data = {
            "Qwen2.5-72B": {
                "recall": list(np.clip(np.random.normal(0.92, 0.08, 20), 0, 1)),
                "intrusion": list(np.clip(np.random.normal(0.05, 0.04, 20), 0, 1)),
            },
            "Llama3.1-8B": {
                "recall": list(np.clip(np.random.normal(0.80, 0.12, 20), 0, 1)),
                "intrusion": list(np.clip(np.random.normal(0.15, 0.08, 20), 0, 1)),
            },
            "Phi-3-mini": {
                "recall": list(np.clip(np.random.normal(0.65, 0.15, 20), 0, 1)),
                "intrusion": list(np.clip(np.random.normal(0.30, 0.12, 20), 0, 1)),
            },
        }

    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_dark_style(ax, "Selectivity Frontier: Recall vs Intrusion Rate")

    for model in MODEL_ORDER:
        if model not in selectivity_data:
            continue
        d = selectivity_data[model]
        ax.scatter(d["intrusion"], d["recall"], color=COLORS[model],
                   s=50, alpha=0.7, label=model, zorder=3, edgecolors="none")
        # Mean marker
        mean_intr = np.mean(d["intrusion"])
        mean_rec = np.mean(d["recall"])
        ax.scatter([mean_intr], [mean_rec], color=COLORS[model],
                   s=150, marker="D", edgecolors="white", linewidths=1.5, zorder=4)

    # Ideal zone annotation
    ax.annotate("Ideal", xy=(0.02, 0.98), fontsize=10, color="#1a9850",
                fontweight="bold", ha="left", va="top")
    ax.annotate("Poor", xy=(0.85, 0.15), fontsize=10, color="#d73027",
                fontweight="bold", ha="right", va="bottom")

    ax.set_xlabel("Intrusion Rate (lower is better)", fontsize=11)
    ax.set_ylabel("Signal Recall (higher is better)", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(color=GRID_CLR, linewidth=0.5, zorder=0)
    ax.legend(fontsize=9, facecolor=BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    fig.tight_layout()
    _save_multi_format(fig, "selectivity_frontier")
    plt.close(fig)
    print("[OK] selectivity_frontier.png (+.pdf, .eps, .jpg)")


# ── Multi-format export helper ────────────────────────────────────────────────
EXPORT_FORMATS = [".png", ".pdf", ".eps", ".jpg"]


def _save_multi_format(fig, name):
    """Save figure in all 4 formats: PNG, PDF, EPS, JPG."""
    for ext in EXPORT_FORMATS:
        path = os.path.join(FIGURES, f"{name}{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)


def export_all_figures_multi_format():
    """Re-export all existing PNG figures to PDF, EPS, and JPG.

    Call after all figures have been generated to produce multi-format copies.
    """
    import glob
    png_files = glob.glob(os.path.join(FIGURES, "*.png"))
    count = 0
    for png_path in png_files:
        basename = os.path.splitext(os.path.basename(png_path))[0]
        # Re-read and re-save in other formats
        from PIL import Image
        try:
            img = Image.open(png_path)
            # JPG
            jpg_path = os.path.join(FIGURES, f"{basename}.jpg")
            rgb_img = img.convert("RGB")
            rgb_img.save(jpg_path, quality=95)
            count += 1
        except Exception:
            pass  # PIL not available or image issue
    # PDF and EPS require matplotlib re-render, skip for existing PNGs
    # New figures use _save_multi_format directly
    print(f"[OK] Exported {count} figures to JPG format")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Reading results from: {RESULTS}")
    print(f"Writing figures to:   {FIGURES}\n")
    fig_cas_scores()
    fig_task_heatmap()
    fig_difficulty_curves()
    fig_cognitive_profile()
    fig_degradation_power_law()
    fig_arithmetic_vs_geometric()
    fig_shifting_error_breakdown()
    fig_position_bias()
    fig_selectivity_frontier()
    print(f"\nDone — 9 figures saved to {FIGURES}/")
