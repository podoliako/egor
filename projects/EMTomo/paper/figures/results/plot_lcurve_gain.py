"""Plot truth-based evaluation AFTER training-only L-curve checkpoint selection."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from plot_soft_gain_distribution import SERIES

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
SOURCE = PAPER / "exp_data/hard_soft_lcurve_avg_abs_pct_dev.csv"


def main():
    with SOURCE.open() as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [r for r in all_rows if r["included_in_plot"] == "True"]
    excluded = [r["pair_id"] for r in all_rows if r["included_in_plot"] != "True"]
    if len(rows) != 36 or set(excluded) != {"section6_pair05", "section6_pair06"}:
        raise ValueError(f"Unexpected plot scope: n={len(rows)}, excluded={excluded}")
    hard = np.asarray([float(r["hard_avg_abs_pct_dev_at_lcurve"]) for r in rows])
    soft = np.asarray([float(r["soft_avg_abs_pct_dev_at_lcurve"]) for r in rows])
    gain = np.asarray([float(r["soft_gain_percent"]) for r in rows])
    if not np.all(np.isfinite(hard)) or not np.all(np.isfinite(soft)) or np.any(hard <= 0):
        raise ValueError("Invalid selected metrics")
    np.testing.assert_allclose(gain, 100 * (hard - soft) / hard, atol=1e-12)
    noise_levels = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        if row["section"] != "4":
            continue
        path = PAPER.parents[1] / "forward_modeling/experiments/output" / row["experiment_id"] / "metadata.json"
        config = json.loads(path.read_text())["noise"]["config"]
        multiplier = config["relative_sigma"] / .01
        if not np.isclose(multiplier, config["absolute_sigma_s"] / .05):
            raise ValueError(f"Noise components have unequal scaling: {row['pair_id']}")
        noise_levels[i] = multiplier
    cmap = LinearSegmentedColormap.from_list("actual_noise", plt.get_cmap("Greens")(np.linspace(.38, .92, 256)))
    norm = Normalize(vmin=.5, vmax=2.)
    x_min = float(max(0, np.floor(hard.min() * .95 / .5) * .5))
    x_max = float(np.ceil(hard.max() * 1.05 / .5) * .5)
    y_min = float(min(-2, np.floor(gain.min() / 2) * 2))
    y_max = float(max(2, np.ceil(gain.max() / 2) * 2))
    wins = int(np.count_nonzero(gain > 0))
    losses = int(np.count_nonzero(gain < 0))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(9., 5.8))
        ax.axhline(0, color="#444444", linestyle="--", linewidth=1.2, zorder=2)
        for section, (label, color) in SERIES.items():
            mask = np.asarray([r["section"] == section for r in rows])
            if section == "4":
                noise_scatter = ax.scatter(hard[mask], gain[mask], c=noise_levels[mask], cmap=cmap,
                                           norm=norm, s=58, edgecolors="white", linewidths=.65, zorder=3,
                                           label=f"{label} (n={int(mask.sum())})")
            else:
                ax.scatter(hard[mask], gain[mask], c=color, s=58, alpha=.87,
                           edgecolors="white", linewidths=.65, zorder=3,
                           label=f"{label} (n={int(mask.sum())})")
        bar_axes = ax.inset_axes([.52, .55, .43, .027])
        bar = fig.colorbar(noise_scatter, cax=bar_axes, orientation="horizontal")
        bar.set_ticks([.5, .75, 1., 1.5, 2.], labels=["0.5×", "0.75×", "1×", "1.5×", "2×"])
        bar.ax.tick_params(labelsize=8)
        bar.ax.set_title("Actual pick-noise multiplier (darker = noisier)", fontsize=9, pad=8)
        bar.outline.set_linewidth(.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_axisbelow(True)
        ax.grid(color="#dddddd", linewidth=.6)
        ax.set_xlabel("Hard average absolute velocity deviation at L-curve selection (%)", labelpad=10)
        ax.set_ylabel("Soft gain in velocity deviation (%)", labelpad=10)
        ax.set_title("Soft gain versus hard error: L-curve-selected models", loc="left", pad=31)
        ax.text(0, 1.025, r"Viewer metric: avg_abs_pct_dev; gain = $100\,(E_{hard,L}-E_{soft,L})/E_{hard,L}$",
                transform=ax.transAxes, fontsize=10, color="#444444")
        ax.text(.98, .96, "Soft better", transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color="#555555")
        ax.legend(loc="lower right", frameon=False, fontsize=9.1, labelspacing=.75)
        fig.text(.53, .13,
                 f"N = {len(rows)} comparisons  |  Soft better: {wins}  |  Hard better: {losses}"
                 "  |  Dashed line: no difference",
                 ha="center", fontsize=9.5, color="#444444")
        fig.text(.53, .025,
                 "Iteration selection uses training residuals and slowness change, not the true velocity model.\n"
                 "Plotted errors use truth. Two complex-anomaly pairs with homogeneous starts are excluded.\n"
                 "No clipping or jitter; some hard controls are reused.",
                 ha="center", fontsize=8.8, color="#555555", linespacing=1.5)
        fig.subplots_adjust(left=.12, right=.98, top=.81, bottom=.26)
        for extension in ("png", "pdf"):
            fig.savefig(HERE / f"soft_gain_vs_hard_avg_abs_pct_dev_lcurve.{extension}", dpi=220, facecolor="white")
        plt.close(fig)
    report = dict(source=str(SOURCE.relative_to(PAPER)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  n_csv_pairs=len(all_rows), n_plotted_pairs=len(rows), excluded_pair_ids=excluded,
                  selection_uses_truth=False, evaluation_uses_truth=True,
                  soft_better=wins, hard_better=losses, x_limits=[x_min, x_max], y_limits=[y_min, y_max],
                  points=[dict(pair_id=r["pair_id"], section=r["section"],
                               hard_avg_abs_pct_dev=float(x), soft_avg_abs_pct_dev=float(s),
                               soft_gain_percent=float(y),
                               hard_selected_update=int(r["hard_selected_update_number"]),
                               soft_selected_update=int(r["soft_selected_update_number"]),
                               actual_noise_multiplier=float(level) if np.isfinite(level) else None)
                          for r, x, s, y, level in zip(rows, hard, soft, gain, noise_levels)])
    (HERE / "soft_gain_vs_hard_avg_abs_pct_dev_lcurve.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_csv_pairs", "n_plotted_pairs", "excluded_pair_ids", "soft_better", "hard_better")}, indent=2))


if __name__ == "__main__":
    main()
