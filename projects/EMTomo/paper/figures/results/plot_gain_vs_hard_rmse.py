"""Minimum hard residual RMSE versus relative soft gain, for every analyzed pair."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_soft_gain_distribution import SERIES, SOURCE

HERE = Path(__file__).resolve().parent


def main():
    with SOURCE.open() as stream:
        rows = list(csv.DictReader(stream))
    hard = np.asarray([float(r["hard_min_residual_rmse_s"]) for r in rows])
    soft = np.asarray([float(r["soft_min_residual_rmse_s"]) for r in rows])
    if not len(rows) or not np.all(np.isfinite(hard)) or not np.all(np.isfinite(soft)):
        raise ValueError("Empty or nonfinite residual metrics")
    if np.any(hard <= 0) or np.any(soft < 0):
        raise ValueError("Invalid residual metrics")
    gain = 100 * (hard - soft) / hard
    np.testing.assert_allclose(gain, [float(r["soft_gain_percent"]) for r in rows], rtol=1e-12, atol=1e-12)
    unknown = {r["section"] for r in rows} - SERIES.keys()
    if unknown:
        raise ValueError(f"Unknown experiment series: {unknown}")
    hard_ms = hard * 1000
    wins = int(np.count_nonzero(gain > 0))
    losses = int(np.count_nonzero(gain < 0))
    x_max = float(np.ceil(hard_ms.max() * 1.07 / 10) * 10)
    y_min = float(min(-5, np.floor(gain.min() / 5) * 5))
    y_max = float(max(2, np.ceil(gain.max() / 5) * 5))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(8.8, 5.7))
        ax.axhline(0, color="#444444", linestyle="--", linewidth=1.2, zorder=2)
        for section, (label, color) in SERIES.items():
            mask = np.asarray([row["section"] == section for row in rows])
            ax.scatter(hard_ms[mask], gain[mask], s=58, c=color, alpha=.87,
                       edgecolors="white", linewidths=.65, zorder=3,
                       label=f"{label} (n={int(mask.sum())})")
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_axisbelow(True)
        ax.grid(color="#dddddd", linewidth=.6)
        ax.set_xlabel("Hard minimum residual RMSE (ms)", labelpad=10)
        ax.set_ylabel("Soft gain in minimum residual RMSE (%)", labelpad=10)
        ax.set_title("Soft gain versus hard residual RMSE", loc="left", pad=31)
        ax.text(0, 1.025, r"Gain = $100\,(\mathrm{RMSE}_{hard,min}-\mathrm{RMSE}_{soft,min})/\mathrm{RMSE}_{hard,min}$",
                transform=ax.transAxes, fontsize=10.5, color="#444444")
        ax.text(.98, .96, "Soft better", transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color="#555555")
        ax.legend(loc="lower right", frameon=False, fontsize=9.2, labelspacing=.75)
        fig.text(.53, .092,
                 f"N = {len(rows)} comparisons  |  Soft better: {wins}  |  Hard better: {losses}"
                 "  |  Dashed line: no difference",
                 ha="center", fontsize=9.5, color="#444444")
        fig.text(.53, .035,
                 "Independent per-run minima; all points shown without clipping or jitter.\n"
                 "Negative gain: hard better. Some hard controls are reused.",
                 ha="center", fontsize=9, color="#555555", linespacing=1.6)
        fig.subplots_adjust(left=.12, right=.98, top=.81, bottom=.24)
        for extension in ("png", "pdf"):
            fig.savefig(HERE / f"soft_gain_vs_hard_rmse.{extension}", dpi=220, facecolor="white")
        plt.close(fig)
    report = dict(source=str(SOURCE.relative_to(HERE.parents[3])),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  n_comparisons=len(rows), soft_better=wins, hard_better=losses,
                  x_units="ms", y_units="percent", x_limits=[0, x_max], y_limits=[y_min, y_max],
                  gain_formula="100 * (hard_min_rmse - soft_min_rmse) / hard_min_rmse",
                  points=[dict(pair_id=row["pair_id"], section=row["section"],
                               hard_min_residual_rmse_ms=float(x), soft_gain_percent=float(y))
                          for row, x, y in zip(rows, hard_ms, gain)],
                  note="All 38 comparison rows; no clipping or jitter; reused hard controls are retained.")
    (HERE / "soft_gain_vs_hard_rmse.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("n_comparisons", "soft_better", "hard_better", "x_limits", "y_limits")}, indent=2))


if __name__ == "__main__":
    main()
