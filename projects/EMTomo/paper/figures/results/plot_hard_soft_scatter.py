"""Hard/soft residual minima for the same comparison rows and series palette."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from plot_soft_gain_distribution import SERIES, SOURCE

HERE = Path(__file__).resolve().parent


def main():
    with SOURCE.open() as stream:
        rows = list(csv.DictReader(stream))
    hard = np.asarray([float(r["hard_min_residual_rmse_s"]) * 1000 for r in rows])
    soft = np.asarray([float(r["soft_min_residual_rmse_s"]) * 1000 for r in rows])
    if not len(rows) or not np.all(np.isfinite(hard)) or not np.all(np.isfinite(soft)):
        raise ValueError("Missing or nonfinite residual metrics")
    if np.any(hard <= 0) or np.any(soft < 0):
        raise ValueError("Invalid residual metrics")
    unknown = {r["section"] for r in rows} - SERIES.keys()
    if unknown:
        raise ValueError(f"Unknown experiment series: {unknown}")
    limit = np.ceil(max(hard.max(), soft.max()) * 1.07 / 10) * 10
    wins = int(np.count_nonzero(soft < hard))
    losses = int(np.count_nonzero(soft > hard))
    ties = int(np.count_nonzero(soft == hard))
    zoom_limit = 25.
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(8.4, 8.2))
        inset = ax.inset_axes([.57, .08, .38, .38])
        for axes, maximum in ((ax, limit), (inset, zoom_limit)):
            axes.plot([0, maximum], [0, maximum], "--", color="#555555", linewidth=1.2, zorder=2)
            axes.set_xlim(0, maximum)
            axes.set_ylim(0, maximum)
            axes.set_aspect("equal", adjustable="box")
            axes.set_axisbelow(True)
            axes.grid(color="#dddddd", linewidth=.55)
        for section, (label, color) in SERIES.items():
            mask = np.asarray([r["section"] == section for r in rows])
            ax.scatter(hard[mask], soft[mask], s=57, c=color, alpha=.85,
                       edgecolors="white", linewidths=.6, zorder=4,
                       label=f"{label} (n={int(mask.sum())})")
            inset.scatter(hard[mask], soft[mask], s=35, c=color, alpha=.85,
                          edgecolors="white", linewidths=.45, zorder=4)
        ax.add_patch(Rectangle((0, 0), zoom_limit, zoom_limit, facecolor="none",
                               edgecolor="#999999", linewidth=.9, linestyle=":", zorder=3))
        inset.set_title("Low-RMSE detail (ms)", fontsize=9, pad=6)
        inset.set_xticks([0, 10, 20])
        inset.set_yticks([0, 10, 20])
        inset.tick_params(labelsize=8)
        ax.set_xlabel("Hard minimum residual RMSE (ms)", labelpad=9)
        ax.set_ylabel("Soft minimum residual RMSE (ms)", labelpad=9)
        ax.set_title("Hard vs soft: minimum residual RMSE", loc="left", pad=15)
        ax.legend(loc="upper left", frameon=False, fontsize=9.2, labelspacing=.8,
                  handletextpad=.6)
        ax.text(.43, .79, "Above diagonal:\nhard better", transform=ax.transAxes,
                color="#555555", fontsize=10, ha="center", va="center")
        ax.text(.96, .035, "Below diagonal: soft better", transform=ax.transAxes,
                color="#555555", fontsize=9.5, ha="right")
        fig.text(.52, .092,
                 f"N = {len(rows)} comparisons  |  Soft better: {wins}  |  Hard better: {losses}"
                 + (f"  |  Ties: {ties}" if ties else "") + "  |  Dashed line: y = x",
                 ha="center", fontsize=10, color="#444444")
        fig.text(.52, .038,
                 "Each coordinate is the independent minimum over that run's complete iterations.\n"
                 "Points are not jittered; overlapping comparisons may coincide. Some hard controls are reused.",
                 ha="center", fontsize=9, color="#555555", linespacing=1.6)
        fig.subplots_adjust(left=.12, right=.97, top=.92, bottom=.17)
        for extension in ("png", "pdf"):
            fig.savefig(HERE / f"hard_vs_soft_residual_rmse_scatter.{extension}", dpi=220, facecolor="white")
        plt.close(fig)
    report = dict(source=str(SOURCE.relative_to(HERE.parents[3])),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  n_comparisons=len(rows), soft_better=wins, hard_better=losses, ties=ties,
                  units="ms", main_limits=[0, float(limit)], inset_limits=[0, zoom_limit],
                  points=[dict(pair_id=r["pair_id"], section=r["section"],
                               hard_min_residual_rmse_ms=float(h), soft_min_residual_rmse_ms=float(s))
                          for r, h, s in zip(rows, hard, soft)],
                  note="Same comparison rows and series colors as the gain histogram; independent per-run minima; no jitter.")
    (HERE / "hard_vs_soft_residual_rmse_scatter.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("n_comparisons", "soft_better", "hard_better", "main_limits")}, indent=2))


if __name__ == "__main__":
    main()
