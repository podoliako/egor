"""Compare velocity and hypocentre gains for the existing discussion points."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_soft_gain_vs_top_n import SERIES, sha

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "soft_gain_vs_top_n_lcurve.json"
STEM = "soft_velocity_gain_vs_hypocenter_gain"


def main():
    source = json.loads(SOURCE.read_text())
    points = source["points"]
    x = np.asarray([p["p90_gain_percent"] for p in points], dtype=float)
    y = np.asarray([p["soft_gain_percent"] for p in points], dtype=float)
    assert len({p["pair_id"] for p in points}) == len(points)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    for p, px, py in zip(points, x, y):
        assert np.isclose(px, 100 * (p["hard_p90_m"] - p["soft_p90_m"]) / p["hard_p90_m"])
        assert np.isclose(py, 100 * (p["hard_avg_abs_pct_dev"] - p["soft_avg_abs_pct_dev"]) / p["hard_avg_abs_pct_dev"])
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.axhline(0, color="0.4", linewidth=.9)
        ax.axvline(0, color="0.4", linewidth=.9)
        for section, (label, color) in SERIES.items():
            mask = np.asarray([p["section"] == section for p in points])
            ax.scatter(x[mask], y[mask], s=45, color=color, alpha=.9,
                       edgecolors="white", linewidths=.7, zorder=3,
                       label=f"{label} ({mask.sum()})")
        ax.set_xlabel("Soft gain in hypocentre P90 (%)")
        ax.set_ylabel("Soft gain in velocity error (%)")
        ax.grid(color="0.9", linewidth=.6)
        ax.set_axisbelow(True)
        ax.margins(.09)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3,
                  frameon=False, fontsize=8, handletextpad=.3, columnspacing=1.2,
                  borderaxespad=0)
        fig.savefig(HERE.parent / "png" / f"{STEM}.png", dpi=220, bbox_inches="tight")
        fig.savefig(HERE / f"{STEM}.pdf", bbox_inches="tight")
        plt.close(fig)
    report = dict(source=SOURCE.name, source_sha256=sha(SOURCE),
                  n_points=len(points), x_metric=source["p90_metric"],
                  y_metric=source["metric"], checkpoint_policy=source["checkpoint_policy"],
                  p90_timing=source["p90_timing"],
                  coordinate_policy="Original gains, linear axes, no jitter or offsets",
                  points=points)
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated and plotted {len(points)} pairs")
    print(f"Hypocentre gain range: {x.min():.3f} to {x.max():.3f}%")
    print(f"Velocity gain range: {y.min():.3f} to {y.max():.3f}%")


if __name__ == "__main__":
    main()
