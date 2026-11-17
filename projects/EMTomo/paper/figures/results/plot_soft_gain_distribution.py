"""Stacked comparison distribution with an explicitly categorical left overflow bin."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "exp_data/hard_soft_min_residual_rmse.csv"
SERIES = {
    "2": ("Checkerboard: one-sided stations", "#4477AA"),
    "3": ("Checkerboard: two station clusters", "#EE9944"),
    "4": ("Checkerboard: real geometry / noise", "#228833"),
    "5": ("Vertical fault: real geometry", "#AA3377"),
    "6": ("Complex anomalies: real geometry", "#66CCEE"),
}
OVERFLOW_THRESHOLD = -15.


def main():
    with SOURCE.open() as stream:
        rows = list(csv.DictReader(stream))
    gains = []
    for row in rows:
        hard = float(row["hard_min_residual_rmse_s"])
        soft = float(row["soft_min_residual_rmse_s"])
        if not np.isfinite(hard) or not np.isfinite(soft) or hard <= 0:
            raise ValueError(f"Invalid residual metric: {row['pair_id']}")
        gain = 100 * (hard - soft) / hard
        if not np.isclose(gain, float(row["soft_gain_percent"]), rtol=1e-12, atol=1e-12):
            raise ValueError(f"Gain column mismatch: {row['pair_id']}")
        if row["section"] not in SERIES:
            raise ValueError(f"Unknown experiment series: {row['section']}")
        gains.append(gain)
    gains = np.asarray(gains)
    if not len(gains):
        raise ValueError("No comparison rows")
    # Regular one-percentage-point bins have an edge at zero. The overflow
    # column is separate and categorical: its width does not represent the tail.
    high = max(1., np.floor(gains.max()) + 1.)
    bins = np.arange(OVERFLOW_THRESHOLD, high + .5, 1.)
    overflow_x = OVERFLOW_THRESHOLD - 1.5
    centers = np.concatenate(([overflow_x], (bins[:-1] + bins[1:]) / 2))
    histograms = {}
    for section in SERIES:
        values = gains[[r["section"] == section for r in rows]]
        tail = int(np.count_nonzero(values < OVERFLOW_THRESHOLD))
        regular, _ = np.histogram(values[values >= OVERFLOW_THRESHOLD], bins=bins)
        histograms[section] = np.concatenate(([tail], regular))
    counts = np.sum(list(histograms.values()), axis=0)
    assert int(counts.sum()) == len(rows)
    median = float(np.median(gains))
    wins = int(np.count_nonzero(gains > 0))
    losses = int(np.count_nonzero(gains < 0))
    ties = int(np.count_nonzero(gains == 0))
    overflow_n = int(np.count_nonzero(gains < OVERFLOW_THRESHOLD))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(9., 5.4))
        bottom = np.zeros(len(centers), dtype=int)
        for section, (label, color) in SERIES.items():
            height = histograms[section]
            ax.bar(centers, height, bottom=bottom, width=.92, color=color,
                   edgecolor="white", linewidth=.6, zorder=3,
                   label=f"{label} (n={int(height.sum())})")
            bottom += height
        ax.axvline(0., color="#303030", linewidth=1.1, linestyle="--", zorder=4)
        ax.axvline(median, color="#555555", linewidth=1.1, linestyle=":", zorder=4)
        # Isolate the open-ended category visually rather than silently clipping
        # tail values into an ordinary numerical histogram bin.
        ax.axvspan(OVERFLOW_THRESHOLD - 1., OVERFLOW_THRESHOLD, color="#eeeeee", zorder=1)
        ax.text(overflow_x, counts[0] + .25, f"n={overflow_n}", ha="center", fontsize=9, color="#555555")
        ticks = [overflow_x, -15., -12., -9., -6., -3., 0.]
        labels = ["< −15", "−15", "−12", "−9", "−6", "−3", "0"]
        if high > 2:
            ticks.append(high - 1)
            labels.append(f"{high - 1:g}")
        ax.set_xticks(ticks, labels)
        ax.set_xlim(overflow_x - .8, high + .3)
        ax.set_ylim(0, max(counts) * 1.22)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
        ax.set_xlabel("Soft gain in minimum residual RMSE (%)", labelpad=10)
        ax.set_ylabel("Number of comparisons")
        ax.set_title("Distribution of soft gain relative to hard", loc="left", pad=33)
        ax.text(0, 1.025, r"$100\,(\mathrm{RMSE}_{hard,min}-\mathrm{RMSE}_{soft,min})/\mathrm{RMSE}_{hard,min}$",
                transform=ax.transAxes, fontsize=11, color="#444444")
        ax.legend(loc="upper left", frameon=False, fontsize=9.4, labelspacing=.65)
        fig.text(.52, .095,
                 f"N = {len(gains)} comparisons  |  Soft better: {wins}/{len(gains)} ({100*wins/len(gains):.1f}%)"
                 f"  |  Median gain: {median:.2f}% (dotted line)",
                 ha="center", fontsize=9.5, color="#444444")
        fig.text(.52, .035,
                 "Negative: hard better; positive: soft better. Leftmost column pools all gains < −15%.\n"
                 "Descriptive comparison counts; some hard controls are reused.",
                 ha="center", fontsize=9, color="#555555", linespacing=1.6)
        fig.subplots_adjust(left=.085, right=.98, bottom=.24, top=.81)
        for stem in ("soft_gain_distribution", "soft_gain_distribution_stacked"):
            for extension in ("png", "pdf"):
                fig.savefig(HERE / f"{stem}.{extension}", dpi=220, facecolor="white")
        plt.close(fig)
    report = dict(source=str(SOURCE.relative_to(HERE.parents[3])),
                  source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  n_comparisons=len(gains), soft_wins=wins, hard_wins=losses, ties=ties,
                  median_gain_percent=median, mean_gain_percent=float(gains.mean()),
                  min_gain_percent=float(gains.min()), max_gain_percent=float(gains.max()),
                  overflow_threshold_percent=OVERFLOW_THRESHOLD, overflow_rule="gain < -15",
                  overflow_count=overflow_n, overflow_display_position=overflow_x,
                  regular_histogram_edges_percent=bins.tolist(),
                  stacked_series={section: dict(label=SERIES[section][0], color=SERIES[section][1],
                                                counts_overflow_then_regular=values.astype(int).tolist())
                                  for section, values in histograms.items()},
                  comparisons=[dict(pair_id=r["pair_id"], section=r["section"], gain_percent=float(g))
                               for r, g in zip(rows, gains)],
                  note="Tail is pooled into an open-ended category; statistics use original gains. Comparison rows are not independent replications.")
    (HERE / "soft_gain_distribution.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("n_comparisons", "soft_wins", "hard_wins", "median_gain_percent", "overflow_count")}, indent=2))


if __name__ == "__main__":
    main()
