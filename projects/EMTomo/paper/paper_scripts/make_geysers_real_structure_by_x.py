"""Split the real-data Guo--Thurber (2021) Vp comparison along model X.

python3 -B paper/paper_scripts/make_geysers_real_structure_by_x.py [--spacing 125]

Same inputs, shared resolvability mask (R > 0.4), sampling and metric definitions as
make_geysers_real_structure_metrics.py; only the masked samples are grouped by X.
Adds lateral-anomaly metrics: Vp minus its mean over the masked samples of the same depth
level within each group, so the shared depth gradient does not dominate the correlation.
Writes paper/exp_data/geysers_real_structure_by_x.json. Descriptive, not truth error.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_geysers_real_structure_metrics as base  # noqa: E402

OUTPUT = HERE.parent / "exp_data/geysers_real_structure_by_x.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=125.0)
    parser.add_argument("--bin-km", type=float, default=2.0)
    args = parser.parse_args()
    models, published, geometry, _, hashes = base.load_inputs()
    points, _ = base.midpoint_grid(geometry["domain_extent_m"], args.spacing)
    values = np.empty((len(points), 2))
    for start in range(0, len(points), 65536):
        values[start:start + 65536] = published.sample(points[start:start + 65536])
    mask = base.published_mask(values)
    pts = points[mask]
    reference = values[mask, 0] * 1000
    sampled = {name: base.sample_cells(array, pts, cell_km) * 1000 for name, (array, cell_km) in models.items()}
    x = pts[:, 0]
    x_mid = float(0.5 * (x.min() + x.max()))
    x_median = float(np.median(x))

    z = pts[:, 2]

    def lateral(values, selection):
        """Remove the mean of each depth level within the selection (lateral anomalies)."""
        out = values[selection].astype(float).copy()
        zs = z[selection]
        for level in np.unique(zs):
            at = zs == level
            out[at] -= out[at].mean()
        return out

    def anomaly_corr(model, selection):
        a, b = lateral(model, selection), lateral(reference, selection)
        den = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / den) if den > 0 else None

    def group(selection):
        metrics = {n: base.descriptive_metrics(v[selection], reference[selection]) for n, v in sampled.items()}
        for n, v in sampled.items():
            metrics[n]["pearson_lateral_anomaly"] = anomaly_corr(v, selection)
            metrics[n]["rms_lateral_anomaly_difference_m_s"] = float(
                np.sqrt(np.mean((lateral(v, selection) - lateral(reference, selection)) ** 2)))
        return dict(samples=int(selection.sum()),
                    x_range_km=[float(x[selection].min()), float(x[selection].max())], metrics=metrics)

    edges = np.arange(np.floor(x.min() / args.bin_km) * args.bin_km, x.max() + args.bin_km, args.bin_km)
    bins = [dict(x_km=[float(a), float(b)], **group((x >= a) & (x < b)))
            for a, b in zip(edges[:-1], edges[1:]) if np.any((x >= a) & (x < b))]
    report = dict(
        description=__doc__, spacing_m=args.spacing, mask="finite published Vp and R > 0.4 (shared)",
        difference_convention="EMTomo model Vp minus published Vp, m/s",
        masked_samples=int(mask.sum()), masked_x_range_km=[float(x.min()), float(x.max())],
        split_x_km=x_mid, split_rule="midpoint of the masked X range", masked_x_median_km=x_median,
        whole=group(np.ones_like(x, dtype=bool)), left=group(x < x_mid), right=group(x >= x_mid),
        bins=bins, source_sha256=hashes, script_sha256=base.sha(Path(__file__)))
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name in ("whole", "left", "right"):
        g = report[name]
        print(name, g["x_range_km"], g["samples"],
              {m: {k: (round(v, 3) if v is not None else None) for k, v in r.items()} for m, r in g["metrics"].items()})
    for b in bins:
        print(b["x_km"], b["samples"], {m: (round(r["rms_difference_m_s"]), round(r["mean_signed_difference_m_s"]),
                                            round(r["pearson_vp"], 2) if r["pearson_vp"] is not None else None,
                                            None if r["pearson_lateral_anomaly"] is None else round(r["pearson_lateral_anomaly"], 2),
                                            round(r["rms_lateral_anomaly_difference_m_s"]))
                                        for m, r in b["metrics"].items()})


if __name__ == "__main__":
    main()
