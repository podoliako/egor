"""Markdown summary of EMTomo runs on a dipping-layers experiment.

Usage (from projects/EMTomo):
    python studies/dipwedge/report.py EXPERIMENT_ID RUN_NAME_PREFIX [EXPERIMENTS_ROOT RUNS_ROOT]
Truth is used here only for evaluation.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_data import DEFAULT_EXPERIMENTS_ROOT, load_tomography_experiment  # noqa: E402
from experiment_runner import prepare_inversion  # noqa: E402
from main import CONFIG  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "runs"


def _rms(values) -> float:
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")


class Truth:
    def __init__(self, experiment_id, experiments_root, cell_m):
        generation = json.loads((Path(experiments_root) / "input" / experiment_id / "generation.json").read_text())
        params = generation["parameters"]
        self.anomalies = params["anomalies"]
        self.dip_top, self.dip_bottom = params["dipping_boundary_km"]
        self.length_x = params["lengths_km"][0]
        speeds = params["layer_velocities_m_s"]
        self.midpoint = (speeds[-2] + speeds[-1]) / 2
        self.reference = prepare_inversion(
            experiment_id, replace(CONFIG, cell_size=cell_m), experiments_root,
        ).reference_model.velocity
        self.cell_km = cell_m / 1000
        nx, ny, nz = self.reference.shape
        self.x, self.y, self.z = [(np.arange(n) + 0.5) * self.cell_km for n in (nx, ny, nz)]
        self.dip = self.dip_top + (self.dip_bottom - self.dip_top) * self.x / self.length_x
        centres = np.stack(np.meshgrid(self.x, self.y, self.z, indexing="ij"), axis=-1)
        self.cores = [np.linalg.norm(centres - a["center_km"], axis=-1) < 0.6 * a["radius_km"]
                      for a in self.anomalies]
        self.spheres = np.any([np.linalg.norm(centres - a["center_km"], axis=-1) < a["radius_km"] + self.cell_km
                               for a in self.anomalies], axis=0)

    def wedge(self, flat_km):
        """Cells whose layer differs between a flat interface at ``flat_km`` and the dipping one."""
        lo = np.minimum(self.dip, flat_km)[:, None, None]
        hi = np.maximum(self.dip, flat_km)[:, None, None]
        z = self.z[None, None, :]
        mask = (z + self.cell_km / 2 > lo) & (z - self.cell_km / 2 < hi)
        return np.broadcast_to(mask, self.reference.shape) & ~self.spheres

    def boundary_depth(self, model):
        """Depth (km) where the y-median profile crosses the last-interface midpoint velocity."""
        profile = np.median(model, axis=1)
        window = (self.z > self.dip_top - 1.5) & (self.z < self.dip_bottom + 1.5)
        ks = np.flatnonzero(window)
        out = []
        for column in profile:
            depth = np.nan
            for k in ks[:-1]:
                a, b = column[k] - self.midpoint, column[k + 1] - self.midpoint
                if a * b <= 0 and a != b:
                    depth = self.z[k] + self.cell_km * a / (a - b)
                    break
            out.append(depth)
        return np.asarray(out)


def _model_at(run_dir: Path, cycle: int, n_cycles: int) -> np.ndarray:
    return np.load(run_dir / ("final_model.npy" if cycle >= n_cycles else f"iter_{cycle}/model.npy"))


def _data_misfit(run_dir: Path, iteration: int, stride: int = 4) -> float:
    """RMS of origin-centred station residuals of the most likely hypothesis (every ``stride``-th event)."""
    values = []
    for path in sorted(run_dir.glob(f"iter_{iteration}/event_*/residuals.npy"))[::stride]:
        pairwise = np.load(path)
        if pairwise.size:
            values.append(pairwise.mean(axis=1))
    return _rms(np.concatenate(values)) if values else float("nan")


def _truth_misfit(truth: "Truth", data, stations_m, params, cache={}):
    """Misfit of the cell-averaged truth on the run's own grid, at the true hypocentres."""
    from eikonal import station_travel_time_fields
    from instruments.instruments_coords import metric_to_cell_coord, sample_cell_centered_trilinear_batch
    from velocity_model import VelocityModel

    key = (truth.cell_km, params["subdivision"], params["slowness_interpolation"])
    if key not in cache:
        fine = VelocityModel(truth.reference, truth.cell_km * 1000).refined(key[1], key[2])
        times = station_travel_time_fields(fine, stations_m, n_workers=16)
        residuals = []
        for observed, xyz in zip(data.arrival_times_s, data.reference_event_coordinates_m):
            r = observed - sample_cell_centered_trilinear_batch(times, metric_to_cell_coord(xyz, fine.cell_size))
            residuals.append(r - r.mean())
        cache[key] = _rms(np.concatenate(residuals))
    return cache[key]


def main(experiment_id, prefix, experiments_root=DEFAULT_EXPERIMENTS_ROOT, runs_root=RUNS, misfit=False):
    experiments_root, runs_root = Path(experiments_root), Path(runs_root)
    data = load_tomography_experiment(experiment_id, experiments_root)
    metadata = json.loads((experiments_root / "output" / experiment_id / "metadata.json").read_text())
    truths, rows = {}, []
    for meta_path in sorted(runs_root.glob(f"run_{prefix}*/meta.json")):
        run_dir = meta_path.parent
        meta = json.loads(meta_path.read_text())
        params = meta["run_params"]
        if (meta.get("source_experiment") or {}).get("id") != experiment_id:
            continue
        name = params["run_name"].removeprefix(prefix).removesuffix(f"_{experiment_id}")
        if not (run_dir / "final_model.npy").exists():
            rows.append({"name": name, "unfinished": True})
            continue
        cell_m = meta["grid_info"]["coarse_cell_size"]  # exact; run_params value is rounded
        truth = truths.setdefault(cell_m, Truth(experiment_id, experiments_root, cell_m))
        ref = truth.reference
        n = params["n_cycles"]
        start = np.load(run_dir / "initial_model.npy")
        quality = [json.loads(line)["rms_m_s"] for line in (run_dir / "quality.jsonl").read_text().splitlines() if line]
        curve = [_rms(start - ref)] + quality
        best = int(np.argmin(curve))
        final = np.load(run_dir / "final_model.npy")
        # Where the start model actually switches layers (cell faces), not the nominal depth.
        flat = np.nanmedian(truth.boundary_depth(start)) if np.isfinite(truth.boundary_depth(start)).any() \
            else (truth.dip_top + truth.dip_bottom) / 2
        wedge = truth.wedge(flat)
        needed = (ref - start)[wedge]

        def wedge_scores(model):
            change = (model - start)[wedge]
            gain = float(change @ needed / (needed @ needed)) if needed.any() else np.nan
            corr = float(np.corrcoef(change, needed)[0, 1]) if change.std() > 0 and needed.std() > 0 else np.nan
            return gain, corr, _rms((model - ref)[wedge])

        illuminated = truth.z[None, None, :] < data.reference_event_coordinates_m[:, 2].max() / 1000
        background = np.broadcast_to(illuminated, ref.shape) & ~truth.spheres & ~wedge
        recovery = [float(np.mean(final[c] - start[c]) / np.mean(ref[c] - start[c])) if c.any() else np.nan
                    for c in truth.cores]
        fine = meta["grid_info"]["fine_cell_size"]

        def hypo_errors(iteration):
            errors = []
            for event, true in enumerate(data.reference_event_coordinates_m):
                with np.load(run_dir / f"iter_{iteration}" / f"event_{event}" / "weights.npz") as saved:
                    position = saved["positions"][int(np.argmax(saved["weight_values"]))]
                errors.append(np.linalg.norm((position + 0.5) * fine - true))
            return np.asarray(errors)

        last = hypo_errors(n - 1)
        failed = 0
        for path in run_dir.glob("iter_*/ray*.json"):
            try:
                failed += int(json.loads(path.read_text()).get("failed", 0))
            except (ValueError, OSError):
                pass
        misfits = None
        if misfit:
            cycles = sorted({0, 1, 2, 4, 8, n - 1})
            misfits = ({c: _data_misfit(run_dir, c) for c in cycles if c < n},
                       _truth_misfit(truth, data, np.asarray(meta["station_locs"]), params))
        rows.append({
            "name": name, "curve": curve, "misfits": misfits, "best": best, "recovery": recovery,
            "wedge_best": wedge_scores(_model_at(run_dir, best, n)), "wedge_final": wedge_scores(final),
            "boundary": truth.boundary_depth(final), "boundary_best": truth.boundary_depth(_model_at(run_dir, best, n)),
            "background": (_rms((start - ref)[background]), _rms((final - ref)[background])),
            "hypo": (np.median(hypo_errors(0)), np.median(last), np.percentile(last, 90)),
            "time": json.loads((run_dir / "timing_summary.json").read_text())["mean_iter_s"],
            "failed_rays": failed,
        })

    print(f"# EMTomo on `{experiment_id}`\n")
    print(f"{len(data.station_ids)} stations, {len(data.event_ids)} events. "
          f"Forward convergence r→2r: `{json.dumps(metadata.get('convergence'))}`\n")
    print("- RMS: m/s against the slowness-averaged truth on inversion cells; cycle 0 = start model.")
    print("- Spheres A–F: mean(final − start)/mean(truth − start) over cells within 0.6 R.")
    print("- Wedge: cells whose layer differs between the flat start interface and the true dipping one "
          "(spheres excluded). gain = projection of the model change on the needed change "
          "(1 = fully corrected, 0 = untouched); corr = correlation of the two.")
    print("- Background: RMS outside spheres and wedge, above the deepest event.")
    print("- Boundary depth: where the y-median profile crosses the mid velocity of the last two layers.\n")
    done = [r for r in rows if not r.get("unfinished")]
    done.sort(key=lambda r: r["curve"][r["best"]])
    print("| run | RMS start → best @cycle → final | spheres A–F (final) | wedge gain / corr @best | "
          "wedge gain / corr final | background start → final | hypo error m: median first → last, p90 | s/cycle |")
    print("|---|---|---|---|---|---|---|---:|")
    for r in done:
        c = r["curve"]
        wb, wf = r["wedge_best"], r["wedge_final"]
        print(f"| {r['name']} | {c[0]:.0f} → {c[r['best']]:.0f} @{r['best']} → {c[-1]:.0f} | "
              + " ".join(f"{v:.2f}" for v in r["recovery"])
              + f" | {wb[0]:.2f} / {wb[1]:.2f} | {wf[0]:.2f} / {wf[1]:.2f} | "
              f"{r['background'][0]:.0f} → {r['background'][1]:.0f} | "
              f"{r['hypo'][0]:.0f} → {r['hypo'][1]:.0f}, {r['hypo'][2]:.0f} | {r['time']:.0f} |")
    for r in rows:
        if r.get("unfinished"):
            print(f"| {r['name']} | unfinished | | | | | | |")
    if done:
        truth = next(iter(truths.values()))
        step = max(1, len(truth.x) // 8)
        print("\n## Dipping boundary depth (km) by x, final model\n")
        print("| run | " + " | ".join(f"x={v:.1f}" for v in truth.x[::step]) + " |")
        print("|---|" + "---:|" * len(truth.x[::step]))
        print("| **truth (cell average)** | " + " | ".join(f"{v:.2f}" for v in truth.boundary_depth(truth.reference)[::step]) + " |")
        print("| true interface | " + " | ".join(f"{v:.2f}" for v in truth.dip[::step]) + " |")
        for r in done:
            print(f"| {r['name']} | " + " | ".join(f"{v:.2f}" for v in r["boundary"][::step]) + " |")
        print("\n## RMS (m/s) by cycle\n")
        n = max(len(r["curve"]) for r in done)
        print("| run | " + " | ".join(str(i) for i in range(n)) + " |")
        print("|---|" + "---:|" * n)
        for r in done:
            print(f"| {r['name']} | " + " | ".join(f"{v:.0f}" for v in r["curve"]) + " |")
        if misfit:
            print("\n## Data misfit (s): RMS of centred residuals, every 4th event\n")
            print("Columns are the models entering each cycle (0 = start). 'truth' = the cell-averaged "
                  "true model on the same grid at the true hypocentres; going below it means fitting noise. "
                  "For EM runs relocation also absorbs residuals, so there it is only a reference level.\n")
            cycles = sorted({c for r in done for c in r["misfits"][0]})
            print("| run | truth | " + " | ".join(f"c{c}" for c in cycles) + " |")
            print("|---|---:|" + "---:|" * len(cycles))
            for r in done:
                values, floor = r["misfits"]
                print(f"| {r['name']} | {floor:.4f} | " + " | ".join(
                    f"{values[c]:.4f}" if c in values else "" for c in cycles) + " |")
        if any(r["failed_rays"] for r in done):
            print("\nFailed rays: " + ", ".join(f"{r['name']}: {r['failed_rays']}" for r in done if r["failed_rays"]))


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if a != "--misfit"]
    main(*arguments[:4], misfit="--misfit" in sys.argv[1:])
