"""Saved-artifact-only weighted-mean counterpart; fail closed on provenance mismatch."""
from pathlib import Path
import csv
import hashlib
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_soft_gain_vs_top_n import SERIES, ROOT, sha

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "soft_gain_vs_top_n_lcurve.json"
STEM = "soft_velocity_gain_vs_weighted_mean_hypocenter_gain"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def xyz(records):
    return np.array([[float(r[a + "_m"]) for a in "xyz"] for r in records])


def point_locations(positions, weights, side, shape):
    """Use every saved weight, including tiny weights; no re-tempering or truncation."""
    require(positions.ndim == 2 and positions.shape[1] == 3 and len(positions) > 0,
            "Invalid positions")
    require(weights.shape == (len(positions),) and np.isfinite(weights).all()
            and (weights >= 0).all() and np.isfinite(positions).all()
            and (positions >= 0).all() and (positions < np.asarray(shape)).all(),
            "Invalid saved hypotheses")
    total = float(weights.sum())
    require(total > 0 and np.isclose(total, 1, rtol=0, atol=1e-6), "Unnormalized weights")
    coordinates = (positions + 0.5) * side
    return (weights / total) @ coordinates, coordinates[np.argmax(weights)], total


def extract():
    source = read(SOURCE)
    points = source["points"]
    require(len(points) == len({p["pair_id"] for p in points}) == 28, "Expected 28 unique pairs")
    hashes = {}

    def track(path):
        hashes[os.path.relpath(path, ROOT)] = sha(path)
        return path

    track(SOURCE)
    reports = {policy: read(track(ROOT / "paper/figures" / filename)) for policy, filename in (
        ("lcurve", "hard_soft_lcurve_comparison_metrics.json"),
        ("final", "final_comparison_metrics.json"))}
    require(sha(ROOT / "paper/figures/final_comparison_metrics.json") == source["tables_sha256"]
            and sha(ROOT / "paper/figures/hard_soft_lcurve_comparison_metrics.json") == source["lcurve_tables_sha256"],
            "Source table hashes changed")
    selections = {r["pair_id"]: r for r in rows(track(ROOT / "paper/exp_data/hard_soft_lcurve_avg_abs_pct_dev_topn_sweep.csv"))}
    experiments, cache = {}, {}

    def truth_data(identifier):
        if identifier in experiments:
            return experiments[identifier]
        base = ROOT.parent / "forward_modeling/experiments"
        directory = base / "input" / identifier
        metadata = read(track(base / "output" / identifier / "metadata.json"))
        for name, expected in metadata["input_sha256"].items():
            require(sha(track(directory / name)) == expected, f"Changed truth input {identifier}/{name}")
        events, stations = rows(directory / "events.csv"), rows(directory / "stations.csv")
        ids, station_ids = [r["event_id"] for r in events], [r["station_id"] for r in stations]
        require(len(set(ids)) == len(ids) and len(set(station_ids)) == len(station_ids), "Duplicate IDs")
        arrivals = rows(track(base / "output" / identifier / "arrivals.csv"))
        require(len(arrivals) == len(ids) * len(station_ids) and
                {(r["event_id"], r["station_id"]) for r in arrivals} ==
                {(e, s) for e in ids for s in station_ids}, "Arrival ID mismatch")
        with np.load(directory / "model.npz", allow_pickle=False) as model:
            require(np.array_equal(model["origin_m"], [0, 0, 0]), "Nonzero truth origin")
        coordinates = xyz(events)
        require(np.isfinite(coordinates).all(), "Invalid truth coordinates")
        experiments[identifier] = (ids, coordinates, xyz(stations), metadata["input_sha256"])
        return experiments[identifier]

    def evaluate(p, mode):
        run_id, policy = p[f"{mode}_run_id"], p["checkpoint_policy"]
        update = p[f"{mode}_update"]
        iteration = update if policy == "lcurve" else update - 1
        key = (run_id, iteration)
        if key in cache:
            return cache[key]
        reference = next(r for r in reports[policy]["records"] if r["run_id"] == run_id)
        require(reference["weights_iteration_index"] == iteration, "Location checkpoint mismatch")
        run = ROOT / "runs" / run_id
        meta = read(track(run / "meta.json"))
        params, grid = meta["run_params"], meta["grid_info"]
        identifier = meta["source_experiment"]["id"]
        ids, truth, stations, input_hashes = truth_data(identifier)
        require(meta["run_id"] == run_id and params["candidate_mode"] == mode
                and reference["experiment_id"] == identifier, "Run identity mismatch")
        require(meta["source_experiment"]["model_sha256"] == input_hashes["model.npz"]
                and np.array_equal(meta["station_locs"], stations)
                and params["n_events"] == len(ids), "Truth/run geometry mismatch")
        if "event_ids" in meta["source_experiment"]:
            require(meta["source_experiment"]["event_ids"] == ids, "Event order mismatch")
        # Legacy runs lack event IDs: bind the ordered truth and file digests to the original report.
        if policy == "lcurve":
            old_hash = reports[policy]["source_files_sha256"][os.path.relpath(
                ROOT.parent / "forward_modeling/experiments/input" / identifier / "events.csv", ROOT)]
        else:
            old_hash = reports[policy]["experiments"][identifier]["input_sha256"]["events.csv"]
        require(old_hash == input_hashes["events.csv"], "Original report truth changed")
        quality = [json.loads(line) for line in track(run / "quality.jsonl").read_text().splitlines() if line.strip()]
        require([q["iter"] for q in quality] == list(range(len(quality))), "Invalid quality sequence")
        require(np.isclose(quality[update-1]["avg_abs_pct_dev"], p[f"{mode}_avg_abs_pct_dev"],
                           rtol=1e-12), "Velocity checkpoint mismatch")
        selection = selections[p["pair_id"]]
        require(selection[f"{mode}_run_id"] == run_id, "CSV run mismatch")
        if policy == "lcurve":
            require(int(selection[f"{mode}_selected_update_number"]) == iteration
                    and int(selection[f"{mode}_residual_iteration_index"]) == iteration,
                    "Frozen checkpoint mismatch")
        else:
            require(len(quality) == update and not (run / f"iter_{update}").exists(), "Not last E-step")
        cycle = run / f"iter_{iteration}"
        require(read(track(cycle / "complete.json"))["iter"] == iteration, "Incomplete E-step")
        require({d.name for d in cycle.glob("event_*") if d.is_dir()} ==
                {f"event_{i}" for i in range(len(ids))}, "Missing/extra events")
        mean_errors, map_errors, counts, positive, sums = [], [], [], [], []
        digest = hashlib.sha256()
        for i, event_id in enumerate(ids):
            payload = (cycle / f"event_{i}/weights.npz").read_bytes()
            digest.update(f"{i}\t{event_id}\t{hashlib.sha256(payload).hexdigest()}\n".encode())
            with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                positions, weights = saved["positions"], saved["weight_values"]
                require(np.array_equal(saved["weight_shape"], grid["fine_shape"])
                        and 1 <= len(positions) <= params["weights_top_n"], "Candidate grid/count mismatch")
                mean, maximum, total = point_locations(positions, weights, grid["fine_cell_size"], grid["fine_shape"])
                if mode == "hard":
                    require(np.count_nonzero(weights) == 1 and np.array_equal(mean, maximum),
                            "Hard distribution is not one-hot; investigate rather than substitute")
                mean_errors.append(float(np.linalg.norm(mean - truth[i])))
                map_errors.append(float(np.linalg.norm(maximum - truth[i])))
                counts.append(len(weights))
                positive.append(int(np.count_nonzero(weights)))
                sums.append(total)
        require(digest.hexdigest() == reference["row_id_weights_sha256"], "Original weights/event digest mismatch")
        maximum_p90 = float(np.percentile(map_errors, 90, method="linear"))
        require(np.isclose(maximum_p90, reference["hypocenter_p90_m"], rtol=1e-12, atol=1e-9),
                "Original P90 not reproduced")
        result = dict(run_id=run_id, mode=mode, experiment_id=identifier,
                      velocity_update=update, location_model_update=iteration,
                      weights_directory=os.path.relpath(cycle, ROOT), n_events=len(ids),
                      event_order_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                      row_id_weights_sha256=digest.hexdigest(),
                      saved_candidate_count_range=[min(counts), max(counts)],
                      positive_weight_count_range=[min(positive), max(positive)],
                      weight_sum_range=[min(sums), max(sums)],
                      maximum_weight_p90_m=maximum_p90,
                      weighted_mean_p90_m=float(np.percentile(mean_errors, 90, method="linear")),
                      event_ids_saved_in_meta="event_ids" in meta["source_experiment"],
                      validation="truth hashes, geometry, complete event set, checkpoint, original weights digest and original P90 passed")
        cache[key] = result
        print(f"{mode} iter_{iteration}: {maximum_p90:.6f} -> {result['weighted_mean_p90_m']:.6f} m", flush=True)
        return result

    output = []
    for p in points:
        h, s = (evaluate(p, mode) for mode in ("hard", "soft"))
        require(h["experiment_id"] == s["experiment_id"] and h["event_order_sha256"] == s["event_order_sha256"],
                "Pair truth mismatch")
        for mode, record in (("hard", h), ("soft", s)):
            require(np.isclose(record["maximum_weight_p90_m"], p[f"{mode}_p90_m"], rtol=1e-12), "Source P90 mismatch")
        gain = 100 * (h["weighted_mean_p90_m"] - s["weighted_mean_p90_m"]) / h["weighted_mean_p90_m"]
        output.append(dict(p, hard_weighted_mean_p90_m=h["weighted_mean_p90_m"],
                           soft_weighted_mean_p90_m=s["weighted_mean_p90_m"],
                           weighted_mean_p90_gain_percent=gain, plotted_x=gain,
                           plotted_y=p["soft_gain_percent"],
                           gain_change_percentage_points=gain-p["p90_gain_percent"]))
    for path in (Path(__file__), HERE / "make_soft_gain_vs_top_n.py",
                 ROOT / "tomography/tomography_events.py", ROOT / "tomography/tomography_logging.py",
                 ROOT / "experiment_data.py", ROOT / "paper/paper_scripts/make_hard_soft_lcurve_metrics.py",
                 ROOT / "paper/paper_scripts/make_final_comparison_metrics.py"):
        track(path)
    return dict(source=SOURCE.name, source_sha256=sha(SOURCE), n_points=len(output),
                x_metric="Equal-event linear P90 of Euclidean errors of sum(w*x)/sum(w); gain=100*(hard-soft)/hard",
                y_metric=source["metric"], checkpoint_policy=source["checkpoint_policy"],
                p90_timing=source["p90_timing"],
                coordinates="Saved refined positions in cell-centre index units: metres=(positions+0.5)*fine_cell_size",
                weights="All original saved weight_values normalized by their sum; no threshold, top-N change, temperature change or likelihood recomputation. Hard weights are verified one-hot, not uniform over searched candidates.",
                event_alignment=reports["lcurve"]["event_alignment"],
                caveats=reports["lcurve"]["caveats"] + [
                    "Weighted means can lie between separated modes; this is a point-estimator comparison, not posterior uncertainty.",
                    "Full means refer to the saved retained distribution, not discarded search candidates or a continuous posterior.",
                    "Velocity y values and all original pair fields are preserved except plotted_x, which now denotes the new horizontal coordinate.",
                    "No inversions, localization, FMM, solver runs or article changes. Historical event IDs are not independently present in run metadata; row-order protocol and original row/ID/file digests are verified."],
                weights_digest="SHA256(row_index TAB event_id TAB weights_file_SHA256 LF)",
                source_files_sha256=hashes, records=list(cache.values()), points=output)


def main():
    report = extract()
    points = report["points"]
    x = np.array([p["plotted_x"] for p in points])
    y = np.array([p["plotted_y"] for p in points])
    require(np.isfinite(x).all() and np.isfinite(y).all(), "Nonfinite gains")
    report["summary"] = dict(soft_better_weighted_mean=int((x > 0).sum()),
        soft_better_maximum_weight=sum(p["p90_gain_percent"] > 0 for p in points),
        gain_range_percent=[float(x.min()), float(x.max())],
        sign_changed_pair_ids=[p["pair_id"] for p in points if np.sign(p["plotted_x"]) != np.sign(p["p90_gain_percent"])],
        unique_run_checkpoints=len(report["records"]),
        unique_run_checkpoint_events=sum(r["n_events"] for r in report["records"]))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.axhline(0, color="0.4", linewidth=.9)
        ax.axvline(0, color="0.4", linewidth=.9)
        for section, (label, color) in SERIES.items():
            mask = np.array([p["section"] == section for p in points])
            ax.scatter(x[mask], y[mask], s=45, color=color, alpha=.9,
                       edgecolors="white", linewidths=.7, zorder=3,
                       label=f"{label} ({mask.sum()})")
        ax.set_xlabel("Soft gain in weighted-mean hypocentre P90 (%)")
        ax.set_ylabel("Soft gain in velocity error (%)")
        ax.grid(color="0.9", linewidth=.6)
        ax.set_axisbelow(True)
        ax.margins(.09)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, frameon=False, fontsize=8)
        fig.savefig(HERE.parent / "png" / f"{STEM}.png", dpi=220, bbox_inches="tight")
        fig.savefig(HERE / f"{STEM}.pdf", bbox_inches="tight")
        plt.close(fig)
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
