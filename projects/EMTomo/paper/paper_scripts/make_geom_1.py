"""Compare the best velocity updates of the two latest runs with their truth."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "paper/figures/geom_1"


def main():
    candidates = []
    for path in (ROOT / "runs").glob("run_*"):
        if (path / "meta.json").exists() and (path / "quality.jsonl").exists():
            meta = json.loads((path / "meta.json").read_text())
            if "started_at" in meta:
                candidates.append((meta["started_at"], path, meta))
    runs = sorted(candidates, key=lambda row: row[0], reverse=True)[:2]
    assert len(runs) == 2
    fields, records = [], []
    truth = None
    for _, path, meta in runs:
        source = (ROOT.parent / "forward_modeling/experiments/input"
                  / meta["source_experiment"]["id"] / "model.npz")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == meta["source_experiment"]["model_sha256"]
        with np.load(source, allow_pickle=False) as data:
            reference = data["velocity"].copy()
            truth_cell = float(data["cell_size_m"]) / 1000
        if truth is None:
            truth = reference
            fields.append(("True model", truth / 1000, truth_cell))
        else:
            np.testing.assert_array_equal(truth, reference)
        quality = [json.loads(line) for line in (path / "quality.jsonl").read_text().splitlines() if line.strip()]
        best = min(quality, key=lambda row: row["rms_m_s"])
        iteration = best["iter"]
        # Iteration models precede the update; quality is measured after it.
        next_model = path / f"iter_{iteration + 1}/model.npy"
        model_path = next_model if next_model.exists() else path / "final_model.npy"
        model = np.load(model_path, allow_pickle=False)
        cell = meta["grid_info"]["coarse_cell_size"] / 1000
        repeats = np.array(model.shape) // np.array(truth.shape)
        assert np.all(np.array(model.shape) == np.array(truth.shape) * repeats)
        expanded_truth = truth
        for axis, count in enumerate(repeats):
            expanded_truth = np.repeat(expanded_truth, count, axis=axis)
        rms = float(np.sqrt(np.mean((model - expanded_truth) ** 2)))
        np.testing.assert_allclose(rms, best["rms_m_s"], rtol=1e-6)
        mode = meta["run_params"]["candidate_mode"]
        fields.append((f"{mode.capitalize()} · best update {iteration + 1}\nVolume RMS = {rms:.1f} m/s", model / 1000, cell))
        records.append({"run": path.name, "mode": mode, "quality_iteration_zero_based": iteration,
                        "update_number": iteration + 1, "model": str(model_path.relative_to(ROOT)),
                        "volume_rms_m_s": rms, "source_model": str(source)})

    low = min(float(field.min()) for _, field, _ in fields)
    high = max(float(field.max()) for _, field, _ in fields)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 11, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), layout="constrained")
    for col, (label, field, cell) in enumerate(fields):
        nx, ny, nz = np.array(field.shape) * cell
        assert 0 <= 1.5 < nz and 0 <= 6.4 < ny
        horizontal = field[:, :, int(np.floor(1.5 / cell))].T
        vertical = field[:, int(np.floor(6.4 / cell)), :].T
        for row, section in enumerate((horizontal, vertical)):
            ax = axes[row, col]
            image = ax.imshow(section, extent=(0, nx, 0, ny) if row == 0 else (0, nx, nz, 0),
                              origin="lower" if row == 0 else "upper", aspect="equal",
                              interpolation="nearest", cmap="RdBu_r", vmin=low, vmax=high)
            ax.set_xlabel("$X$ (km)")
            ax.set_ylabel("$Y$ (km)" if row == 0 else "Depth (km)")
            ax.set_title(label if row == 0 else "")
            ax.set_xticks(np.arange(0, nx + .01, 4))
        if col == 0:
            axes[0, col].text(0, 1.24, "$Z = 1.5$ km", transform=axes[0, col].transAxes, fontsize=13)
            axes[1, col].set_title("$Y = 6.4$ km", loc="left")
    fig.colorbar(image, ax=axes, location="right", shrink=.85, label="$V_P$ (km/s)")
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT.with_suffix("." + suffix), dpi=220)
    plt.close(fig)
    report = {"slices_km": {"z": 1.5, "y": 6.4}, "selection": "Minimum post-update full-volume velocity RMS against truth",
              "sampling": "Native piecewise-constant cells; no smoothing; zero-based quality iteration refers to post-update model",
              "color_limits_km_s": [low, high], "runs": records}
    OUTPUT.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
