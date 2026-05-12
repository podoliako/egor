#!/usr/bin/env python3
"""
Tomography log viewer — HTTP backend.

Install:  pip install flask numpy
Run:      python server.py [--runs-dir /runs] [--host 0.0.0.0] [--port 5050]
Open:     http://localhost:5050
"""

import argparse
import json
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

app = Flask(__name__)
RUNS_DIR   = Path("runs")
VIEWER_DIR = Path(__file__).parent


# ─── helpers ──────────────────────────────────────────────────────────────────

def _rd(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    if not d.is_dir():
        abort(404, description=f"Run not found: {run_id}")
    return d


def _npy(path: Path):
    return np.load(path) if path.exists() else None


def _slice_y(arr: np.ndarray, y: int) -> np.ndarray:
    """arr[:, y, :] → (nx, nz), y clamped."""
    y = int(np.clip(y, 0, arr.shape[1] - 1))
    return arr[:, y, :]


def _arr_resp(arr, y: int):
    """Turn 3-D array into a slice response dict, or a null response."""
    if arr is None or arr.ndim != 3:
        return {"slice": None, "shape": None, "full_shape": None, "vmin": 0, "vmax": 1}
    s2d = _slice_y(arr, y)
    fin = s2d[np.isfinite(s2d)]
    return {
        "slice":      s2d.tolist(),
        "shape":      list(s2d.shape),
        "full_shape": list(arr.shape),
        "vmin": float(fin.min()) if fin.size else 0.0,
        "vmax": float(fin.max()) if fin.size else 1.0,
    }


# ─── static ───────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    return send_from_directory(str(VIEWER_DIR), "viewer.html")


# ─── run list ─────────────────────────────────────────────────────────────────

@app.route("/api/runs")
def api_runs():
    if not RUNS_DIR.exists():
        return jsonify([])
    runs = sorted(
        (d.name for d in RUNS_DIR.iterdir() if d.is_dir()),
        reverse=True,
    )
    return jsonify(runs)


# ─── meta + info ──────────────────────────────────────────────────────────────

@app.route("/api/runs/<rid>/meta")
def api_meta(rid):
    p = _rd(rid) / "meta.json"
    if not p.exists():
        return jsonify({})
    meta = json.loads(p.read_text())

    # enrich with grid ny values for Y-sliders
    gi = meta.get("grid_info", {})
    meta["coarse_ny"] = gi.get("coarse_shape", [1, 1, 1])[1] if "coarse_shape" in gi else 1
    meta["fine_ny"]   = gi.get("fine_shape",   [1, 1, 1])[1] if "fine_shape"   in gi else 1

    return jsonify(meta)


@app.route("/api/runs/<rid>/info")
def api_info(rid):
    rd = _rd(rid)

    iters = sorted(
        int(d.name[5:]) for d in rd.iterdir()
        if d.is_dir() and d.name.startswith("iter_")
    )

    n_stations = 0
    for i in iters:
        p = rd / f"iter_{i}" / "station_fields.npy"
        if p.exists():
            n_stations = int(np.load(p, mmap_mode="r").shape[0])
            break

    meta_path = rd / "meta.json"
    n_events = 0
    if meta_path.exists():
        n_events = len(json.loads(meta_path.read_text()).get("event_locs", []))

    return jsonify({
        "iterations":     iters,
        "n_stations":     n_stations,
        "n_events":       n_events,
        "has_true_model": (rd / "true_model.npy").exists(),
    })


# ─── timing ───────────────────────────────────────────────────────────────────

@app.route("/api/runs/<rid>/timing")
def api_timing(rid):
    """Read timing.jsonl → [{iter, elapsed_s}, ...]"""
    p = _rd(rid) / "timing.jsonl"
    if not p.exists():
        return jsonify([])
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return jsonify(rows)


@app.route("/api/runs/<rid>/quality")
def api_quality(rid):
    """Read quality.jsonl → [{iter, avg_abs_pct_dev}, ...]"""
    p = _rd(rid) / "quality.jsonl"
    if not p.exists():
        return jsonify([])
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return jsonify(rows)


# ─── iter / event / weight lists ──────────────────────────────────────────────

@app.route("/api/runs/<rid>/iters_list")
def api_iters_list(rid):
    rd = _rd(rid)
    iters = sorted(
        [d.name for d in rd.iterdir() if d.is_dir() and d.name.startswith("iter_")],
        key=lambda x: int(x.split("_")[1]),
    )
    return jsonify(iters)


@app.route("/api/runs/<rid>/events_list")
def api_events_list(rid):
    rd  = _rd(rid)
    itr = request.args.get("iter", "0")
    d   = rd / f"iter_{itr}"
    if not d.exists():
        return jsonify([])
    evs = sorted(
        [x.name for x in d.iterdir() if x.is_dir() and x.name.startswith("event_")],
        key=lambda x: int(x.split("_")[1]),
    )
    return jsonify(evs)


@app.route("/api/runs/<rid>/weights_list")
def api_weights_list(rid):
    rd  = _rd(rid)
    itr = request.args.get("iter",  "0")
    ev  = request.args.get("event", "0")
    d   = rd / f"iter_{itr}" / f"event_{ev}"
    if not d.exists():
        return jsonify([])
    ws = sorted(
        [x.name for x in d.iterdir() if x.is_dir() and x.name.startswith("weight_")],
        key=lambda x: int(x.split("_")[1]),
    )
    return jsonify(ws)


# ─── main slice endpoint ───────────────────────────────────────────────────────

@app.route("/api/runs/<rid>/slice")
def api_slice(rid):
    """
    Universal 2-D y-slice endpoint.

    Query params
    ────────────
    type        model | true_model | station_field | weights | misfit | G | delta_s
    y           int   y-slice index
    iter        int   iteration number

    type=model         model_type: initial | true | iter
    type=station_field station: int
    type=weights       event: int
    type=misfit        event: int
    type=G             event: int, weight: int, station: int   (fine grid)
    type=delta_s       (no extra params)
    """
    rd    = _rd(rid)
    dtype = request.args.get("type", "model")
    y     = request.args.get("y",    0, type=int)
    it    = request.args.get("iter", 0, type=int)

    arr = None

    if dtype == "model":
        mt = request.args.get("model_type", "iter")
        paths = {
            "initial": rd / "initial_model.npy",
            "true":    rd / "true_model.npy",
            "iter":    rd / f"iter_{it}" / "model.npy",
        }
        arr = _npy(paths.get(mt, Path("__none__")))

    elif dtype == "true_model":
        arr = _npy(rd / "true_model.npy")

    elif dtype == "delta_s":
        arr = _npy(rd / f"iter_{it}" / "delta_s.npy")

    elif dtype == "station_field":
        sta  = request.args.get("station", 0, type=int)
        full = _npy(rd / f"iter_{it}" / "station_fields.npy")
        if full is not None and sta < full.shape[0]:
            arr = full[sta]   # shape (nx, ny, nz)

    elif dtype == "weights":
        ev  = request.args.get("event", 0, type=int)
        arr = _npy(rd / f"iter_{it}" / f"event_{ev}" / "weights.npy")

    elif dtype == "misfit":
        ev  = request.args.get("event", 0, type=int)
        arr = _npy(rd / f"iter_{it}" / f"event_{ev}" / "misfit.npy")

    elif dtype == "G":
        ev     = request.args.get("event",   0, type=int)
        wt     = request.args.get("weight",  0, type=int)
        sta    = request.args.get("station", 0, type=int)
        arr    = _npy(rd / f"iter_{it}" / f"event_{ev}" / f"weight_{wt}" / f"G_station_{sta}.npy")

    return jsonify(_arr_resp(arr, y))


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tomography viewer server")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--host",     default="0.0.0.0")
    ap.add_argument("--port",     type=int, default=5050)
    args = ap.parse_args()

    RUNS_DIR = Path(args.runs_dir)
    print(f"  Runs dir : {RUNS_DIR.resolve()}")
    print(f"  Viewer   : http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False)
