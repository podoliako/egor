#!/usr/bin/env python3
"""
Tomography log viewer — HTTP backend.

Install:  pip install flask numpy
Run:      python server.py [--runs-dir runs] [--port 5050]
Open:     http://localhost:5050
"""

import argparse
import json
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

app = Flask(__name__)
RUNS_DIR = Path("runs")          # overridden by CLI arg
VIEWER_DIR = Path(__file__).parent  # directory containing viewer.html


# ─── helpers ─────────────────────────────────────────────────────────────────

def _rd(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    if not d.is_dir():
        abort(404, description=f"Run not found: {run_id}")
    return d


def _npy(path: Path):
    """Load .npy or return None if missing."""
    return np.load(path) if path.exists() else None


def _slice_y(arr: np.ndarray, y: int) -> np.ndarray:
    """arr[:, y, :]  →  shape (nx, nz), y clamped to valid range."""
    y = int(np.clip(y, 0, arr.shape[1] - 1))
    return arr[:, y, :]


# ─── routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    return send_from_directory(str(VIEWER_DIR), "viewer.html")


@app.route("/api/runs")
def api_runs():
    if not RUNS_DIR.exists():
        return jsonify([])
    runs = sorted(
        (d.name for d in RUNS_DIR.iterdir()
         if d.is_dir() and d.name.startswith("run_")),
        reverse=True,
    )
    return jsonify(runs)


@app.route("/api/runs/<rid>/meta")
def api_meta(rid):
    p = _rd(rid) / "meta.json"
    if not p.exists():
        return jsonify({})
    return p.read_text(), 200, {"Content-Type": "application/json"}


@app.route("/api/runs/<rid>/info")
def api_info(rid):
    rd = _rd(rid)

    # available iteration indices
    iters = sorted(
        int(d.name[5:]) for d in rd.iterdir()
        if d.is_dir() and d.name.startswith("iter_")
    )

    # number of stations — infer from first station_fields.npy found
    n_stations = 0
    for i in iters:
        p = rd / f"iter_{i}" / "station_fields.npy"
        if p.exists():
            n_stations = int(np.load(p, mmap_mode="r").shape[0])
            break

    # number of events — from meta
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


@app.route("/api/runs/<rid>/slice")
def api_slice(rid):
    """
    Return a 2-D y-slice of the requested 3-D array.

    Required query params
    ─────────────────────
    type    model | station_field | weights | G
    y       int   (slice index along Y axis)
    iter    int   (iteration number)

    type=model         model_type: initial | true | iter | delta_s
    type=station_field station: int
    type=weights       event: int
    type=G             event, grid (coarse|fine), station_a, [station_b]

    Response JSON
    ─────────────
    {
      slice:      [[float, ...], ...],   # shape (nx, nz)
      shape:      [nx, nz],
      full_shape: [nx, ny, nz],
      vmin: float, vmax: float,          # for colorscale
    }
    """
    rd   = _rd(rid)
    dtype = request.args.get("type", "model")
    y    = request.args.get("y",    0, type=int)
    it   = request.args.get("iter", 0, type=int)

    arr = None

    if dtype == "model":
        mt = request.args.get("model_type", "iter")
        path_map = {
            "initial": rd / "initial_model.npy",
            "true":    rd / "true_model.npy",
            "iter":    rd / f"iter_{it}" / "model.npy",
            "delta_s": rd / f"iter_{it}" / "delta_s.npy",
        }
        arr = _npy(path_map.get(mt, Path("__none__")))

    elif dtype == "station_field":
        sta  = request.args.get("station", 0, type=int)
        full = _npy(rd / f"iter_{it}" / "station_fields.npy")
        if full is not None and sta < full.shape[0]:
            arr = full[sta]

    elif dtype == "weights":
        ev  = request.args.get("event", 0, type=int)
        arr = _npy(rd / f"iter_{it}" / f"event_{ev}" / "weights.npy")

    elif dtype == "G":
        ev      = request.args.get("event",     0, type=int)
        grid    = request.args.get("grid",  "coarse")
        ev_dir  = rd / f"iter_{it}" / f"event_{ev}"
        if grid == "fine":
            sa  = request.args.get("station_a", 0, type=int)
            arr = _npy(ev_dir / f"G_station_{sa}.npy")
        else:
            sa  = request.args.get("station_a", 0, type=int)
            sb  = request.args.get("station_b", 1, type=int)
            sa, sb = sorted([sa, sb])
            arr = _npy(ev_dir / f"G_s{sa}_s{sb}.npy")

    if arr is None or arr.ndim != 3:
        return jsonify({"slice": None, "shape": None, "full_shape": None,
                        "vmin": 0, "vmax": 1})

    s2d = _slice_y(arr, y)
    finite = s2d[np.isfinite(s2d)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0

    return jsonify({
        "slice":      s2d.tolist(),
        "shape":      list(s2d.shape),
        "full_shape": list(arr.shape),
        "vmin": vmin,
        "vmax": vmax,
    })


@app.route("/api/runs/<rid>/G_files")
def api_g_files(rid):
    """List available G_station_* and G_s*_s* files for a given iter+event."""
    rd     = _rd(rid)
    it     = request.args.get("iter",  0, type=int)
    ev     = request.args.get("event", 0, type=int)
    ev_dir = rd / f"iter_{it}" / f"event_{ev}"

    if not ev_dir.is_dir():
        return jsonify({"station_indices": [], "pairs": []})

    sta_indices = sorted(
        f.stem[len("G_station_"):] for f in ev_dir.glob("G_station_*.npy")
    )
    pairs = sorted(
        f.stem for f in ev_dir.glob("G_s*_s*.npy")
        if "station" not in f.name
    )
    return jsonify({"station_indices": sta_indices, "pairs": pairs})


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tomography viewer server")
    ap.add_argument("--runs-dir", default="runs",
                    help="Directory that contains run_* folders  (default: ./runs)")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    RUNS_DIR = Path(args.runs_dir)
    print(f"  Runs dir : {RUNS_DIR.resolve()}")
    print(f"  Viewer   : http://localhost:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=True)