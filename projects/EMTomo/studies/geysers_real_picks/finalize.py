"""Freeze regional GEY start and native training inputs; NEVER launch a solver.

Requires prepare.py output and the archived official NCEDC shadow2000.txt.
The split is immutable; this stage creates a separate methods/ directory.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from studies.geysers_real_picks.tomodd_inputs import Domain, GradientModel, build_inputs, write_bundle, SUPPORT_FILES
from config import InversionConfig
from experiment_data import load_real_pick_experiment
from experiment_runner import prepare_real_inversion
from velocity_model import VelocityModel

DATA = ROOT / "data/geysers_2011/real_experiment_v1"
SOURCE = ROOT / "data/geysers_2011/regional_model/shadow2000.txt"
SOURCE_URL = "https://ncedc.org/ftp/pub/doc/ncsn/shadow2000.txt"
SOURCE_SHA256 = "b292262941f880fa8f5bc81114c4bc4a080a89b843b068f22b8e53ec53abaa23"
SCREEN = ROOT / "runs/geysers_faulted_layers/soft_screen_20261028"
SUPPORT = ROOT / "runs/matched_geometry_20261021/snapshot/support"
SOLVER = ROOT / "runs/tomodd_guard_fix_20261027/build/tomoDD-SE"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def regional_profile(source=SOURCE):
    if sha(source) != SOURCE_SHA256:
        raise ValueError("Official model source hash changed; review before changing the experiment")
    text = source.read_text(errors="replace")
    section = text.split("Depth-Velocity points for models with linear gradients in layers", 1)[1]
    match = re.search(r"^GEY\s+Z ([^\n]+)\nV ([^\n]+)", section, re.M)
    if match is None:
        raise ValueError("Missing GEY gradient model")
    # The official text omits spaces in '8.0020.0022.00'; fixed two decimals disambiguate.
    depth, speed = [[float(v) for v in re.findall(r"\d+\.\d{2}", part)] for part in match.groups()]
    if depth != [0., 3., 8., 20., 22.] or speed != [4.10, 5.47, 5.75, 6.02, 7.90]:
        raise ValueError("Unexpected GEY model nodes")
    return dict(
        name="NCSN GEY regional linear-gradient Vp", source_url=SOURCE_URL,
        source_sha256=SOURCE_SHA256, source_section="Table 8: MULTIPLE VELOCITY MODELS",
        original_reference="Eberhart-Phillips & Oppenheimer, JGR 1984 (as credited by NCEDC)",
        regional_reference="Oppenheimer et al., NCSN Bulletin 1992, USGS OFR 93-578",
        source_depth_km=depth, vp_km_s=speed, interpolation="linear in Vp; not slowness",
        datum_policy="Assume source zero and catalog depth zero are sea level; not independently verified",
        local_depth_km=[d + 1.166 for d in depth], top_elevation_m=1166,
        above_zero_policy="constant Vp=4.10 km/s above source zero (explicit experiment choice)",
        below_last_node_policy="constant Vp=7.90 km/s, per source half-space convention",
        station_delays="not imported; no station-specific delay parameters",
        egs_catalog_model="unknown: model-code columns 111-113 of downloaded headers are blank",
        grid_representation="EMTomo cell-center samples, nearest subdivision; tomoDD nodal samples with native interpolation",
        vp_vs=1.73, vp_vs_policy="unused in P-only inversion; placeholder required by native MOD format",
    )


def finalize(data=DATA):
    output = data / "methods"
    if output.exists():
        raise FileExistsError(f"Refusing existing methods: {output}")
    split_manifest = json.loads((data / "manifest.json").read_text())
    for name, digest in split_manifest["output_sha256"].items():
        if sha(data / name) != digest:
            raise ValueError(f"Split artifact changed: {name}")
    observations = load_real_pick_experiment(data / "emtomo")
    profile = regional_profile()
    g = json.loads((data / "geometry.json").read_text())
    domain = Domain(g["reference_latitude"], g["reference_longitude"], g["x_azimuth_degrees"],
                    tuple(v / 1000 for v in g["domain_extent_m"]), g["top_elevation_m"],
                    horizontal_offset_km=tuple(v / 1000 for v in g["geographic_reference_in_new_xy_m"]))
    model = GradientModel(tuple(profile["local_depth_km"]), tuple(profile["vp_km_s"]))
    shape = tuple(int(v / 1000) for v in observations.domain_extent_m)
    depth = np.arange(shape[2]) + .5
    start = np.broadcast_to(np.interp(depth, model.depths_km, model.vp_km_s) * 1000, shape).copy()
    stations = rows(data / "stations.csv")
    for row, xyz in zip(stations, observations.station_coordinates_m):
        np.testing.assert_allclose(domain.local(row["latitude"], row["longitude"],
                                               -float(row["elevation_m"]) / 1000) * 1000, xyz, atol=1e-8)
    # Only train.csv is supplied to native pairing; holdout.csv is never parsed here.
    bundle = build_inputs(rows(data / "train.csv"), rows(data / "catalog.csv"), stations,
                          domain=domain, model=model, sigma_km=0., seed=42)
    configs, source_hashes = {}, {}
    for mode, case in (("soft", "s16"), ("hard", "h07")):
        path = SCREEN / "cases" / case / "result.json"
        original = json.loads(path.read_text())
        config = original["config"].copy()
        config.update(initial_layer_boundaries_km=None, initial_layer_velocities_m_s=None,
                      initial_gradient_m_s=None, initial_velocity_m_s=4100.,
                      weight_noise_relative_sigma=None, weight_noise_absolute_sigma_s=None,
                      run_name=f"geysers-real-{mode}", run_version="1.5-sparse-real",
                      runs_dir=str(ROOT / "runs"))
        configs[mode] = asdict(InversionConfig(**config))
        source_hashes[str(path.relative_to(ROOT))] = sha(path)
        prepare_real_inversion(data / "emtomo", InversionConfig(**config),
                               initial_model=VelocityModel(start, 1000.))
    support = {name: (SUPPORT / name).read_bytes() for name in SUPPORT_FILES}
    solver_hash = sha(SOLVER)
    output.mkdir()
    np.save(output / "initial_velocity_m_s.npy", start, allow_pickle=False)
    save_json(output / "regional_model.json", profile)
    save_json(output / "emtomo_configs.json", configs)
    native = write_bundle(bundle, output / "tomodd", support_files=support)
    manifest = dict(
        schema="geysers.real_methods.v1", preparation_complete=True, inversions_launched=False,
        split_manifest_sha256=sha(data / "manifest.json"),
        regional_source_sha256=SOURCE_SHA256, config_sources_sha256=source_hashes,
        source_code_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                           (Path(__file__), Path(__file__).with_name("tomodd_inputs.py"))},
        config_changes="Regional explicit start, sparse real observations, unknown pick sigma (not synthetic zero); other numerical settings retained",
        noise_policy="No noise added to measurements or tomoDD catalog; EMTomo likelihood sigmas must be explicitly chosen using train only before inversion",
        catalog_policy="tomoDD catalog starts, sigma=0; EMTomo has no catalog-location/origin input or prior",
        cv_policy="Future scoring must use final model and training-only location/origin estimate; never refit origin on holdout",
        native_validation="Python format/geometry/pair audit only; native solver has not been executed",
        native_solver={"path": str(SOLVER), "sha256": solver_hash, "horizontal_guard_fixed": True},
        counts={"events": len(observations.event_ids), "stations": len(observations.station_ids),
                "train": native["absolute_observations"], "holdout": split_manifest["counts"]["holdout_picks"],
                "differential_observations": native["differential_observations"],
                "event_pairs": len(native["event_pairs"]), "isolated_events": len(native["isolated_event_ids"])},
        output_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "initial_profile_m_s": start[0, 0].tolist(),
                      **manifest["counts"], "inversions_launched": False}, indent=2))
    return manifest


if __name__ == "__main__":
    finalize()
