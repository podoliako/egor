"""Solver-free tests: deterministic constrained split and training-only contract."""
import copy
import csv
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("geysers_prepare", Path(__file__).with_name("prepare.py"))
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)


def inputs():
    epochs = [dict(network="BG", station=f"S{i:02}", latitude="38.80275", longitude="-122.77070",
                   elevation_m=str(1166 - i), site_name="test", start_time="2010-01-01T00:00:00",
                   end_time="3000-01-01T00:00:00") for i in range(31)]
    picks, catalog = [], []
    for j, n in enumerate([5, 6, 7, 8, 15, 20, 31]):
        event = str(100 + j)
        origin = datetime(2011, 4, 1, 12, j, tzinfo=timezone.utc)
        catalog.append(dict(event_id=event, origin_time="unused original catalog time",
                            latitude="38.8", longitude="-122.77", depth_km="2.1",
                            magnitude="1.2", magnitude_type="ML"))
        for i in range(n):
            picks.append(dict(event_id=event, origin_time_utc=p.iso(origin), network="BG",
                              station=f"S{i:02}", location="", channel="DPZ", phase="P",
                              onset="", polarity="", weight_code="7" if i % 2 else "5",
                              pick_time_utc=p.iso(origin + timedelta(seconds=i - 1)),
                              seconds_from_catalog_origin=str(i - 1), seconds_from_first_P=str(i),
                              in_existing_26_station_geometry="True", source_file="test.html",
                              source_line=str(i + 1)))
    return picks, catalog, epochs


def rows(bundle, name):
    return bundle["tables"][name][1]


def assert_arrays_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_exact_quotas_caps_and_seeded_ties():
    counts = dict(a=6, b=7, c=8, d=15, e=20, f=31)
    assert p.allocate_quotas(counts) == dict(a=0, b=1, c=1, d=2, e=2, f=3)
    counts = {str(i): 7 for i in range(20)}
    quota = p.allocate_quotas(counts)
    expected = set(sorted(counts, key=lambda e: p.rank(42, "quota", e))[:14])
    assert {e for e, q in quota.items() if q} == expected
    assert quota == p.allocate_quotas(dict(reversed(list(counts.items()))))
    assert quota != p.allocate_quotas(counts, seed=43)
    # Round-half-up: 15 * .1 = 1.5 -> 2, not a truncated target.
    assert p.allocate_quotas({"a": 15}) == {"a": 2}
    capped = {str(i): 6 for i in range(10)} | {"large": 31}
    assert p.allocate_quotas(capped)["large"] == 9
    with pytest.raises(ValueError, match="infeasible"):
        p.allocate_quotas({"only": 6})
    with pytest.raises(ValueError, match="filtered"):
        p.allocate_quotas({"small": 5})


def test_determinism_reordering_coverage_and_no_mutation():
    source = inputs()
    before = copy.deepcopy(source)
    a = p.build_bundle(*source)
    b = p.build_bundle(*(list(reversed(table)) for table in source))
    assert source == before
    assert a["tables"] == b["tables"]
    assert a["counts"] == b["counts"]
    assert_arrays_equal(a["arrays"], b["arrays"])
    train = {p.pick_key(r) for r in rows(a, "train.csv")}
    held = {p.pick_key(r) for r in rows(a, "holdout.csv")}
    excluded = {p.pick_key(r) for r in rows(a, "excluded_picks.csv")}
    assert not train & held and not (train | held) & excluded
    assert train | held | excluded == {p.pick_key(r) for r in source[0]}
    assert len(held) == (len(train | held) + 5) // 10 == 9
    assert min(Counter(r["event_id"] for r in rows(a, "train.csv")).values()) >= 6
    assert not any(r["event_id"] == "101" for r in rows(a, "holdout.csv"))
    assert rows(a, "excluded_events.csv") == [dict(event_id="100", available_picks=5, reason=p.REASON)]
    assert all(r["reason"] == p.REASON for r in rows(a, "excluded_picks.csv"))
    assert len(rows(a, "catalog.csv")) == 7  # Includes excluded original event.
    for row in rows(a, "catalog.csv"):
        assert row["origin_time_utc"] == next(r["origin_time_utc"] for r in source[0]
                                              if r["event_id"] == row["event_id"])


def test_heldout_earliest_and_catalog_cannot_affect_training():
    source = inputs()
    original = p.build_bundle(*source)
    held = rows(original, "holdout.csv")[0]
    modified = copy.deepcopy(source)
    # Make a held-out pick earlier than every training pick (including across UTC day).
    for row in modified[0]:
        row["seconds_from_first_P"] = "999999"  # Must never be used or exported.
        if p.pick_key(row) == p.pick_key(held):
            row["pick_time_utc"] = "2011-03-31T23:59:59.500000Z"
            row["seconds_from_catalog_origin"] = "-43200.5"
        row["origin_time_utc"] = "2011-04-01T23:00:00Z"
    for row in modified[1]:
        row.update(latitude="10", longitude="20", depth_km="300", magnitude="5")
    changed = p.build_bundle(*modified)
    assert_arrays_equal(original["arrays"], changed["arrays"])
    assert rows(original, "event_time_references.csv") == rows(changed, "event_time_references.csv")
    relative = next(r for r in rows(changed, "holdout.csv") if p.pick_key(r) == p.pick_key(held))
    assert relative["seconds_from_train_reference"] < 0
    assert relative["seconds_from_catalog_origin"] == "-43200.5"
    for name in ("train.csv", "holdout.csv"):
        assert "seconds_from_first_P" not in changed["tables"][name][0]
        for row in rows(changed, name):
            assert "seconds_from_first_P" not in row
    references = {r["event_id"]: p.utc(r["training_reference_utc"])
                  for r in rows(changed, "event_time_references.csv")}
    for name in ("train.csv", "holdout.csv"):
        for row in rows(changed, name):
            assert row["seconds_from_train_reference"] == (
                p.utc(row["pick_time_utc"]) - references[row["event_id"]]).total_seconds()
    assert np.all(np.nanmin(changed["arrays"]["arrival_times_s"], axis=1) == 0)


@pytest.mark.parametrize("failure", ["duplicate", "origin", "epoch", "bounds", "missing_catalog"])
def test_invalid_input_fails_closed(failure):
    source = inputs()
    if failure == "duplicate":
        source[0].append(source[0][0].copy())
    elif failure == "origin":
        source[0][0]["origin_time_utc"] = "2011-04-01T12:01:00Z"
    elif failure == "epoch":
        source[2][0]["start_time"] = "2011-04-02T00:00:00"
    elif failure == "bounds":
        source[2][0]["longitude"] = "0"
    else:
        source[1].pop()
    with pytest.raises(ValueError):
        p.build_bundle(*source)


def test_write_contract_hashes_and_never_overwrite(tmp_path):
    source = inputs()
    a, b = tmp_path / "a", tmp_path / "b"
    bundle = p.build_bundle(*source)
    hashes = {"input.csv": p.sha256(b"test input")}
    manifest = p.write_bundle(a, bundle, hashes)
    p.write_bundle(b, p.build_bundle(*(list(reversed(t)) for t in source)), hashes)
    files_a = {str(f.relative_to(a)): f.read_bytes() for f in a.rglob("*") if f.is_file()}
    files_b = {str(f.relative_to(b)): f.read_bytes() for f in b.rglob("*") if f.is_file()}
    assert files_a == files_b
    assert set(f.name for f in (a / "emtomo").iterdir()) == {"arrivals.npz", "metadata.json"}
    assert manifest["source_sha256"] == hashes
    assert manifest["noise"]["sigma_s"] is None
    assert manifest["velocity_model"]["chosen"] is None
    assert manifest["hypocenter_truth"]["available"] is False
    assert set(manifest["output_sha256"]) == set(files_a) - {"manifest.json"}
    for name, digest in manifest["output_sha256"].items():
        assert p.sha256(files_a[name]) == digest
    metadata = json.loads(files_a["emtomo/metadata.json"])
    assert metadata["schema"] == "emtomo.real_picks.v1"
    assert metadata["units"] == {"coordinates": "m", "arrival_time": "s"}
    assert metadata["minimum_training_picks"] == 6
    assert metadata["time_reference"] == "earliest_training_pick_per_event"
    assert metadata["arrays_sha256"] == p.sha256(files_a["emtomo/arrivals.npz"])
    assert metadata["geometry"] == json.loads(files_a["geometry.json"])
    assert metadata["geometry"]["geographic_reference_in_new_xy_m"] == [13000, 3500]
    assert "2011-" not in files_a["emtomo/metadata.json"].decode()
    with np.load(a / "emtomo/arrivals.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"event_ids", "station_ids", "station_coordinates_m", "arrival_times_s"}
        assert arrays["event_ids"].dtype.kind == arrays["station_ids"].dtype.kind == "U"
        assert list(arrays["event_ids"]) == sorted(arrays["event_ids"])
        assert list(arrays["station_ids"]) == sorted(arrays["station_ids"])
        assert np.all(arrays["station_coordinates_m"] >= 0)
        assert np.all(arrays["station_coordinates_m"] <= p.EXTENT)
        assert np.isnan(arrays["arrival_times_s"]).any()
        event_index = {e: i for i, e in enumerate(arrays["event_ids"])}
        station_index = {s: i for i, s in enumerate(arrays["station_ids"])}
        for row in rows(bundle, "holdout.csv"):
            assert np.isnan(arrays["arrival_times_s"][event_index[row["event_id"]],
                                                    station_index[f'BG.{row["station"]}']])
        for row in rows(bundle, "train.csv"):
            assert arrays["arrival_times_s"][event_index[row["event_id"]],
                                             station_index[f'BG.{row["station"]}']] == row["seconds_from_train_reference"]
    with pytest.raises(FileExistsError):
        p.write_bundle(a, bundle, hashes)
    assert files_a == {str(f.relative_to(a)): f.read_bytes() for f in a.rglob("*") if f.is_file()}


def test_prepared_real_dataset():
    if not (p.OUTPUT / "manifest.json").exists():
        pytest.skip("Run prepare.py once to enable the real artifact audit")
    manifest = json.loads((p.OUTPUT / "manifest.json").read_text())
    counts = manifest["counts"]
    assert {k: counts[k] for k in ("source_events", "source_picks", "retained_events", "retained_picks",
                                  "training_picks", "holdout_picks", "excluded_events", "excluded_picks",
                                  "six_pick_events_without_holdout", "stations")} == dict(
        source_events=3776, source_picks=49158, retained_events=3471, retained_picks=47748,
        training_picks=42973, holdout_picks=4775, excluded_events=305, excluded_picks=1410,
        six_pick_events_without_holdout=285, stations=31)
    for name, digest in manifest["source_sha256"].items():
        assert p.sha256((p.ROOT / name).read_bytes()) == digest
    for name, digest in manifest["output_sha256"].items():
        assert p.sha256((p.OUTPUT / name).read_bytes()) == digest
    def read(name):
        with (p.OUTPUT / name).open() as stream:
            return list(csv.DictReader(stream))
    train, held, excluded = (read(n) for n in ("train.csv", "holdout.csv", "excluded_picks.csv"))
    keys = [{p.pick_key(r) for r in table} for table in (train, held, excluded)]
    assert not keys[0] & keys[1] and not (keys[0] | keys[1]) & keys[2]
    assert len(set.union(*keys)) == 49158
    quotas = read("split_counts.csv")
    actual = Counter(r["event_id"] for r in held)
    expected = p.allocate_quotas({r["event_id"]: int(r["available_picks"]) for r in quotas})
    assert all(actual[e] == q for e, q in expected.items())
    assert min(Counter(r["event_id"] for r in train).values()) >= 6
    assert sum(float(r["seconds_from_catalog_origin"]) < 0 for r in train + held + excluded) == 2
    references = {r["event_id"]: p.utc(r["training_reference_utc"]) for r in read("event_time_references.csv")}
    assert any(float(r["seconds_from_train_reference"]) < 0 for r in held)
    for row in train + held:
        assert float(row["seconds_from_train_reference"]) == (
            p.utc(row["pick_time_utc"]) - references[row["event_id"]]).total_seconds()
    stations = read("stations.csv")
    assert {r["station"] for r in stations} >= {"AL1", "AL2", "AL6", "HBW", "HER"}
    for station in stations:
        x, y = p._geometry.local_xy_km(float(station["latitude"]), float(station["longitude"]))
        assert float(station["x_m"]) == (x + 10) * 1000 + 3000
        assert float(station["y_m"]) == (y + 3.5) * 1000
        assert float(station["z_m"]) == 1166 - float(station["elevation_m"])
    with np.load(p.OUTPUT / "emtomo/arrivals.npz", allow_pickle=False) as arrays:
        assert arrays["arrival_times_s"].shape == (3471, 31)
        assert np.isfinite(arrays["arrival_times_s"]).sum() == 42973
        assert np.all(np.isfinite(arrays["arrival_times_s"]).sum(axis=1) >= 6)
        assert np.all(np.nanmin(arrays["arrival_times_s"], axis=1) == 0)
        assert np.all(arrays["station_coordinates_m"] >= 0)
        assert np.all(arrays["station_coordinates_m"] <= p.EXTENT)
    before = (p.OUTPUT / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        p.prepare()
    assert (p.OUTPUT / "manifest.json").read_bytes() == before
