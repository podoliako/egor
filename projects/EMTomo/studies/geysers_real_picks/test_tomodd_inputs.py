"""In-memory adapter tests: no real pick files, downloads, native runs, or outputs.

Run from EMTomo: python3 -B -m unittest studies.geysers_real_picks.test_tomodd_inputs -v
The optional snapshot cross-check reads Python source and extracts pure functions
with ast; it never imports the synthetic adapter or executes a study entrypoint.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from studies.geysers_real_picks import tomodd_inputs as adapter

PROJECT = Path(__file__).resolve().parents[2]
SNAPSHOT = PROJECT / "runs/matched_geometry_20261021/snapshot/benchmark/tomodd_benchmark"


def snapshot_function(filename, function_name, namespace):
    path = SNAPSHOT / filename
    if not path.is_file():
        raise unittest.SkipTest("Optional matched-geometry snapshot is unavailable")
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[function_name]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.domain = adapter.Domain(38.80275, -122.77070, 135.0, (20., 7., 7.), 1166.)
        self.model = adapter.LayerModel((.7, 1.9, 3.4), (3.8, 4.6, 5.2, 5.7))
        self.origin = datetime(2011, 4, 2, tzinfo=timezone.utc)
        self.initial = {3156432: [10., 3.5, 2.], 3156433: [10.4, 3.7, 2.3]}
        self.events = []
        for index, (event, xyz) in enumerate(self.initial.items()):
            lat, lon, depth = self.domain.geographic(xyz)
            self.events.append(dict(event_id=str(event), latitude=lat, longitude=lon, depth_km=depth,
                                    origin_time_utc=self.origin + timedelta(seconds=index * 2)))
        self.stations = []
        for station, xyz in zip(("STY", "NEG", "THIRD"), ([9., 3., .1], [11., 4., .3], [12., 4., .5])):
            lat, lon, depth = self.domain.geographic(xyz)
            self.stations.append(dict(network="BG", station=station, latitude=lat, longitude=lon,
                                      elevation_m=-depth * 1000))
        self.picks = []
        for event, station, relative in ((3156432, "STY", -.13), (3156432, "NEG", -.01),
                                          (3156433, "STY", 1.2), (3156433, "NEG", 1.8),
                                          (3156433, "THIRD", 2.0)):
            origin = self.events[event - 3156432]["origin_time_utc"]
            self.picks.append(dict(event_id=event, network="BG", station=station, phase="P",
                                   pick_time_utc=origin + timedelta(seconds=relative), weight_code="7"))

    def build(self, **kwargs):
        options = dict(domain=self.domain, model=self.model, initial_xyz_km=self.initial)
        options.update(kwargs)
        return adapter.build_inputs(self.picks, self.events, self.stations, **options)

    def relative_rows(self, text):
        result = {}
        current = None
        for line in text.splitlines():
            fields = line.split()
            if fields[0] == "#":
                current = tuple(map(int, fields[1:]))
            else:
                result[current, fields[0]] = list(map(float, fields[1:-1]))
        return result

    def test_native_schema_and_sparse_counts(self):
        bundle = self.build()
        self.assertEqual(set(bundle.files), {"phase.dat", "ph2dt.inp", "event.dat", "station.dat", "MOD",
                                            "grid.dat", "absolute.dat", "dt.ct", "dt.cc", "tomoDD-SE.inp"})
        self.assertEqual(bundle.manifest["absolute_observations"], 5)
        self.assertEqual(bundle.manifest["differential_observations"], 2)
        self.assertEqual(bundle.files["dt.cc"], "")
        self.assertEqual(len(bundle.files["event.dat"].splitlines()), 2)
        for line in bundle.files["event.dat"].splitlines():
            fields = line.split()
            self.assertEqual(len(fields), 11)
            self.assertEqual(fields[-1], "0")
            self.assertEqual(len(fields[0]), 8)
            self.assertEqual(len(fields[1]), 8)
        for line in bundle.files["phase.dat"].splitlines():
            fields = line.split()
            self.assertEqual(len(fields), 15 if fields[0] == "#" else 4)
        self.assertEqual([len(line.split()) for line in bundle.files["dt.ct"].splitlines()], [3, 5, 5])
        ph2dt = [line for line in bundle.files["ph2dt.inp"].splitlines() if not line.startswith("*")]
        self.assertEqual(ph2dt[:2], ["station.dat", "phase.dat"])
        self.assertEqual(list(map(float, ph2dt[2].split())), [0, 1000, 3, 4, 1, 1, 60, 0])

    def test_negative_catalog_picks_preserved(self):
        bundle = self.build()
        rows = self.relative_rows(bundle.files["absolute.dat"])
        negative = sorted(values[0] for (event, _), values in rows.items() if event == (3156432,))
        self.assertEqual(negative, [-.13, -.01])
        self.assertIn("-0.130000", bundle.files["phase.dat"])
        self.assertIn("-0.010000", bundle.files["phase.dat"])
        self.assertEqual(bundle.manifest["common_origin_shift_s"], 0)

    def test_common_shift_preserves_utc_and_differences_across_day(self):
        plain = self.build()
        shifted = self.build(origin_shift_s=1)
        old = self.relative_rows(plain.files["dt.ct"])
        new = self.relative_rows(shifted.files["dt.ct"])
        for key in old:
            self.assertAlmostEqual(old[key][0] - old[key][1], new[key][0] - new[key][1])
        self.assertTrue(all(v[0] >= 0 for v in self.relative_rows(shifted.files["absolute.dat"]).values()))
        self.assertIn("20110401 23595900", shifted.files["event.dat"])
        self.assert_utc_reconstruction(shifted)
        self.assertEqual(plain.manifest["training_picks"], shifted.manifest["training_picks"])

    def assert_utc_reconstruction(self, bundle):
        origins = {e["event_id"]: adapter._utc(e["export_origin_time_utc"]) for e in bundle.manifest["events"]}
        rows = self.relative_rows(bundle.files["absolute.dat"])
        for pick in bundle.manifest["training_picks"]:
            seconds = rows[(pick["event_id"],), pick["station_alias"]][0]
            reconstructed = origins[pick["event_id"]] + timedelta(seconds=seconds)
            self.assertEqual(reconstructed, adapter._utc(pick["pick_time_utc"]))

    def test_subcentisecond_origin_quantization_and_microsecond_pick(self):
        self.events[0]["origin_time_utc"] = self.origin + timedelta(microseconds=123456)
        self.picks[0]["pick_time_utc"] += timedelta(microseconds=123)
        bundle = self.build()
        self.assertAlmostEqual(bundle.manifest["events"][0]["effective_backward_shift_s"], .003456)
        self.assert_utc_reconstruction(bundle)

    def test_same_station_difference_uses_relative_not_absolute_utc(self):
        bundle = self.build()
        rows = self.relative_rows(bundle.files["dt.ct"])
        differences = sorted(v[0] - v[1] for v in rows.values())
        np.testing.assert_allclose(differences, [-1.81, -1.33])

    def test_geographic_rotation_and_depth_roundtrip(self):
        center = self.domain.local(38.80275, -122.77070, .834)
        np.testing.assert_allclose(center, [10, 3.5, 2])
        north = self.domain.local(38.80275 + 1 / 110.574, -122.77070, .834)
        np.testing.assert_allclose(north - center, [-2**-.5, -2**-.5, 0], atol=1e-10)
        east = self.domain.local(38.80275, -122.77070 + 1 / self.domain.lon_scale, .834)
        np.testing.assert_allclose(east - center, [2**-.5, -2**-.5, 0], atol=1e-10)
        for point in ([0., 0., 0.], [20., 7., 7.], [3., 4., 2.]):
            np.testing.assert_allclose(self.domain.local(*self.domain.geographic(point)), point, atol=1e-9)

    def shifted_domain(self):
        return adapter.Domain(38.80275, -122.77070, 135., (23., 11., 7.), 1166.,
                              horizontal_offset_km=(13., 3.5))

    def test_default_offset_preserves_midpoint_behavior(self):
        explicit = adapter.Domain(38.80275, -122.77070, 135., (20., 7., 7.), 1166.,
                                  horizontal_offset_km=(10., 3.5))
        self.assertEqual(self.domain.effective_horizontal_offset_km, (10., 3.5))
        np.testing.assert_array_equal(self.domain.local(38.81, -122.79, 2),
                                      explicit.local(38.81, -122.79, 2))
        self.assertEqual(self.build().files, self.build(domain=explicit).files)

    def test_shifted_domain_roundtrip_and_old_x_plus_three(self):
        domain = self.shifted_domain()
        np.testing.assert_allclose(domain.local(38.80275, -122.77070, .834), [13, 3.5, 2])
        for xyz in ([0., 0., 0.], [23., 11., 7.], [0., 11., 0.], [23., 0., 7.], [13., 3.5, 2.]):
            with self.subTest(xyz=xyz):
                geographic = domain.geographic(xyz)
                np.testing.assert_allclose(domain.local(*geographic), xyz, atol=1e-9)
                np.testing.assert_allclose(self.domain.local(*geographic), np.array(xyz) - [3, 0, 0], atol=1e-9)
        request_domain = adapter.Domain(38.80275, -122.77070, 135., [23., 11., 7.], 1166.,
                                        horizontal_offset_km=[13., 3.5])
        self.assertEqual(request_domain.horizontal_offset_km, (13., 3.5))

    def test_shifted_domain_preserves_native_events_stations_and_times(self):
        before = self.build()
        initial = {event: np.array(xyz) + [3, 0, 0] for event, xyz in self.initial.items()}
        after = self.build(domain=self.shifted_domain(), initial_xyz_km=initial)
        for name in ("phase.dat", "event.dat", "station.dat", "absolute.dat", "dt.ct", "tomoDD-SE.inp"):
            self.assertEqual(before.files[name], after.files[name], name)
        self.assertEqual(after.manifest["domain"]["horizontal_offset_km"], (13., 3.5))
        self.assert_utc_reconstruction(after)

    def test_shifted_native_grid_encloses_translated_box(self):
        domain = self.shifted_domain()
        files, shape = adapter._model_files(domain, self.model)
        self.assertEqual(shape, [27, 27, 9])
        lon, lat, depths = [np.array(list(map(float, line.split()))) for line in files["MOD"].splitlines()[1:4]]
        east = (lon - domain.center_longitude) * domain.lon_scale
        north = (lat - domain.center_latitude) * 110.574
        # Independent inverse of x=u+13, y=v+3.5 at the box midpoint.
        expected_east = (-1.5 - 2.) / math.sqrt(2)
        expected_north = (1.5 - 2.) / math.sqrt(2)
        self.assertAlmostEqual((east[1] + east[-2]) / 2, expected_east, places=6)
        self.assertAlmostEqual((north[1] + north[-2]) / 2, expected_north, places=6)
        for x in (0, 23):
            for y in (0, 11):
                lat_corner, lon_corner, _ = self.domain.geographic([x - 3, y, 0])
                e = (lon_corner - domain.center_longitude) * domain.lon_scale
                n = (lat_corner - domain.center_latitude) * 110.574
                self.assertTrue(east[1] - .5 <= e <= east[-2] + .5)
                self.assertTrue(north[1] - .5 <= n <= north[-2] + .5)
                self.assertTrue(lon[0] < lon_corner < lon[-1])
                self.assertTrue(lat[0] < lat_corner < lat[-1])
        np.testing.assert_allclose(np.diff(east)[1:-1], 1, atol=1e-6)
        np.testing.assert_allclose(np.diff(north)[1:-1], 1, atol=2e-6)
        np.testing.assert_allclose(depths[1:-1], np.arange(7) + .5 - 1.166)
        self.assertIn("38.80275 -122.7707", files["grid.dat"])

    def test_her_al2_bounds_in_shifted_domain_and_native_grid(self):
        # Read-only source checked during implementation: data/geysers_2011/
        # bg_stations_2011.txt, BG rows valid in 2011; no runtime data loading.
        rows = [("HER", 38.844761, -122.913803, 668., [.93684664, 8.99367271, .498]),
                ("AL2", 38.816006, -122.897995, 657., [4.154847, 10.27224268, .509])]
        domain = self.shifted_domain()
        wrong_midpoint = adapter.Domain(38.80275, -122.77070, 135., (23., 11., 7.), 1166.)
        for name, latitude, longitude, elevation, expected in rows:
            xyz = domain.local(latitude, longitude, -elevation / 1000)
            np.testing.assert_allclose(xyz, expected, atol=1e-8)
            domain.check_inside(xyz)
            with self.assertRaisesRegex(ValueError, "outside the physical domain"):
                wrong_midpoint.check_inside(wrong_midpoint.local(latitude, longitude, -elevation / 1000))
            self.stations.append(dict(network="BG", station=name, latitude=latitude,
                                      longitude=longitude, elevation_m=elevation))
            self.picks.append(dict(self.picks[0], station=name))
        initial = {event: np.array(xyz) + [3, 0, 0] for event, xyz in self.initial.items()}
        bundle = self.build(domain=domain, initial_xyz_km=initial)
        lon, lat, depths = [np.array(list(map(float, line.split()))) for line in bundle.files["MOD"].splitlines()[1:4]]
        for line in bundle.files["station.dat"].splitlines():
            _, latitude, longitude, elevation = line.split()
            self.assertTrue(lon[0] < float(longitude) < lon[-1])
            self.assertTrue(lat[0] < float(latitude) < lat[-1])
            self.assertTrue(depths[0] < -float(elevation) / 1000 < depths[-1] - 1)

    def test_offset_validation_and_extreme_geometry(self):
        for offset in ((1,), (1, 2, 3), (float("nan"), 0), (0, float("inf"))):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                adapter.Domain(38.80275, -122.77070, 135., (23, 11, 7), 1166., offset)
        for lengths in ((89, 7, 7), (20, 79, 7), (20, 7, 34)):
            domain = adapter.Domain(38.80275, -122.77070, 90., lengths, 1166.)
            with self.subTest(lengths=lengths), self.assertRaisesRegex(ValueError, "maxnx/maxny/maxnz"):
                adapter._model_files(domain, self.model)
        huge = adapter.Domain(38.80275, -122.77070, 135., (1.7e308, 1.7e308, 7), 1166.)
        with self.assertRaisesRegex(ValueError, "finite"):
            adapter._model_files(huge, self.model)
        remote = adapter.Domain(38.80275, -122.77070, 135., (23, 11, 7), 1166., (1e9, 0))
        with self.assertRaisesRegex(ValueError, "geographic boundary"):
            adapter._model_files(remote, self.model)
        deep = adapter.Domain(38.80275, -122.77070, 135., (23, 11, 7), -1e14, (13, 3.5))
        with self.assertRaisesRegex(ValueError, "collapse in native REAL"):
            adapter._model_files(deep, self.model)
        dateline = adapter.Domain(38.80275, 179.99, 135., (23, 11, 7), 1166., (13, 3.5))
        with self.assertRaisesRegex(ValueError, "geographic boundary"):
            adapter._model_files(dateline, self.model)

    def test_real_geography_and_station_elevations_not_artificial_embedding(self):
        bundle = self.build()
        fields = bundle.files["event.dat"].splitlines()[0].split()
        np.testing.assert_allclose(list(map(float, fields[2:5])), [38.80275, -122.77070, .834], atol=1e-8)
        aliases = {s["alias"]: s["station"] for s in bundle.manifest["stations"]}
        source = {s["station"]: s for s in self.stations}
        for line in bundle.files["station.dat"].splitlines():
            alias, lat, lon, elev = line.split()
            self.assertAlmostEqual(float(elev), source[aliases[alias]]["elevation_m"], places=6)
            self.assertAlmostEqual(float(lat), source[aliases[alias]]["latitude"], places=7)
            self.assertAlmostEqual(float(lon), source[aliases[alias]]["longitude"], places=7)

    def test_mod_dimensions_order_padding_datum_and_layers(self):
        bundle = self.build()
        lines = bundle.files["MOD"].splitlines()
        bld, nx, ny, nz = lines[0].split()
        nx, ny, nz = map(int, (nx, ny, nz))
        self.assertEqual(float(bld), .001)
        self.assertEqual([nx, ny, nz], [22, 22, 9])
        lon, lat, depths = [np.array(list(map(float, line.split()))) for line in lines[1:4]]
        self.assertEqual(len(lon), nx)
        self.assertEqual(len(lat), ny)
        np.testing.assert_allclose(depths[1:-1], np.arange(7) + .5 - 1.166)
        self.assertAlmostEqual(depths[-1] - depths[-2], 12)
        self.assertEqual(len(lines), 4 + 2 * nz * ny)
        vp = np.array([list(map(float, line.split())) for line in lines[4:4 + nz * ny]]).reshape(nz, ny, nx)
        expected = np.array(self.model.vp_km_s)[np.searchsorted(self.model.boundaries_km, depths + 1.166, side="right")]
        np.testing.assert_allclose(vp[:, 0, 0], expected)
        np.testing.assert_allclose(vp, np.broadcast_to(expected[:, None, None], vp.shape))
        self.assertTrue(all(float(v) == 1.73 for line in lines[4 + nz * ny:] for v in line.split()))
        for event in bundle.files["event.dat"].splitlines():
            f = event.split()
            self.assertTrue(lon[0] < float(f[3]) < lon[-1])
            self.assertTrue(lat[0] < float(f[2]) < lat[-1])
            self.assertTrue(float(f[4]) < depths[-1] - 1)  # unchanged bottom guard

    def test_recipe_control(self):
        text = self.build().files["tomoDD-SE.inp"]
        self.assertIn("\n" + adapter.SCHEDULE + "\n", text)
        self.assertIn("* weight1 weight2 weight3 CC_format\n5 5 5 1", text)
        self.assertIn("* iuses iuseq invdel stepl\n1 0 0 0.05", text)
        self.assertIn("* IDAT IPHA DIST\n2 1 1000", text)
        self.assertIn("* source parameter information:\n\n", text)
        self.assertTrue(all(len(line) <= 80 for line in text.splitlines() if not line.startswith("*")))

    def test_catalog_superset_cannot_add_observations_or_change_noise(self):
        before = self.build(initial_xyz_km=None, sigma_km=.5, seed=42)
        extra = dict(self.events[0], event_id=999, origin_time_utc="not read for unused event")
        self.events.append(extra)
        self.stations.append(dict(self.stations[0], station="UNUSED"))
        after = self.build(initial_xyz_km=None, sigma_km=.5, seed=42)
        self.assertEqual(before, after)
        self.assertNotIn("999", after.files["event.dat"])

    def test_single_training_pick_never_fills_missing_cells(self):
        self.picks = self.picks[:1]
        bundle = self.build(initial_xyz_km={3156432: self.initial[3156432]})
        self.assertEqual(bundle.manifest["absolute_observations"], 1)
        self.assertEqual(bundle.manifest["differential_observations"], 0)
        self.assertEqual(bundle.manifest["isolated_event_ids"], [3156432])
        self.assertEqual(bundle.files["dt.ct"], "")
        self.assertEqual(len(bundle.files["station.dat"].splitlines()), 1)

    def test_no_common_training_station_has_no_pair(self):
        self.picks = [self.picks[0], self.picks[-1]]
        bundle = self.build()
        self.assertEqual(bundle.files["dt.ct"], "")
        self.assertEqual(bundle.manifest["isolated_event_ids"], [3156432, 3156433])

    def test_pair_degree_and_radius(self):
        xyz = np.array([[i * .1, 0., 0.] for i in range(12)] + [[10., 0., 0.]])
        pairs, degree = adapter._pairs(xyz, [{"same": 1}] * len(xyz))
        self.assertLessEqual(max(degree), 4)
        self.assertEqual(degree[-1], 0)
        self.assertEqual(len(set(pairs)), len(pairs))
        self.assertTrue(all(np.linalg.norm(xyz[a] - xyz[b]) <= 3 for a, b in pairs))

    def test_explicit_sigma_seed_required_and_exclusive(self):
        for options in ({"initial_xyz_km": None}, {"initial_xyz_km": None, "sigma_km": 1},
                        {"initial_xyz_km": None, "seed": 42}, {"sigma_km": 1, "seed": 42}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "explicit sigma_km"):
                self.build(**options)

    def test_sigma_zero_and_seeded_repeatability(self):
        zero = self.build(initial_xyz_km=None, sigma_km=0, seed=42)
        for row in zero.manifest["events"]:
            np.testing.assert_allclose(row["initial_xyz_km"], row["catalog_xyz_km"])
        a = self.build(initial_xyz_km=None, sigma_km=1, seed=42)
        b = self.build(initial_xyz_km=None, sigma_km=1, seed=42)
        self.assertEqual(a, b)
        c = self.build(initial_xyz_km=None, sigma_km=1, seed=43)
        self.assertNotEqual(a.files["event.dat"], c.files["event.dat"])
        self.assertEqual(a.files["absolute.dat"], c.files["absolute.dat"])

    def test_noise_matches_old_recipe_function(self):
        old = snapshot_function("external_checkerboard.py", "perturb_initial_event_xyz", {"np": np})
        xyz = np.array([[.01, 3, .01], [19.99, 6.99, 6.99], [10, 3, 3.]])
        expected, _ = old(xyz, np.array(self.domain.lengths_km), 1, 42)
        actual = adapter._perturb(xyz, np.array(self.domain.lengths_km), 1, 42)
        np.testing.assert_array_equal(actual, expected)

    def test_complete_pair_graph_matches_old_recipe_function(self):
        old = snapshot_function("external_checkerboard.py", "select_event_pairs", {"np": np})
        xyz = np.random.default_rng(3).uniform(0, 1, size=(20, 3))
        expected = old(xyz, 3., 4)
        actual, _ = adapter._pairs(xyz, [{"station": 1}] * len(xyz))
        self.assertEqual(actual, expected)

    def test_control_data_tokens_match_snapshot_renderer(self):
        old = snapshot_function("runner.py", "_input_text", {"math": __import__("math")})
        cfg = SimpleNamespace(center_lat=self.domain.center_latitude, center_lon=self.domain.center_longitude)
        solver = SimpleNamespace(smoothing_lon_lat_depth=(5., 5., 5.),
                                 iteration_sets=((3, 1., .08, -9., 1., 40., 1, 0.),))
        reference = old(cfg, False, solver, .05)
        def data(text):
            return [line.split() for line in text.splitlines() if not line.startswith("*")]
        expected, actual = data(reference), data(self.build().files["tomoDD-SE.inp"])
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            try:
                self.assertEqual(list(map(float, a)), list(map(float, b)))
            except ValueError:
                self.assertEqual(a, b)

    def test_input_order_determinism_and_no_mutation(self):
        original = deepcopy((self.picks, self.events, self.stations, self.initial))
        before = self.build()
        self.assertEqual(original, (self.picks, self.events, self.stations, self.initial))
        self.picks.reverse()
        self.events.reverse()
        self.stations.reverse()
        self.assertEqual(before, self.build())

    def test_raw_quality_codes_not_interpreted_and_explicit_weights(self):
        self.picks[0]["weight_code"] = "5"
        plain = self.build()
        self.assertTrue(all(p["weight"] == 1 for p in plain.manifest["training_picks"]))
        self.picks[0]["weight"] = .5
        weighted = self.build()
        rows = self.relative_rows(weighted.files["dt.ct"])
        self.assertIn(.75, [values[2] for values in rows.values()])

    def test_small_explicit_weight_not_rounded_to_zero(self):
        self.picks[0]["weight"] = 1e-9
        rows = self.relative_rows(self.build().files["absolute.dat"])
        self.assertIn(1e-9, [value[1] for value in rows.values()])

    def test_native_header_line_overflow_rejected(self):
        self.events[0]["magnitude"] = 1e300
        with self.assertRaisesRegex(ValueError, "exceeds native line"):
            self.build()

    def test_aliases_distinguish_networks_with_same_station_code(self):
        self.stations.append(dict(self.stations[0], network="XX"))
        self.picks.append(dict(self.picks[0], network="XX"))
        bundle = self.build()
        aliases = [row["alias"] for row in bundle.manifest["stations"]]
        self.assertEqual(len(aliases), len(set(aliases)))
        self.assertTrue(all(len(alias) == 7 for alias in aliases))
        self.assertEqual(len(aliases), 4)

    def test_duplicate_pick_even_different_channel_rejected(self):
        self.picks.append(dict(self.picks[0], channel="OTHER"))
        with self.assertRaisesRegex(ValueError, "Duplicate training"):
            self.build()

    def test_duplicate_catalog_and_station_rejected(self):
        self.events.append(dict(self.events[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate catalog"):
            self.build()
        self.events.pop()
        self.stations.append(dict(self.stations[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate station"):
            self.build()

    def test_unknown_event_station_and_non_p_phase_rejected(self):
        for field, value in (("event_id", 1), ("station", "MISSING"), ("phase", "S")):
            original = self.picks[0][field]
            self.picks[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.build()
            self.picks[0][field] = original

    def test_bad_sigma_seed_shift_weight_rejected(self):
        for sigma, seed in ((-1, 42), (float("nan"), 42), (1, None), (1, -1), (1, 4.2)):
            with self.subTest(sigma=sigma, seed=seed), self.assertRaises(ValueError):
                self.build(initial_xyz_km=None, sigma_km=sigma, seed=seed)
        for shift in (-1, .001, float("nan")):
            with self.subTest(shift=shift), self.assertRaises(ValueError):
                self.build(origin_shift_s=shift)
        for weight in (0, -1, 7, float("nan")):
            self.picks[0]["weight"] = weight
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                self.build()

    def test_naive_timestamp_rejected(self):
        self.picks[0]["pick_time_utc"] = "2011-04-02T00:00:00"
        with self.assertRaisesRegex(ValueError, "explicit timezone"):
            self.build()

    def test_initial_xyz_coverage_nonfinite_and_bounds(self):
        for initial in ({3156432: [10, 3, 2]}, dict(self.initial, extra=[0, 0, 0]),
                        {3156432: [-1, 0, 0], 3156433: [10, 3, 2]},
                        {3156432: [10, 3, float("nan")], 3156433: [10, 3, 2]}):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                self.build(initial_xyz_km=initial)

    def test_station_above_top_rejected_not_flattened(self):
        self.stations[0]["elevation_m"] = 2000
        with self.assertRaisesRegex(ValueError, "outside the physical domain"):
            self.build()

    def test_invalid_model_and_domain(self):
        with self.assertRaises(ValueError):
            adapter.LayerModel((2, 1), (3, 4, 5))
        with self.assertRaises(ValueError):
            adapter.LayerModel((), (-1,))
        with self.assertRaises(ValueError):
            self.build(model=adapter.LayerModel((8,), (4, 5)))
        with self.assertRaises(ValueError):
            adapter.Domain(38, -122, 135, (20, 7, 7), 5000)
        with self.assertRaises(ValueError):
            self.build(domain=adapter.Domain(38.80275, -122.77070, 135, (200, 200, 7), 1166))

    def test_no_files_or_processes_used_to_build(self):
        with patch("pathlib.Path.read_text", side_effect=AssertionError("no loading")), \
             patch("pathlib.Path.write_text", side_effect=AssertionError("no writing")), \
             patch("builtins.open", side_effect=AssertionError("no loading")), \
             patch("subprocess.Popen", side_effect=AssertionError("no solver")):
            self.build()

    def test_manifest_hashes_and_no_solver_claim(self):
        bundle = self.build()
        self.assertFalse(bundle.manifest["solver_launched"])
        self.assertTrue(bundle.manifest["requires_corrected_horizontal_guard"])
        for name, text in bundle.files.items():
            self.assertEqual(bundle.manifest["file_sha256"][name], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(bundle.manifest["missing_support_files"], list(adapter.SUPPORT_FILES))

    def test_writer_exclusive_and_support_manifest_without_disk_writes(self):
        bundle = self.build()
        support = {name: b"explicit caller bytes\n" for name in adapter.SUPPORT_FILES}
        with patch.object(Path, "mkdir") as mkdir, patch.object(Path, "write_bytes") as write, \
             patch.object(Path, "write_text") as manifest_write:
            manifest = adapter.write_bundle(bundle, Path("unused-test-output"), support_files=support)
        mkdir.assert_called_once_with(exist_ok=False)
        self.assertEqual(write.call_count, len(bundle.files) + 2)
        manifest_write.assert_called_once()
        self.assertEqual(manifest["missing_support_files"], [])
        self.assertEqual(bundle.manifest["missing_support_files"], list(adapter.SUPPORT_FILES))
        for name, data in support.items():
            self.assertEqual(manifest["file_sha256"][name], hashlib.sha256(data).hexdigest())

    def test_writer_existing_output_refused_before_write(self):
        bundle = self.build()
        with patch.object(Path, "mkdir", side_effect=FileExistsError), patch.object(Path, "write_bytes") as write:
            with self.assertRaises(FileExistsError):
                adapter.write_bundle(bundle, "unused-test-output")
        write.assert_not_called()

    def test_writer_rejects_partial_support_and_path_traversal(self):
        bundle = self.build()
        with self.assertRaises(ValueError):
            adapter.write_bundle(bundle, "unused-test-output", support_files={adapter.SUPPORT_FILES[0]: b"x"})
        with self.assertRaises(ValueError):
            adapter.write_bundle(adapter.Bundle({"../source": "x"}, bundle.manifest), "unused-test-output")


if __name__ == "__main__":
    unittest.main()
