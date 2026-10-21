"""Read-only static study and dependency checks; no forward/inverse solves."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / 'runs/checkerboard_scaled_20261021'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((STATE / 'manifest.json').read_text())
        cls.worker = load('scaled_worker', 'worker.py')
        cls.supervisor = load('scaled_supervisor', 'supervise.py')
        cls.api = cls.worker.runtime(cls.manifest)

    def test_snapshot_and_build_unchanged(self):
        self.worker.verify(self.manifest)
        build = Path(self.manifest['binary']).parent
        include = (build / 'include/tomoFDD.inc').read_text()
        import re
        self.assertEqual(int(re.search(r'MAXOBS\s*=\s*(\d+)', include)[1]), 120)
        self.assertEqual(int(re.search(r'MAXSTA\s*=\s*(\d+)', include)[1]), 150)

    def test_exact_scaled_geometry_and_source_models(self):
        load_inputs = self.api[4]
        for case in self.manifest['cases'][:3]:
            model, stations, events = load_inputs(Path(self.manifest['forward_root']) / 'experiments', case['experiment_id'])
            generation = json.loads((Path(case['input_dir']) / 'generation.json').read_text())
            old_id = generation['derived_from']['experiment_id']
            old_model, old_stations, old_events = load_inputs(ROOT.parent / 'forward_modeling/experiments', old_id)
            np.testing.assert_array_equal(model.velocity, old_model.velocity)
            np.testing.assert_array_equal(stations.coordinates_m, old_stations.coordinates_m / 10)
            np.testing.assert_array_equal(events.coordinates_m, old_events.coordinates_m / 10)
            self.assertEqual(model.cell_size_m / case['refinement'], 125)
            self.assertEqual((len(events.ids), len(stations.ids)), (1200, 98))

    def test_tomodd_grid_schedule_and_pair_capacity(self):
        load_inputs, adapter = self.api[4], self.api[3]
        for case in self.manifest['cases'][:3]:
            model, stations, events = load_inputs(Path(self.manifest['forward_root']) / 'experiments', case['experiment_id'])
            # Dummy picks are only in-memory for geometry construction; never saved or inverted.
            data = adapter.ExternalData(model.velocity / 1000, model.cell_size_m / 1000,
                np.array(model.origin_m) / 1000, stations.ids, stations.coordinates_m / 1000,
                events.ids, events.coordinates_m / 1000, np.zeros((1200, 98)), {}, {})
            grid, _ = adapter.build_tomodd_geometry(data, 1., 0)
            self.assertEqual(grid.initial_vp.shape, (14, 14, 26))
            self.assertTrue(np.all(grid.initial_vp == 5.))
            initial, _ = adapter.perturb_initial_event_xyz(data.event_xyz_km, data.lengths_km, 1., 42)
            pairs = adapter.select_event_pairs(initial, 3., 4)
            self.assertLess(117600 + len(pairs) * 98, adapter.MAX_DATA)
        cfg = adapter._solver_config(3, 1.)
        self.assertEqual(cfg.smoothing_lon_lat_depth, (5., 5., 5.))
        self.assertEqual(cfg.iteration_sets, ((3, .1, .08, -9., 10., 40., 1, 0.),))

    def test_dependencies_wait_and_block(self):
        case = next(c for c in self.manifest['cases'] if c['method'] == 'emtomo')
        rows = {c['id']: {'status': 'queued'} for c in self.manifest['cases']}
        self.assertEqual(self.supervisor.dependency_state(case, rows), 'waiting')
        for dependency in case['depends_on']:
            rows[dependency]['status'] = 'completed'
        self.assertEqual(self.supervisor.dependency_state(case, rows), 'ready')
        rows[case['depends_on'][0]]['status'] = 'failed'
        self.assertEqual(self.supervisor.dependency_state(case, rows), 'blocked')


if __name__ == '__main__':
    unittest.main()
