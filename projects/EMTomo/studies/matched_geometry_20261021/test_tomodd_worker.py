"""No solver/build execution: unit guards plus read-only real-data preflights.

Run: python3 -B -m unittest discover -s <this-study> -p test_tomodd_worker.py -v
Integration tests skip when the existing benchmark/forward inputs are unavailable.
All temporary files are under this study and removed after the tests.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
STUDY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('matched_tomodd_worker', STUDY / 'tomodd_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
FORWARD = Path('/mnt/disk01/egor/projects/forward_modeling')


class WorkerUnitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='.tomodd-test-', dir=STUDY)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_existing_empty_directory_and_dangling_link_rejected(self):
        with self.assertRaisesRegex(ValueError, 'already exists'):
            worker.new_output(self.root)
        link = self.root / 'dangling'
        link.symlink_to(self.root / 'absent')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            worker.new_output(link)
        fresh = self.root / 'new'
        self.assertEqual(worker.new_output(fresh), fresh)
        self.assertFalse(fresh.exists())

    def test_missing_parent_rejected(self):
        with self.assertRaisesRegex(ValueError, 'parent must already exist'):
            worker.new_output(self.root / 'absent' / 'new')

    def test_identical_protocol_between_geometries(self):
        values = []
        for geometry in worker.EXPERIMENTS:
            args = argparse.Namespace(geometry=geometry, forward_root=FORWARD, output=self.root / 'new')
            arguments = worker.run_arguments(args, worker.DEFAULT_BUILD / 'tomoDD-SE')
            arguments.pop('experiment_id')
            values.append(arguments)
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[0]['initial_location_sigma_km'], 1.0)
        self.assertEqual(values[0]['initial_location_seed'], 42)
        self.assertEqual(values[0]['iterations'], 3)
        self.assertEqual(values[0]['timeout_s'], 0)
        self.assertEqual(values[0]['free_buffer_layers'], 0)
        self.assertEqual(values[0]['ray_step_km'], 0.05)

    @unittest.skipUnless(worker.LIVE_ROOT.is_dir(), 'live benchmark unavailable')
    def test_live_root_and_escaping_source_rejected(self):
        with self.assertRaisesRegex(ValueError, 'snapshot, not the live'):
            worker.load_adapter(worker.LIVE_ROOT)
        package = self.root / 'tomodd_benchmark'
        package.symlink_to(worker.LIVE_ROOT / 'tomodd_benchmark', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes its root'):
            worker.load_adapter(self.root)

    @unittest.skipUnless(worker.DEFAULT_BUILD.is_dir(), 'existing solver unavailable')
    def test_existing_build_verified_without_subprocess(self):
        with patch.object(subprocess, 'run', side_effect=AssertionError('Must not execute anything')):
            binary, manifest = worker.verify_build(worker.DEFAULT_BUILD)
        self.assertEqual(binary.name, 'tomoDD-SE')
        self.assertEqual(manifest.name, 'manifest.json')

    @unittest.skipUnless(worker.DEFAULT_BUILD.is_dir(), 'existing solver unavailable')
    def test_modified_recipe_rejected(self):
        manifest = json.loads((worker.DEFAULT_BUILD / 'manifest.json').read_text())
        manifest['recipe']['max_events'] = 2000
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'signature mismatch'):
            worker.verify_build(self.root)


@unittest.skipUnless(worker.LIVE_ROOT.is_dir() and FORWARD.is_dir(), 'existing data unavailable')
class ReadOnlyIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='.tomodd-test-', dir=STUDY)
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.snapshot = cls.root / 'snapshot'
        shutil.copytree(worker.LIVE_ROOT / 'tomodd_benchmark', cls.snapshot / 'tomodd_benchmark',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        cls.env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(worker.LIVE_ROOT))

    def test_both_real_data_preflights_use_snapshot_not_pythonpath(self):
        for geometry, (_, events, stations) in worker.EXPERIMENTS.items():
            with self.subTest(geometry=geometry):
                output = self.root / geometry
                process = subprocess.run(
                    [sys.executable, '-B', str(STUDY / 'tomodd_worker.py'), '--geometry', geometry,
                     '--output', str(output), '--benchmark-root', str(self.snapshot),
                     '--forward-root', str(FORWARD), '--validate-only'],
                    env=self.env, capture_output=True, text=True, timeout=120)
                self.assertEqual(process.returncode, 0, process.stderr)
                info = json.loads(process.stdout)
                self.assertEqual(info['state'], 'validated')
                self.assertFalse(info['solver_launched'])
                self.assertEqual((info['events'], info['stations']), (events, stations))
                self.assertTrue(Path(info['adapter']).is_relative_to(self.snapshot))
                self.assertEqual(info['arguments']['initial_location_sigma_km'], 1.0)
                self.assertFalse(output.exists())
                self.assertFalse(list(self.snapshot.rglob('*.pyc')))
                print(f"{geometry}: pairs={info['event_pairs']}, "
                      f"observations={info['absolute_observations'] + info['differential_observations']}")

    def test_completion_checks_against_saved_old_run(self):
        # Exercise the real parser/history filenames, not a guessed mock layout.
        old = worker.LIVE_ROOT / 'runs' / (
            'geysers11_20110401-20110420_flat4_elev1166_26st_3776ev_'
            'tomodd-1km-layerstart-ray005-deg4-sigma0p5km-seed42-3it-coordfix-nolimit')
        if not (old / 'results.json').is_file():
            self.skipTest('saved old results unavailable')
        code = (
            'import importlib.util,json,sys; from pathlib import Path; '
            's=importlib.util.spec_from_file_location("worker",sys.argv[1]); '
            'w=importlib.util.module_from_spec(s); s.loader.exec_module(w); '
            'a=w.load_adapter(Path(sys.argv[2])); p=Path(sys.argv[3]); '
            'print(w.check_completed(p,a,json.loads((p/"results.json").read_text())))'
        )
        cache = self.root / 'cache'
        result = subprocess.run([sys.executable, '-B', '-c', code, str(STUDY / 'tomodd_worker.py'),
                                 str(self.snapshot), str(old)],
                                env=dict(self.env, MPLCONFIGDIR=str(cache), MPLBACKEND='Agg'),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('tomoDD.vel', result.stdout)


if __name__ == '__main__':
    unittest.main()
