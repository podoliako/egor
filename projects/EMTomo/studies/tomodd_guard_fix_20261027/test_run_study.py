"""Focused source-only, frozen provenance, and exact dd1 reuse regressions."""
import copy
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('guard_fix_under_test', HERE / 'run_study.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SourceOnlyTests(unittest.TestCase):
    def setUp(self):
        self.original = (m.FROZEN / 'RaySPDR2.f').read_bytes()
        self.corrected = m.guard.patch_source(self.original)

    def test_only_four_offsets_and_bottom_preserved(self):
        proof = m.patch_proof(self.original, self.corrected)
        self.assertEqual(proof['changed_lines_1based'], [840, 841])
        self.assertEqual(proof['removed_horizontal_offsets'], 4)
        self.assertTrue(proof['bottom_guard_unchanged'])
        self.assertEqual(self.corrected.replace(m.guard.NEW_GUARD, m.guard.OLD_GUARD, 1), self.original)

    def test_existing_causal_source_tests(self):
        self.assertEqual(m.guard.self_tests(self.original)['status'], 'passed')

    def test_reject_unfixed_or_extra_changes(self):
        for bad in (self.original, self.corrected + b'\n',
                    self.corrected.replace(b'if(dp.GE.zn(nz)-1)', b'if(dp.GE.zn(nz)-2)', 1)):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                m.patch_proof(self.original, bad)

    def test_patch_is_not_repeatable_or_ambiguous(self):
        for bad in (self.corrected, self.original + m.guard.OLD_GUARD):
            with self.assertRaises(ValueError):
                m.guard.patch_source(bad)


class BuildProvenanceTests(unittest.TestCase):
    def test_verified_actual_production_build(self):
        built = m.verify_corrected()
        self.assertTrue(built['production_inversion'])
        self.assertTrue(built['original_relink_byte_identical'])
        self.assertEqual(m.digest(m.BUILD / 'original-tomoDD-SE'), m.worker.BINARY_SHA256)
        self.assertNotEqual(m.digest(m.BUILD / 'tomoDD-SE'), m.worker.BINARY_SHA256)
        self.assertEqual(len([c for c in built['commands'] if '-c' in c]), 1)
        self.assertIn(str(m.BUILD / 'RaySPDR2.f'), built['commands'][0])

    def test_frozen_makefile_order_and_only_ray_substitution(self):
        frozen = m.frozen_contract()
        _, _, ordered = m.link_plan(frozen)
        built = m.read(m.BUILD / 'manifest.json')
        self.assertEqual(ordered, built['link_object_order'])
        new = [arg for arg in built['commands'][2] if arg.endswith('.o')]
        expected = [str((m.BUILD if name == 'RaySPDR2.o' else m.FROZEN) / name) for name in ordered]
        self.assertEqual(new, expected)
        self.assertEqual(set(built['unchanged_link_object_sha256']), set(ordered) - {'RaySPDR2.o'})

    def test_reject_corrected_artifact_manifest_tampering(self):
        actual_read = m.read
        for key in ('tomoDD-SE', 'RaySPDR2.o', 'RaySPDR2.f'):
            def altered(path):
                value = actual_read(path)
                if path == m.BUILD / 'manifest.json':
                    value['artifact_sha256'][key] = '0' * 64
                return value
            with self.subTest(key=key), patch.object(m, 'read', side_effect=altered):
                with self.assertRaisesRegex(ValueError, 'artifact changed'):
                    m.verify_corrected()

    def test_reject_compiler_provenance_tampering(self):
        actual_read = m.read
        def altered(path):
            value = actual_read(path)
            if path == m.BUILD / 'manifest.json':
                value['compiler']['sha256'] = '0' * 64
            return value
        with patch.object(m, 'read', side_effect=altered):
            with self.assertRaisesRegex(ValueError, 'Compiler provenance'):
                m.verify_corrected()


class InputReuseTests(unittest.TestCase):
    def setUp(self):
        self.previous_baseline = m.shared.BASELINE
        self.geometry = 'synthetic'
        m.shared.BASELINE = m.PROJECT / 'runs/matched_geometry_20261021/outputs/synthetic-tomodd-sigma1'
        self.reference = m.read(m.REFERENCES[self.geometry] / 'worker_info.json')
        self.work = m.REFERENCES[self.geometry] / 'work/inversion'
        self.control = (self.work / 'tomoDD-SE.inp').read_text()

    def tearDown(self):
        m.shared.BASELINE = self.previous_baseline

    def test_actual_dd1_input_and_control_reuse_both_geometries(self):
        for geometry in m.GEOMETRIES:
            m.shared.BASELINE = m.PROJECT / f'runs/matched_geometry_20261021/outputs/{geometry}-tomodd-sigma1'
            reference = m.read(m.REFERENCES[geometry] / 'worker_info.json')
            work = m.REFERENCES[geometry] / 'work/inversion'
            proof = m.check_dd1_inputs(work, (work / 'tomoDD-SE.inp').read_text(), geometry, reference)
            self.assertTrue(proof['all_inputs_byte_identical'])
            self.assertTrue(proof['control_byte_identical'])
            m.check_reference(reference, geometry)

    def test_live_preflight_tuple_arguments_match_json_reference(self):
        live = copy.deepcopy(self.reference)
        live['arguments']['initial_layer_boundaries_km'] = m.worker.BOUNDARIES
        live['arguments']['initial_layer_velocities_m_s'] = m.worker.VELOCITIES
        live['arguments']['run_dir'] = str(m.ROOT / 'validation/synthetic-dd1-d40-s5')
        live['arguments']['timeout_s'] = 0
        m.check_reference(live, self.geometry)

    def test_restore_only_reference_schedule_whitespace(self):
        for geometry in m.GEOMETRIES:
            m.shared.BASELINE = m.PROJECT / f'runs/matched_geometry_20261021/outputs/{geometry}-tomodd-sigma1'
            original = (m.REFERENCES[geometry] / 'work/inversion/tomoDD-SE.inp').read_text()
            compact = m.shared.replace_data_line(original, m.shared.SCHEDULE_MARKER,
                                                  lambda line: ' '.join(line.split()) + '\n')
            self.assertEqual(m.reference_control_layout(compact, geometry), original)
            changed = m.shared.replace_data_line(compact, m.shared.SCHEDULE_MARKER,
                                                  lambda line: line.replace('40', '80'))
            with self.assertRaisesRegex(ValueError, 'schedule tokens differ'):
                m.reference_control_layout(changed, geometry)
            with self.assertRaises(ValueError):
                m.reference_control_layout(compact + '\n', geometry)

    def test_reject_control_change(self):
        with self.assertRaisesRegex(ValueError, 'control bytes differ'):
            m.check_dd1_inputs(self.work, self.control + '\n', self.geometry, self.reference)

    def test_reject_generated_input_change(self):
        with tempfile.TemporaryDirectory(prefix='.input-test-', dir=HERE) as temp:
            folder = Path(temp)
            for name in m.shared.CONSTANT_FILES:
                shutil.copyfile(self.work / name, folder / name)
            with (folder / 'event.dat').open('ab') as stream:
                stream.write(b'\n')
            with self.assertRaisesRegex(ValueError, 'input bytes differ: event.dat'):
                m.check_dd1_inputs(folder, self.control, self.geometry, self.reference)

    def test_reject_old_reference_provenance_change(self):
        for key in m.shared.PROVENANCE_KEYS:
            altered = copy.deepcopy(self.reference)
            altered[key] = 'tampered'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'provenance differs'):
                m.check_reference(altered, self.geometry)

    def test_reject_unrelated_argument_change(self):
        altered = copy.deepcopy(self.reference)
        altered['arguments']['initial_location_seed'] = 43
        with self.assertRaisesRegex(ValueError, 'input/run parameters differ'):
            m.check_reference(altered, self.geometry)

    def test_exact_joint_schedule(self):
        self.assertEqual(m.shared.solver_schedule(m.CASE), ((3, 1., .08, -9., 1., 40., 1, 0.),))
        self.assertEqual(m.CASE['smoothing'], [5., 5., 5.])
        m.shared.check_control(self.control, m.CASE)


if __name__ == '__main__':
    unittest.main(verbosity=2)
