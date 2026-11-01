"""Focused, solver-free tests: python3 -B -m unittest discover -s <study> -v."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sensitivity_study', HERE / 'run_study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def frozen_renderer(cfg, *, synthetic, solver, ray_step_km):
    n, p, s, clip, dd, damp, joint, threshold = solver.iteration_sets[0]
    smooth = solver.smoothing_lon_lat_depth
    return ('* IDAT IPHA DIST\n   2      1     1000\n'
            f'* ray\n{ray_step_km:g}\n' + study.SMOOTH_MARKER + '\n'
            f'  {smooth[0]:g}      {smooth[1]:g}      {smooth[2]:g}       1\n'
            + study.SCHEDULE_MARKER + '\n'
            f'  {n:<2d}     -9    -9    -9   -9   {p:.2f}  {s:.2f}    '
            f'{clip:g}  -9     {dd:g}   {damp:g}   {joint:d}    {threshold:g}\n* CID\n0\n')


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tomodd-unit-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline = self.root / 'baseline'
        self.inversion = self.baseline / 'work/inversion'
        self.inversion.mkdir(parents=True)
        default = SimpleNamespace(iteration_sets=study.worker.SCHEDULE,
                                  smoothing_lon_lat_depth=(5., 5., 5.))
        (self.inversion / 'tomoDD-SE.inp').write_text(
            frozen_renderer(None, synthetic=False, solver=default, ray_step_km=.05))
        for name in study.CONSTANT_FILES:
            (self.inversion / name).write_bytes(('unchanged-' + name).encode())
        self.addCleanup(patch.stopall)
        patch.object(study, 'BASELINE', self.baseline).start()

    def adapter(self):
        return SimpleNamespace(SolverConfig=lambda **kw: SimpleNamespace(**kw),
                               _input_text=frozen_renderer)

    def test_exact_cases_and_serialized_effective_weights(self):
        self.assertEqual([(c['id'], c['dd_weight'], c['damping'], c['smoothing'][0])
                          for c in study.CASES], [
                              ('dd03-d40-s5', .3, 40, 5), ('dd1-d40-s5', 1, 40, 5),
                              ('dd03-d80-s10', .3, 80, 10), ('dd03-d20-s2', .3, 20, 2)])
        for case in study.CASES:
            with self.subTest(case=case['id']):
                adapter = self.adapter()
                info = {}
                study.configure_adapter(adapter, case, self.baseline, info, False)
                solver = adapter._solver_config(3, 1.)
                self.assertEqual(solver.iteration_sets, ((3, case['dd_weight'], .08, -9.,
                                                        1 / case['dd_weight'], case['damping'], 1, 0.),))
                self.assertEqual(solver.smoothing_lon_lat_depth, tuple(case['smoothing']))
                text = adapter._input_text(None, synthetic=False, solver=solver, ray_step_km=.05)
                tokens = study.data_tokens(text, study.SCHEDULE_MARKER)
                self.assertEqual(len(tokens), 13)
                self.assertLessEqual(study.check_data_line_lengths(text), 80)
                self.assertAlmostEqual(float(tokens[5]) * float(tokens[9]), 1., places=15)
                self.assertEqual(info['serialized_weights']['absolute_amplitude'], 1.)
                self.assertEqual(set(info['generated_input_sha256']), set(study.CONSTANT_FILES))
                if case['dd_weight'] == .3:
                    self.assertEqual(tokens[5], '0.30')
                    self.assertEqual(tokens[9], '3.3333333333333335')
                    self.assertAlmostEqual(info['serialized_weights'][
                        'unmodified_renderer_absolute_amplitude'], .999999)

    def test_compact_schedule_regresses_attempt_one_88_column_failure(self):
        for case in study.CASES:
            with self.subTest(case=case['id']):
                solver = SimpleNamespace(iteration_sets=study.solver_schedule(case),
                                         smoothing_lon_lat_depth=tuple(case['smoothing']))
                original = frozen_renderer(None, synthetic=False, solver=solver, ray_step_km=.05)
                if case['dd_weight'] == .3:
                    attempt_one = original.replace('3.33333', '3.3333333333333335')
                    self.assertEqual(max(map(len, (line for line in attempt_one.splitlines()
                                                   if line and not line.startswith('*')))), 88)
                    with self.assertRaisesRegex(ValueError, '80 columns'):
                        study.check_data_line_lengths(attempt_one)
                compact = study.render_control(frozen_renderer, None, case, synthetic=False,
                                               solver=solver, ray_step_km=.05)
                self.assertLessEqual(study.check_data_line_lengths(compact), 80)
                self.assertEqual(len(study.data_tokens(compact, study.SCHEDULE_MARKER)), 13)

    def test_every_noncomment_data_line_is_length_checked(self):
        self.assertEqual(study.check_data_line_lengths('*' + 'comment' * 30 + '\n123\n'), 3)
        with self.assertRaisesRegex(ValueError, '80 columns'):
            study.check_data_line_lengths('x' * 81 + '\n')
        with self.assertRaisesRegex(ValueError, '80 columns'):
            study.check_data_line_lengths(' ' * 80 + 'x\n')

    def test_validation_hook_stops_before_solver(self):
        adapter = self.adapter()
        info = {}
        study.configure_adapter(adapter, study.CASES[0], self.baseline, info, True)
        with patch.object(subprocess, 'run', side_effect=AssertionError('No solver')):
            with self.assertRaises(study.InputsValidated):
                adapter._input_text(None, synthetic=False, solver=adapter._solver_config(3, 1.),
                                    ray_step_km=.05)
        self.assertIn('generated_control_sha256', info)

    def test_each_generated_constant_is_checked(self):
        generated = self.root / 'generated'
        generated.mkdir()
        for name in study.CONSTANT_FILES:
            (generated / name).write_bytes((self.inversion / name).read_bytes())
        study.check_generated(generated, self.baseline)
        for name in study.CONSTANT_FILES:
            with self.subTest(name=name):
                original = (generated / name).read_bytes()
                (generated / name).write_bytes(original + b'changed')
                with self.assertRaisesRegex(ValueError, name.replace('.', r'\.')):
                    study.check_generated(generated, self.baseline)
                (generated / name).write_bytes(original)

    def test_controls_outside_schedule_and_smoothing_are_frozen(self):
        adapter = self.adapter()
        study.configure_adapter(adapter, study.CASES[0], self.baseline, {}, False)
        with self.assertRaisesRegex(ValueError, 'Other generated solver controls'):
            adapter._input_text(None, synthetic=False, solver=adapter._solver_config(3, 1.),
                                ray_step_km=.1)
        with self.assertRaisesRegex(ValueError, 'Unexpected solver geometry'):
            adapter._solver_config(4, 1.)

    def test_root_and_output_safety(self):
        fresh = self.root / 'study'
        self.assertEqual(study.check_root(fresh), fresh)
        fresh.mkdir()
        self.assertEqual(study.check_root(fresh), fresh)
        (fresh / 'status.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            study.check_root(fresh)
        link = self.root / 'dangling'
        link.symlink_to(self.root / 'missing')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            study.check_root(link)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            study.worker.new_output(fresh)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            study.worker.new_output(link)
        with self.assertRaisesRegex(ValueError, 'protected'):
            study.check_root(study.OLD / 'new-study')

    def test_exclusive_manifest_never_overwrites(self):
        path = self.root / 'manifest.json'
        study.write_json(path, {'first': True}, exclusive=True)
        with self.assertRaises(FileExistsError):
            study.write_json(path, {'first': False}, exclusive=True)
        self.assertEqual(json.loads(path.read_text()), {'first': True})

    def test_preflight_precedes_override_and_arguments_remain_fixed(self):
        case = study.CASES[0]
        adapter = self.adapter()
        baseline_info = {k: {} for k in ('input_sha256', 'support_sha256', 'snapshot_source_sha256')}
        baseline_info.update(state='completed', solver_sha256='binary')
        (self.baseline / 'worker_info.json').write_text(json.dumps(baseline_info))
        output = self.root / 'temporary-generated'
        root = self.root / 'study'

        def preflight(args, actual_adapter):
            self.assertIs(actual_adapter, adapter)
            self.assertFalse(hasattr(adapter, '_solver_config'))
            self.assertEqual(args.geometry, 'synthetic')
            self.assertEqual(args.benchmark_root, study.SNAPSHOT / 'benchmark')
            return study.worker.run_arguments(args, args.solver_build / 'tomoDD-SE'), baseline_info.copy()

        with patch.object(study.worker, 'load_adapter', return_value=adapter), \
                patch.object(study.worker, 'preflight', side_effect=preflight):
            _, arguments, info = study.prepare_case(case, root, output)
        self.assertEqual(arguments['timeout_s'], 1200)
        self.assertEqual(arguments['run_dir'], output)
        self.assertEqual(info['arguments']['run_dir'], str(root / 'outputs' / case['id']))
        expected = {'iterations': 3, 'node_spacing_km': 1., 'ray_step_km': .05,
                    'initial_location_sigma_km': 1., 'initial_location_seed': 42,
                    'max_pair_degree': 4, 'max_pair_distance_km': 3., 'free_buffer_layers': 0}
        for key, value in expected.items():
            self.assertEqual(arguments[key], value)

    def test_normal_adapter_receives_empty_claimed_directory_and_retains_failure(self):
        root = self.root / 'study'
        (root / 'outputs').mkdir(parents=True)
        output = root / 'outputs' / study.CASES[0]['id']
        adapter = self.adapter()

        def run(**arguments):
            self.assertEqual(list(output.iterdir()), [])
            (output / 'solver-diagnostic.log').write_text('failure retained')
            raise RuntimeError('intentional solver-free failure')

        adapter.run = run
        with patch.object(study, 'prepare_case', return_value=(adapter, {}, {})):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                study.run_case(study.CASES[0], root, False)
        info = json.loads((output / 'worker_info.json').read_text())
        self.assertEqual(info['state'], 'failed')
        self.assertGreaterEqual(info['elapsed_s'], 0)
        self.assertTrue((output / 'solver-diagnostic.log').is_file())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            study.run_case(study.CASES[0], root, False)

    def test_normal_completion_retains_adapter_artifacts_and_worker_info(self):
        root = self.root / 'study'
        (root / 'outputs').mkdir(parents=True)
        case = study.CASES[0]
        output = root / 'outputs' / case['id']
        adapter = self.adapter()
        metrics = {'event_count': 1920, 'final_retained_event_count': 1920}

        def run(**arguments):
            inversion = output / 'work/inversion'
            inversion.mkdir(parents=True)
            for name in study.CONSTANT_FILES:
                (inversion / name).write_bytes((self.inversion / name).read_bytes())
            text = adapter._input_text(None, synthetic=False, solver=adapter._solver_config(3, 1.),
                                       ray_step_km=.05)
            (inversion / 'tomoDD-SE.inp').write_text(text)
            for name in ('config.json', 'results.json', 'state.npz', 'truth.npz'):
                (output / name).write_text('retained artifact')
            return metrics

        adapter.run = run
        with patch.object(study, 'prepare_case', return_value=(adapter, {}, {})), \
                patch.object(study.worker, 'check_completed', return_value=['verified-history']) as check, \
                patch('builtins.print'):
            self.assertEqual(study.run_case(case, root, False), 0)
        check.assert_called_once_with(output, adapter, metrics)
        info = json.loads((output / 'worker_info.json').read_text())
        self.assertEqual(info['state'], 'completed')
        self.assertGreaterEqual(info['elapsed_s'], 0)
        self.assertEqual(len(info['generated_input_sha256']), 9)
        self.assertEqual(info['iteration_artifacts'], ['verified-history'])

    def test_failed_worker_group_is_killed_even_if_worker_already_exited(self):
        log_dir = self.root / 'logs'
        log_dir.mkdir()
        process = Mock(pid=12345, returncode=1)
        process.poll.return_value = 1
        case = study.CASES[0]
        with patch.object(study, 'CASES', (case,)), \
                patch.object(study.subprocess, 'Popen', return_value=process), \
                patch.object(study.os, 'killpg') as kill:
            records = study.run_children(self.root, 1, False, log_dir)
        kill.assert_called_once_with(12345, study.signal.SIGKILL)
        self.assertEqual(records[case['id']]['state'], 'failed')

    def test_child_timeout_kills_process_group_and_reports_failure(self):
        log_dir = self.root / 'logs'
        log_dir.mkdir()
        process = Mock(pid=12345, returncode=-9)
        process.poll.return_value = -9
        case = study.CASES[0]
        with patch.object(study, 'CASES', (case,)), \
                patch.object(study, 'WORKER_TIMEOUT_S', -1), \
                patch.object(study.subprocess, 'Popen', return_value=process), \
                patch.object(study.os, 'killpg') as kill:
            records = study.run_children(self.root, 1, False, log_dir)
        kill.assert_called_once_with(12345, study.signal.SIGKILL)
        self.assertTrue(records[case['id']]['timed_out'])
        self.assertEqual(records[case['id']]['state'], 'failed')

    def test_manifest_and_status_exist_before_normal_launch_and_failure_is_retained(self):
        root = self.root / 'study'
        for name in ('config.json', 'results.json', 'worker_info.json'):
            (self.baseline / name).write_text('{}')

        def children(actual_root, jobs, validate_only, log_dir, status=None, reuse_completed_from=None):
            self.assertEqual(jobs, 4)
            if validate_only:
                self.assertFalse(root.exists())
                return {c['id']: {'state': 'validated', 'info': study.case_config(c, root)}
                        for c in study.CASES}
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['baseline'], str(self.baseline))
            self.assertEqual(manifest['cases'],
                             [json.loads(json.dumps(study.case_config(c, root))) for c in study.CASES])
            self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'running')
            self.assertTrue((root / 'outputs').is_dir())
            return {c['id']: {'state': 'failed' if i == 0 else 'completed'}
                    for i, c in enumerate(study.CASES)}

        with patch.object(study, 'run_children', side_effect=children), patch('builtins.print'):
            self.assertEqual(study.main(['--study-root', str(root), '--jobs', '4']), 1)
        self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'failed')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            study.main(['--study-root', str(root)])

    def reuse_fixture(self):
        case = next(c for c in study.CASES if c['id'] == study.REUSABLE_CASE)
        source = self.root / 'attempt-one'
        output = source / 'outputs' / case['id']
        inversion = output / 'work/inversion'
        inversion.mkdir(parents=True)
        for name in study.CONSTANT_FILES:
            (inversion / name).write_bytes((self.inversion / name).read_bytes())
        saved = source / 'run_study_attempt1.py'
        saved.write_text('saved attempt-one script; never execute')
        script_hash = study.worker.digest(saved)
        arguments = study.worker.run_arguments(SimpleNamespace(
            geometry='synthetic', output=output,
            forward_root=study.SNAPSHOT / 'projects/forward_modeling'), study.SNAPSHOT / 'solver/tomoDD-SE')
        arguments['timeout_s'] = 1200
        arguments = json.loads(json.dumps(arguments, default=str))
        solver = SimpleNamespace(iteration_sets=study.solver_schedule(case),
                                 smoothing_lon_lat_depth=case['smoothing'])
        control = frozen_renderer(None, synthetic=False, solver=solver, ray_step_km=.05)
        (inversion / 'tomoDD-SE.inp').write_text(control)
        old = json.loads(json.dumps(study.case_config(case, source)))
        old.update(state='completed', script_sha256=script_hash, arguments=arguments, events=1920,
                   generated_input_sha256=study.check_generated(inversion),
                   generated_control_sha256=study.worker.digest(inversion / 'tomoDD-SE.inp'))
        old.update({key: 'frozen-' + key for key in study.PROVENANCE_KEYS})
        config = {'solver': {'iteration_sets': old['solver_schedule'],
                             'smoothing_lon_lat_depth': case['smoothing'],
                             'binary_sha256': old['solver_sha256'], 'binary': arguments['binary'],
                             'timeout_s': 1200, 'ray_derivative_step_km': .05},
                  'initial_hypocenters': {'sigma_km': 1., 'seed': 42},
                  'source_experiment': arguments['experiment_id'], 'source_root': arguments['forward_root'],
                  'tomodd_grid': {'main_node_spacing_km': 1., 'free_buffer_layers': 0},
                  'observations': {'max_pair_degree': 4, 'max_pair_distance_km': 3.}}
        manifest = {'baseline': str(self.baseline), 'snapshot': str(study.SNAPSHOT), 'cases': [old],
                    'script_sha256': script_hash, 'worker_sha256': old['worker_sha256']}
        for path, data in ((source / 'manifest.json', manifest), (output / 'worker_info.json', old),
                           (output / 'config.json', config), (output / 'results.json', {'event_count': 1920}),
                           (source / 'status.json', {'cases': {case['id']: {
                               'state': 'completed', 'exit_code': 0, 'output': str(output)}}})):
            path.write_text(json.dumps(data))
        for name in ('state.npz', 'truth.npz', 'work/inversion/history.dat'):
            (output / name).write_text('retained nonempty artifact')
        fresh = json.loads(json.dumps(old))
        fresh.update(study.case_config(case, self.root / 'retry'))
        fresh['script_sha256'] = 'new-script-hash'
        fresh['generated_control_sha256'] = 'new-compact-preflight-control-hash'
        fresh['arguments']['run_dir'] = fresh['output']
        # Exercise tuple/list normalization for the live, not-yet-serialized preflight.
        fresh['arguments']['initial_layer_boundaries_km'] = study.worker.BOUNDARIES
        return case, source, output, fresh

    def test_reuse_verifies_old_artifacts_without_modifying_them(self):
        case, source, output, fresh = self.reuse_fixture()
        before = {str(p): study.worker.digest(p) for p in source.rglob('*') if p.is_file()}
        with patch.object(study.worker, 'check_completed', return_value=['work/inversion/history.dat']) as check:
            imported = study.validate_reuse(case, source, self.adapter(), fresh)
        self.assertTrue(imported['reuse_validated'])
        self.assertEqual(imported['output'], str(output))
        self.assertEqual(imported['reused_from'], str(output))
        self.assertEqual(imported['reuse_provenance']['manifest_sha256'],
                         study.worker.digest(source / 'manifest.json'))
        self.assertEqual(imported['preflight_control_sha256'], 'new-compact-preflight-control-hash')
        check.assert_called_once()
        self.assertEqual(before, {str(p): study.worker.digest(p) for p in source.rglob('*') if p.is_file()})

    def test_reuse_rejects_changed_config_inputs_control_and_manifest(self):
        case, source, output, fresh = self.reuse_fixture()
        config_path = output / 'config.json'
        original = config_path.read_text()
        mutations = [('solver', 'iteration_sets', [[4, 1, .08, -9, 1, 40, 1, 0]]),
                     ('solver', 'smoothing_lon_lat_depth', [2, 2, 2]),
                     ('solver', 'binary_sha256', 'changed'),
                     ('initial_hypocenters', 'sigma_km', .5),
                     ('initial_hypocenters', 'seed', 43)]
        for section, key, value in mutations:
            with self.subTest(key=key):
                config = json.loads(original)
                config[section][key] = value
                config_path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    study.validate_reuse(case, source, self.adapter(), fresh)
                config_path.write_text(original)
        for name in (*study.CONSTANT_FILES, 'tomoDD-SE.inp'):
            with self.subTest(file=name):
                path = output / 'work/inversion' / name
                data = path.read_bytes()
                path.write_bytes(data + b'changed')
                with self.assertRaises(ValueError):
                    study.validate_reuse(case, source, self.adapter(), fresh)
                path.write_bytes(data)
        (source / 'run_study_attempt1.py').write_text('changed saved script')
        with self.assertRaisesRegex(ValueError, 'saved script hash'):
            study.validate_reuse(case, source, self.adapter(), fresh)
        with self.assertRaisesRegex(ValueError, 'Only dd1'):
            study.validate_reuse(study.CASES[0], source, self.adapter(), fresh)

    def test_normal_queue_skips_only_validated_reused_dd1(self):
        log_dir = self.root / 'logs'
        log_dir.mkdir()
        status = {'cases': {c['id']: {'state': 'completed'} for c in study.CASES}}
        status['cases'][study.REUSABLE_CASE].update(reused_from='original-output', reuse_validated=True)
        for case in study.CASES:
            output = self.root / 'outputs' / case['id']
            output.mkdir(parents=True)
            (output / 'worker_info.json').write_text(json.dumps({'state': 'completed'}))
        process = Mock(pid=12345, returncode=0)
        process.poll.return_value = 0
        with patch.object(study.subprocess, 'Popen', return_value=process) as launch:
            results = study.run_children(self.root, 4, False, log_dir, status)
        commands = [call.args[0] for call in launch.call_args_list]
        launched = [cmd[cmd.index('--case') + 1] for cmd in commands]
        self.assertEqual(launched, [c['id'] for c in study.CASES if c['id'] != study.REUSABLE_CASE])
        self.assertEqual(results[study.REUSABLE_CASE]['reused_from'], 'original-output')
        self.assertFalse((log_dir / (study.REUSABLE_CASE + '.log')).exists())
        status['cases'][study.CASES[0]['id']].update(reused_from='not-approved', reuse_validated=True)
        with self.assertRaisesRegex(ValueError, 'Unvalidated case reuse'):
            study.run_children(self.root, 4, False, log_dir, status)

    def test_retry_manifest_points_to_original_completed_output(self):
        case, source, output, fresh = self.reuse_fixture()
        root = self.root / 'retry'
        for name in ('config.json', 'results.json', 'worker_info.json'):
            (self.baseline / name).write_text('{}')
        with patch.object(study.worker, 'check_completed', return_value=['work/inversion/history.dat']):
            imported = study.validate_reuse(case, source, self.adapter(), fresh)

        def children(actual_root, jobs, validate_only, log_dir, status=None, reuse_completed_from=None):
            if validate_only:
                self.assertEqual(reuse_completed_from, source)
                records = {c['id']: {'state': 'validated', 'info': study.case_config(c, root)}
                           for c in study.CASES}
                records[case['id']]['info'].update(imported)
                return records
            manifest = json.loads((root / 'manifest.json').read_text())
            declared = next(c for c in manifest['cases'] if c['id'] == case['id'])
            self.assertEqual(declared['output'], str(output))
            self.assertEqual(manifest['original_manifest_sha256'], study.worker.digest(source / 'manifest.json'))
            self.assertEqual(manifest['script_version'], study.SCRIPT_VERSION)
            self.assertEqual(status['cases'][case['id']]['state'], 'completed')
            self.assertEqual(status['cases'][case['id']]['reused_from'], str(output))
            self.assertFalse((root / 'outputs' / case['id']).exists())
            return {c['id']: {'state': 'completed'} for c in study.CASES}

        with patch.object(study, 'run_children', side_effect=children), patch('builtins.print'):
            code = study.main(['--study-root', str(root), '--reuse-completed-from', str(source)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'completed')

    def test_parent_validation_does_not_create_study_root(self):
        root = self.root / 'study'
        records = {c['id']: {'state': 'validated', 'info': study.case_config(c, root)}
                   for c in study.CASES}
        with patch.object(study, 'run_children', return_value=records), patch('builtins.print'):
            self.assertEqual(study.main(['--study-root', str(root), '--validate-only']), 0)
        self.assertFalse(root.exists())


if __name__ == '__main__':
    unittest.main()
