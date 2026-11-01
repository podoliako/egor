"""Stdlib-only lifecycle tests; no worker or numerical code is executed."""
import csv
import fcntl
import importlib.util

from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location("soft_screen_pipeline", Path(__file__).with_name("pipeline.py"))
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        self.cases = [{"id": i, "config": {"n_cycles": 2}, "experiment_id": "fake"}
                      for i in ("b", "a", "c")]
        self.manifest = dict(project_root=str(self.state), cases=self.cases, case_timeout_s=10)
        self.save_manifest()
        self.clock, self.active, self.maximum, self.order = 0, 0, 0, []
        self.behavior, self.processes, self.beats = {}, [], []

    def save_manifest(self):
        queue.write(self.state / "manifest.json", self.manifest)

    def result(self, case_id, status="completed", **extra):
        directory = self.state / "cases" / case_id
        directory.mkdir(parents=True, exist_ok=True)
        run = self.state / "runs" / case_id
        run.mkdir(parents=True, exist_ok=True)
        if status == "completed":
            (run / "final_model.npy").touch()
            for i in range(2):
                (run / f"iter_{i}").mkdir(exist_ok=True)
                (run / f"iter_{i}" / "complete.json").write_text("{}")
        queue.write(directory / "result.json", dict(
            status=status, run_dir=str(run), final_rms_m_s=12.5, **extra))
        return run

    def popen(self, command, **kwargs):
        self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
        case_id = command[-1]
        self.order.append(case_id)
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        process = Mock(pid=10000 + len(self.processes))
        process.alive = True
        self.processes.append(process)
        mode = self.behavior.get(case_id, "completed")

        def wait(timeout=None):
            if not process.alive:
                return 0
            if mode in ("term", "int", "timeout", "heartbeat"):
                if timeout is not None:
                    if mode == "heartbeat" and self.clock >= 5:
                        self.result(case_id)
                    else:
                        self.clock += timeout
                        if mode in ("term", "int"):
                            sig = signal.SIGTERM if mode == "term" else signal.SIGINT
                            signal.getsignal(sig)(sig, None)
                        self.beats.append(queue.read(self.state / "status.json"))
                        raise subprocess.TimeoutExpired(command, timeout)
            process.alive = False
            self.active -= 1
            if mode not in ("term", "int", "timeout", "heartbeat", "missing", "crash"):
                self.result(case_id, mode)
            return 7 if mode == "crash" else 0
        process.wait.side_effect = wait
        return process

    def execute(self):
        self.kills = []
        with patch.object(queue.subprocess, "Popen", side_effect=self.popen) as spawn, \
                patch.object(queue.os, "killpg", side_effect=lambda pid, sig: self.kills.append(
                    (pid, sig, self.clock))) as kill, \
                patch.object(queue.time, "time", side_effect=lambda: 1000 + self.clock), \
                patch.object(queue.time, "monotonic", side_effect=lambda: self.clock):
            code = queue.run(self.state)
        self.spawn, self.kill = spawn, kill
        self.status = queue.read(self.state / "status.json")
        return code

    def test_order_one_worker_continue_failure_and_command(self):
        self.behavior = {"b": "failed", "a": "crash"}
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.order, ["b", "a", "c"])
        self.assertEqual((self.maximum, self.active), (1, 0))
        self.assertEqual(self.status["status"], "completed_with_failures")
        self.assertEqual([self.status[k] for k in ("completed", "failed", "pending", "total")], [1, 2, 0, 3])
        command = self.spawn.call_args_list[0].args[0]
        options = self.spawn.call_args_list[0].kwargs
        self.assertEqual(command, [sys.executable, "-B", str(Path(queue.__file__).with_name("worker.py")),
                                   "--state", str(self.state), "--case", "b"])
        self.assertTrue(options["start_new_session"])
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertEqual(options["stderr"], subprocess.STDOUT)
        self.assertEqual(options["cwd"], self.state)
        self.assertEqual(Path(options["stdout"].name), self.state / "cases/b/run.log")
        for name in ("OPENBLAS", "OMP", "MKL", "NUMBA"):
            self.assertEqual(options["env"][name + "_NUM_THREADS"], "1")
        with (self.state / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["case_id"] for r in rows], self.order)
        self.assertEqual(rows[-1]["final_rms_m_s"], "12.5")

    def test_resume_skips_completed_failed_and_marks_running(self):
        self.result("b")
        self.result("a", "failed", error="original")
        self.result("c", "running")
        original = (self.state / "cases/a/result.json").read_bytes()
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.order, [])
        self.assertEqual((self.state / "cases/a/result.json").read_bytes(), original)
        self.assertEqual(queue.read(self.state / "cases/c/result.json")["status"], "failed")
        self.assertEqual(self.status["completed"], 1)
        self.result("a")
        self.result("c")
        self.assertEqual(self.execute(), 0)
        self.assertEqual((self.order, self.status["status"]), ([], "completed"))

    def test_incomplete_or_malformed_results_never_rerun(self):
        for missing in ("final_model.npy", "iter_0/complete.json", "iter_1/complete.json"):
            with self.subTest(missing=missing):
                run = self.result("b")
                (run / missing).unlink()
                self.result("a", "failed")
                self.result("c", "failed")
                self.assertEqual(self.execute(), 1)
                self.assertEqual(self.order, [])
                self.assertEqual(self.status["failed"], 3)
        (self.state / "cases/b/result.json").write_text("{broken")
        self.assertEqual(self.execute(), 1)
        (self.state / "cases/b/result.json").unlink()
        (self.state / "cases/b/run.log").touch()
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.order, [])

    def test_duplicate_lock_does_not_touch_status_or_spawn(self):
        queue.write(self.state / "status.json", {"owner": "other"})
        with (self.state / "queue.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.execute(), 2)
        self.assertEqual(self.status, {"owner": "other"})
        self.spawn.assert_not_called()

    def test_term_int_timeout_kill_group_reap_and_restore_handlers(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
        for mode in ("term", "int", "timeout"):
            with self.subTest(mode=mode):
                # Isolate attempts using independent state directories.
                self.state = Path(self.tmp.name) / mode
                self.state.mkdir()
                self.manifest["project_root"] = str(self.state)
                self.save_manifest()
                self.behavior, self.order, self.processes = {"b": mode}, [], []
                self.assertEqual(self.execute(), 1 if mode == "timeout" else 130)
                self.assertEqual(self.order, ["b", "a", "c"] if mode == "timeout" else ["b"])
                pid = self.processes[0].pid

                self.kill.assert_any_call(pid, signal.SIGTERM)
                self.kill.assert_any_call(pid, signal.SIGKILL)
                self.assertEqual(self.kills[1][2] - self.kills[0][2], 30)
                self.assertTrue(all(c.kwargs.get("timeout", 0) <= 5
                                    for c in self.processes[0].wait.call_args_list))
                self.assertEqual(self.processes[0].wait.call_args_list[-1], unittest.mock.call())
                self.assertEqual(self.active, 0)
                self.assertIsNone(self.status["worker_pid"])
                self.assertIsNone(self.status["current_case"])
                self.assertEqual(self.status["status"], "completed_with_failures" if mode == "timeout" else "interrupted")
                for sig, previous in handlers.items():
                    self.assertEqual(signal.getsignal(sig), previous)

    def test_heartbeat_missing_result_and_launch_exception(self):
        self.behavior = {"b": "heartbeat", "a": "missing"}
        self.assertEqual(self.execute(), 1)
        self.assertEqual(self.beats[0]["current_case"], "b")
        self.assertEqual(self.beats[0]["worker_pid"], 10000)
        self.assertEqual(self.beats[0]["heartbeat"], 1000)
        self.assertEqual(self.status["heartbeat"], 1005)
        self.assertEqual(self.status["failed"], 1)
        self.result("b", "running")
        (self.state / "cases/c/result.json").unlink()
        (self.state / "cases/c/run.log").unlink()
        with patch.object(queue.subprocess, "Popen", side_effect=OSError("cannot spawn")):
            self.assertEqual(queue.run(self.state), 1)
        self.assertIn("cannot spawn", queue.read(self.state / "cases/c/result.json")["error"])

    def test_manifest_exception_is_recorded(self):
        self.manifest["case_timeout_s"] = -1
        self.save_manifest()
        self.assertEqual(self.execute(), 130)
        self.spawn.assert_not_called()
        self.assertIn("ValueError", self.status["error"])
        self.assertEqual(self.status["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
