"""Regression tests for unattended execution and independent final-table checks."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path("/home/abc/temp/modern")
sys.path[:0] = [str(ROOT), str(ROOT / "bridge"), "/home/abc/workspace/bayes_occ_mpc"]
import numpy as np
import analysis
import run_bridge_cohort as runner
from modern_worker import BridgeController


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task = runner.plan_queue("D1")[0]
        self.task["cases"] = self.task["cases"][:1]
        self.row = {**self.task, "case_id": self.task["cases"][0],
                    "event": "reach_goal", "nav_time": 10.0,
                    "success_without_overlap": 1, "collision_union": 0,
                    "steps": [{}], "plan_step_ms": [1.0], "pipeline_step_ms": [2.0],
                    "code_sha256": "test", "threads": 4}
        self.row["layout_sha256_16"] = runner._layout_hashes_for(
            self.task, self.task["scene_id"])[str(self.row["case_id"])]

    def test_block_validation(self):
        with patch.object(runner, "ROOT", self.root):
            path = runner.block_path(self.task)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(self.row) + "\n")
            self.assertTrue(runner.block_done(self.task))
            self.row["layout_sha256_16"] = "wrong"
            path.write_text(json.dumps(self.row) + "\n")
            self.assertFalse(runner.block_done(self.task))
            with self.assertRaises(RuntimeError):
                runner.validated_rows("D1")

    def test_interrupted_prefix_not_appended(self):
        with patch.object(runner, "ROOT", self.root):
            path = runner.block_path(self.task)
            path.parent.mkdir(parents=True)
            partial = path.with_suffix(".jsonl.partial")
            partial.write_text('{"interrupted":true}\n')
            factory = types.SimpleNamespace(close=lambda: None)
            with patch.object(runner, "make_planner", return_value=(factory, None)), \
                 patch.object(runner, "_run_one", return_value=self.row):
                runner.run_block(self.task, {})
            self.assertEqual(len(path.read_text().splitlines()), 1)
            self.assertEqual(len(list(path.parent.glob("*.superseded.*"))), 1)
            self.assertTrue(runner.block_done(self.task))

    def test_latency_groups_thread_profiles(self):
        rows = [{**self.row, "arm": "bayes_full", "condition": "occluded", "threads": t}
                for t in (1, 4)]
        target = self.root / "latency.csv"
        analysis.latency(rows, target)
        with target.open() as handle:
            result = list(csv.DictReader(handle))
        self.assertEqual({r["threads"] for r in result}, {"1", "4"})


class StatisticsTests(unittest.TestCase):
    def test_collision_precedes_success_and_error(self):
        for event in ("reach_goal", "timeout", "error"):
            values = analysis.episode_values({"event": event, "collision_union": 1,
                                              "success_without_overlap": 1, "nav_time": 8})
            self.assertEqual([values[k] for k in ("sr", "cr", "tr", "err", "time")], [0, 1, 0, 0, 25])

    def test_error_is_not_removed(self):
        row = {"event": "error", "collision_union": None, "nav_time": 0}
        values = analysis.episode_values(row)
        self.assertEqual(values["time"], 25)
        self.assertEqual(values["cr_unknown"], 1)
        self.assertEqual(sum(values[k] for k in ("sr", "cr", "tr", "err")), 1)

    def test_bootstrap_and_permutation(self):
        a = {"s": {i: {"time": 8.0} for i in range(30)}}
        b = {"s": {i: {"time": 10.0} for i in range(30)}}
        with patch.object(analysis, "BOOTSTRAP", 1000):
            same = analysis.paired_bootstrap(a, a, "time", np.random.default_rng(1))
            different = analysis.paired_bootstrap(a, b, "time", np.random.default_rng(1))
        self.assertEqual(same["p"], 1)
        self.assertEqual(different["difference"], -2)
        self.assertEqual(different["ci_low"], -2)
        self.assertGreater(different["p"], 0)
        self.assertLess(different["p"], 0.01)
        del b["s"][0]
        with self.assertRaises(ValueError):
            analysis.paired_bootstrap(a, b, "time", np.random.default_rng(1))

    def test_holm(self):
        got = analysis.holm({"a": {"p": .01}, "b": {"p": .04}, "c": {"p": .03}})
        self.assertEqual(got, {"a": .03, "c": .06, "b": .06})


class TransportTests(unittest.TestCase):
    def peer(self, script):
        peer = BridgeController.__new__(BridgeController)
        peer.variant = "fake"
        peer.process = subprocess.Popen([sys.executable, "-u", "-c", script],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, bufsize=1)
        peer._reply_buffer = bytearray()
        peer._log = subprocess.DEVNULL
        os.set_blocking(peer.process.stdin.fileno(), False)
        os.set_blocking(peer.process.stdout.fileno(), False)
        def cleanup():
            if peer.process.poll() is None:
                peer.process.kill()
            peer.process.wait(timeout=5)
            peer.process.stdin.close()
            peer.process.stdout.close()
        self.addCleanup(cleanup)
        return peer

    def test_large_request_and_replies(self):
        peer = self.peer("import sys\nfor line in sys.stdin:\n print('OK '+str(len(line.strip())),flush=True)")
        with patch.dict(os.environ, {"MODERN_IPC_TIMEOUT_S": "10"}):
            self.assertEqual(peer._exchange("a" * 120000), "OK 120000")
            self.assertEqual(peer._exchange("INFO"), "OK 4")

    def test_hung_peer_has_deadline(self):
        peer = self.peer("import time; time.sleep(30)")
        with patch.dict(os.environ, {"MODERN_IPC_TIMEOUT_S": "0.15"}):
            with self.assertRaises(TimeoutError):
                peer._exchange("INFO")


def verify_final():
    summaries = json.loads((ROOT / "final/main_summaries.json").read_text())
    for queue, condition, count in (("T-OCC", "occluded", 7200), ("T-FULL", "full", 5400)):
        rows = runner.validated_rows(queue)
        if len(rows) != count:
            raise RuntimeError(f"{queue}: wrong episode count {len(rows)}")
        for arm, table in summaries[condition].items():
            group = [r for r in rows if r["arm"] == arm]
            if len(group) != 1800:
                raise RuntimeError(f"{arm}: missing repeats")
            safe = [bool(r.get("success_without_overlap")) and not r.get("collision_union")
                    and r["event"] != "error" for r in group]
            sr = sum(safe) / len(group)
            penalised = sum(r["nav_time"] if ok else 25.0 for r, ok in zip(group, safe)) / len(group)
            if abs(sr - table["macro"]["sr"]) > 1e-12 or abs(penalised - table["macro"]["time"]) > 1e-10:
                raise RuntimeError(f"independent count mismatch: {arm}")
            keys = {(r["scene_id"], r["layout_sha256_16"], r["repeat"]) for r in group}
            if len(keys) != 1800:
                raise RuntimeError(f"duplicate pairing keys: {arm}")
    if len(runner.validated_rows("L")) != 240:
        raise RuntimeError("incomplete latency queue")
    print("FINAL VERIFIED: complete layouts/repeats, independently recomputed SR and penalised time")


if __name__ == "__main__":
    if "--verify-final" in sys.argv:
        verify_final()
    else:
        unittest.main(verbosity=2)
