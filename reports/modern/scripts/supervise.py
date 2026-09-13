"""Durable supervisor for the existing modern-baseline runner, not a new evaluator."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path("/home/abc/temp/modern")
PYTHON = sys.executable
RUNNER = ROOT / "bridge/run_bridge_cohort.py"
QUEUES = ("D1", "D2", "T-OCC", "T-FULL", "L")
MAX_COUNTS = {"D1": 1440, "D2": 1800, "T-OCC": 7200, "T-FULL": 5400, "L": 240}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if text[0] == "Z":
            return None
        return text[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


class Supervisor:
    def __init__(self):
        self.child = None
        self.stage = "PRECHECK"
        self.started = time.time()
        self.last_note = 0.0
        self.last_answer_size = (Path("/home/abc/temp/answer.md").stat().st_size
                                 if Path("/home/abc/temp/answer.md").exists() else 0)
        self.cache = {}
        self.stage_elapsed = {}
        self.eta_seconds_per_episode = {"bayes": 7.0, "tmpc": 4.0, "shmpc": 5.0}

    def progress(self):
        report = {}
        for queue in QUEUES:
            manifest = ROOT / "tasks" / f"{queue}.jsonl"
            tasks = [json.loads(line) for line in manifest.read_text().splitlines()
                     if line.strip()] if manifest.exists() else []
            expected = sum(len(task["cases"]) for task in tasks) if tasks else MAX_COUNTS[queue]
            completed = errors = 0
            for path in (ROOT / "runs" / queue).glob("*.jsonl"):
                stamp = (path.stat().st_mtime_ns, path.stat().st_size)
                cached = self.cache.get(str(path))
                if not cached or cached[0] != stamp:
                    try:
                        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                        cached = (stamp, len(rows), sum(r["event"] == "error" for r in rows))
                    except (ValueError, KeyError):
                        cached = (stamp, 0, 0)
                    self.cache[str(path)] = cached
                completed += cached[1]
                errors += cached[2]
            report[queue] = {"completed": completed, "planned": expected, "errors": errors}
        return report

    def publish(self, status="RUNNING", message="", force=False):
        now = time.time()
        report = self.progress()
        payload = {"status": status, "stage": self.stage, "pid": os.getpid(),
                   "child_pid": self.child.pid if self.child and self.child.poll() is None else None,
                   "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                   "elapsed_hours": (now - self.started) / 3600,
                   "workers": 1, "threads_per_worker": 4, "device": "local CPU",
                   "queues": report, "message": message}
        # Wall-clock estimates, not a deadline promise. Refine from completed
        # serial queue stages; coarse D1 timings included competing workers.
        remaining = {q: max(0, v["planned"] - v["completed"]) for q, v in report.items()}
        rates = self.eta_seconds_per_episode
        general = sum(rates.values()) / 3
        seconds = ((remaining["D1"] + remaining["D2"]) * general
                   + remaining["T-OCC"] * (2*rates["bayes"] + rates["tmpc"] + rates["shmpc"]) / 4
                   + remaining["T-FULL"] * general + remaining["L"] * general)
        payload["remaining_hours_range"] = [round(seconds / 3600 * .8 + .3, 1),
                                            round(seconds / 3600 * 1.5 + 1, 1)]
        payload["eta_note"] = "Initial sparse/dense probe estimate, updated from serial completed stages; excludes unforeseen repair time."
        if status in ("TEST_DONE", "COMPLETED_WITH_ERRORS_REQUIRES_REVIEW"):
            payload["remaining_hours_range"] = [0, 0]
        atomic_json(ROOT / "snapshot/pipeline_status.json", payload)
        if force or now - self.last_note >= 5400:
            answer = Path("/home/abc/temp/answer.md")
            with answer.open("a+") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                handle.seek(0)
                content = handle.read()
                changed = len(content.encode()) != self.last_answer_size
                lines = [f"\n\n## {payload['updated_at']} Codex 自动执行记录",
                         f"状态: {status}; 阶段: {self.stage}; PID: {os.getpid()}。",
                         "本地串行1 worker，每个规划器最多4线程；不改算法使用CPU求解。"]
                lines += [f"- {q}: {v['completed']}/{v['planned']} 回合，error={v['errors']}。"
                          for q, v in report.items()]
                lines.append(f"剩余粗估 {payload['remaining_hours_range'][0]}–{payload['remaining_hours_range'][1]} 小时；未含新的故障修复。")
                if message:
                    lines.append(message)
                if changed:
                    lines.append("检测到文档另有新增内容；自动守护程序不能代替人工技术答复，已标记待下次交互复核，不冒充已解答。")
                lines.append("阶段按验收、D1、D2、冻结、T-OCC、T-FULL、独占耗时、统计自动串联；只有完整性和哈希检查通过才标完成。")
                handle.write("\n".join(lines) + "\n")
                handle.flush()
                self.last_answer_size = answer.stat().st_size
            self.last_note = now

    def adopt(self, pid, expected):
        identity = process_identity(pid)
        if identity is None:
            return
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode().replace("\0", " ")
        if expected not in cmd:
            raise RuntimeError(f"refusing to adopt unrelated PID {pid}: {cmd}")
        self.stage = f"WAIT_EXISTING_{pid}"
        self.publish(message=f"保留并等待CC已有任务 PID={pid}，不重复启动。", force=True)
        while process_identity(pid) == identity:
            self.publish()
            time.sleep(20)

    def command(self, label, argv, retry=False):
        self.stage = label
        self.publish(message=f"开始 {label}", force=True)
        for attempt in range(2 if retry else 1):
            log = ROOT / "logs" / f"{label}.log"
            log.parent.mkdir(exist_ok=True)
            with log.open("ab", buffering=0) as output:
                output.write((f"\n[{datetime.now().isoformat()}] {argv!r}\n").encode())
                self.child = subprocess.Popen(argv, cwd=ROOT, stdout=output,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                started = time.monotonic()
                while self.child.poll() is None:
                    self.publish()
                    if time.monotonic() - started > 48 * 3600:
                        os.killpg(self.child.pid, signal.SIGTERM)
                        self.child.wait(timeout=30)
                        raise RuntimeError(f"{label}: watchdog expired")
                    time.sleep(10)
                code = self.child.returncode
                self.child = None
            if code == 0:
                self.stage_elapsed[label] = time.monotonic() - started
                return
            if retry and attempt == 0:
                self.publish(message=f"{label}退出码{code}；保留日志，按完整块续跑一次。", force=True)
            else:
                raise RuntimeError(f"{label}: exit={code}, see {log}")

    def queue(self, name):
        before = self.progress()[name]["completed"]
        for command in ("plan", "precheck", "run", "aggregate"):
            args = [PYTHON, "-u", "-B", str(RUNNER), command, name]
            if command == "run":
                args += ["--workers", "1", "--threads", "4"]
            self.command(f"{name}_{command}", args, retry=command == "run")
        elapsed = self.stage_elapsed.get(f"{name}_run", 0)
        completed = self.progress()[name]["completed"] - before
        if completed >= 100 and name in ("D2", "T-OCC", "T-FULL"):
            observed = elapsed / completed
            old_average = sum(self.eta_seconds_per_episode.values()) / 3
            factor = observed / old_average
            self.eta_seconds_per_episode = {k: v * factor for k, v in self.eta_seconds_per_episode.items()}

    def final_report(self):
        summaries = json.loads((ROOT / "final/main_summaries.json").read_text())
        stats = json.loads((ROOT / "final/paired_comparisons.json").read_text())
        counts = self.progress()
        if any(v["completed"] != v["planned"] for v in counts.values()):
            raise RuntimeError("final counts do not match manifests")
        errors = sum(v["errors"] for v in counts.values())
        heading = "TEST_DONE" if not errors else "COMPLETED_WITH_ERRORS_REQUIRES_REVIEW"
        lines = [f"\n\n## {datetime.now().astimezone().isoformat(timespec='seconds')} Codex 正式队列交付",
                 f"状态：{heading}。实际回合数：{sum(v['completed'] for v in counts.values())}。",
                 "统计单位为场景内独立布局，三个规划重复不当作三个独立样本。",
                 "|条件|方法|SR%|CR%|TR%|ERROR%|罚时秒|",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for condition, table in summaries.items():
            for arm, entry in table.items():
                m = entry["macro"]
                lines.append(f"|{condition}|{arm}|{100*m['sr']:.2f}|{100*m['cr']:.2f}|{100*m['tr']:.2f}|{100*m['err']:.2f}|{m['time']:.3f}|")
        for key, value in stats["primary"].items():
            lines.append(f"- {key}: 差 {value['difference']:+.4f}, 95%CI [{value['ci_low']:+.4f}, {value['ci_high']:+.4f}], Holm p={value['p_holm']:.4g}。")
        lines += ["文件：`temp/modern/final/`；续跑/复现入口：`temp/modern/pipeline.sh`。",
                  "结果仅适用于共享跟踪预测的适配版本；执行动力学误差、SH跨时协方差近似与随机并行限制见snapshot/frozen.json。",
                  "不能用不显著证明安全等价，也不能单凭本表证明novelty或贝叶斯不可替代。"]
        for filename in ("FINAL.md", "STATUS.md", "INDEX.md"):
            with (ROOT / filename).open("a") as handle:
                handle.write("\n".join(lines) + "\n")
        atomic_json(ROOT / "final/completion.json", {"status": heading, "queues": counts,
                    "frozen_sha256": hashlib.sha256((ROOT / "snapshot/frozen.json").read_bytes()).hexdigest()})
        self.stage = "FINISHED"
        self.publish(status=heading, message="正式表、CI、多重比较、逐步耗时与失败分析已生成，见temp/modern/final/。", force=True)

    def run(self, args):
        for pid, expected in ((args.adopt_d1_pid, "run_bridge_cohort.py run D1"),
                              (args.adopt_dynamics_pid, "dynamics_crosscheck.py")):
            if pid:
                self.adopt(pid, expected)
        self.command("SELFTEST", [PYTHON, "-B", str(ROOT / "test_takeover.py")])
        self.command("INTERFACE", [PYTHON, "-u", "-B", str(ROOT / "bridge/interface_audit.py")])
        self.command("REGISTRY_CHECK", [PYTHON, "-B", str(ROOT / "case_registry.py"), "--verify-existing"])
        if not (ROOT / "snapshot/dynamics_crosscheck.json").exists():
            self.command("DYNAMICS", [PYTHON, "-u", "-B", str(ROOT / "dynamics_crosscheck.py")])
        if not (ROOT / "snapshot/frozen.json").exists():
            self.queue("D1")
            self.command("D1_SELECT", [PYTHON, "-B", str(ROOT / "risk_sweep/front.py"), "D1"])
            self.queue("D2")
            self.command("D2_SELECT", [PYTHON, "-B", str(ROOT / "risk_sweep/front.py"), "D2"])
        self.command("FREEZE", [PYTHON, "-B", str(ROOT / "freeze.py")])
        self.queue("T-OCC")
        self.queue("T-FULL")
        self.queue("L")
        self.command("ANALYSIS", [PYTHON, "-u", "-B", str(ROOT / "analysis.py")])
        self.command("VERIFY_FINAL", [PYTHON, "-B", str(ROOT / "test_takeover.py"), "--verify-final"])
        self.final_report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adopt-d1-pid", type=int)
    parser.add_argument("--adopt-dynamics-pid", type=int)
    args = parser.parse_args()
    supervisor = Supervisor()
    with (ROOT / "snapshot/supervisor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("an existing supervisor is already running")
        def terminate(signum, _frame):
            if supervisor.child and supervisor.child.poll() is None:
                os.killpg(supervisor.child.pid, signal.SIGTERM)
            raise RuntimeError(f"supervisor received signal {signum}; existing blocks retained")
        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        supervisor.publish(message="接管CC已有任务；之后本地串行运行，保留原候选与新布局划分。", force=True)
        try:
            supervisor.run(args)
        except Exception as exc:
            supervisor.publish(status="FAILED_REQUIRES_REPAIR", message=str(exc), force=True)
            raise


if __name__ == "__main__":
    main()
