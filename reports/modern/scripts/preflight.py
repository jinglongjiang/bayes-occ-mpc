"""Preflight registration for the formal run (order section 13.1).

Records what would otherwise be impossible to reconstruct afterwards: which
commit, which working-tree changes, which untracked sources, which module a
name actually resolves to, and the SHA256 of every artefact that participates
in a solve -- generator config, generated code, solver binary, and the .so the
process really loads.

A hash of the top-level script is not enough: the solver binary is generated
from a configuration, and a mismatch between the two is exactly the failure this
guards against.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/abc/workspace/bayes_occ_mpc")
BUILD = ROOT / "build_modern"
OUT = Path("/home/abc/temp/modern/snapshot")

REPOS = {
    "bayes_occ_mpc": ROOT,
    "CrowdNav": Path("/home/abc/workspace/nav_data/mamba/camrl/CrowdNav"),
    "mpc_planner": ROOT / "src/mpc_planner",
    "scenario_module": ROOT / "src/scenario_module",
    "ros_tools": ROOT / "src/ros_tools",
    "guidance_planner": ROOT / "src/guidance_planner",
}

# Everything that decides what a solve computes.
def artefacts(workspace: Path):
    mp = workspace / "src/mpc_planner"
    return {
        "settings.yaml": mp / "mpc_planner_jackalsimulator/config/settings.yaml",
        "guidance_planner.yaml": mp / "mpc_planner_jackalsimulator/config/guidance_planner.yaml",
        "scenario_params.yaml": workspace / "src/scenario_module/config/params.yaml",
        "generator.py": mp / "mpc_planner_jackalsimulator/scripts/generate_jackalsimulator_solver.py",
        "modules.h": mp / "mpc_planner_modules/include/mpc_planner_modules/modules.h",
        "definitions.h": mp / "mpc_planner_modules/include/mpc_planner_modules/definitions.h",
        "solver.cmake": mp / "mpc_planner_solver/solver.cmake",
        "acados_ocp_solver.so": mp / "mpc_planner_solver/acados/Solver/libacados_ocp_solver_Solver.so",
        "libmpc_planner_solver.so": workspace / "devel/lib/libmpc_planner_solver.so",
        "libmpc_planner_modules.so": workspace / "devel/lib/libmpc_planner_modules.so",
        "mpc_bridge": workspace / "devel/lib/mpc_planner_bridge/mpc_bridge",
    }

PROJECT_SOURCES = [
    ROOT / "continuous_mpc_gate.py",
    ROOT / "unicycle_mpc_gate.py",
    ROOT / "modern_worker.py",
    ROOT / "evaluate_matched_safety.py",
    ROOT / "visible_history_env.py",
    Path("/home/abc/temp/modern/bridge/mpc_bridge.cpp"),
    Path("/home/abc/temp/modern/bridge/interface_audit.py"),
    Path("/home/abc/temp/modern/bridge/verify_uncertainty.py"),
    Path("/home/abc/temp/modern/bridge/run_bridge_cohort.py"),
    Path("/home/abc/temp/modern/risk_sweep/front.py"),
]


def sha256(path: Path):
    if not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args):
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as exc:
        return f"<{type(exc).__name__}: {exc}>"


def loaded_shared_objects(binary: Path, env_extra=None):
    """What the linker actually resolves, not what we think it links."""
    if not binary.exists():
        return None
    env = dict(os.environ)
    env.update(env_extra or {})
    try:
        out = subprocess.run(["ldd", str(binary)], capture_output=True, text=True,
                             timeout=60, env=env).stdout
    except Exception as exc:
        return f"<{type(exc).__name__}: {exc}>"
    resolved = {}
    for line in out.splitlines():
        if "=>" not in line:
            continue
        name, _, rest = line.strip().partition("=>")
        path = rest.strip().split(" ")[0]
        if path.startswith("/") and ("acados" in path or "mpc_planner" in path
                                     or "gsl" in path or "guidance" in path
                                     or "ros_tools" in path or "scenario" in path):
            resolved[name.strip()] = {"path": path, "sha256": sha256(Path(path))}
    return resolved


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snap = {"recorded_at": datetime.now().isoformat(timespec="seconds"),
            "host": os.uname().nodename, "repos": {}, "workspaces": {},
            "project_sources": {}, "modules": {}, "environment": {}}

    for name, repo in REPOS.items():
        if not (repo / ".git").exists():
            snap["repos"][name] = {"note": "not a git repository", "path": str(repo)}
            continue
        snap["repos"][name] = {
            "path": str(repo),
            "head": git(repo, "rev-parse", "HEAD"),
            "dirty": git(repo, "status", "--porcelain"),
            "untracked": git(repo, "ls-files", "--others", "--exclude-standard"),
            "diff_stat": git(repo, "diff", "--stat"),
        }

    for workspace in sorted(BUILD.glob("*")):
        if not (workspace / "src").is_dir():
            continue
        entry = {"path": str(workspace), "artefacts": {}}
        for label, path in artefacts(workspace).items():
            entry["artefacts"][label] = {"path": str(path), "sha256": sha256(path),
                                         "exists": path.exists()}
        binary = workspace / "devel/lib/mpc_planner_bridge/mpc_bridge"
        entry["loaded"] = loaded_shared_objects(binary)
        snap["workspaces"][workspace.name] = entry

    for path in PROJECT_SOURCES:
        snap["project_sources"][str(path)] = sha256(path)

    # Which module a name actually resolves to, not which one we expect.
    sys.path.insert(0, str(ROOT))
    for name in ("continuous_mpc_gate", "unicycle_mpc_gate", "modern_worker",
                 "evaluate_matched_safety"):
        try:
            module = __import__(name)
            snap["modules"][name] = getattr(module, "__file__", "<none>")
        except Exception as exc:
            snap["modules"][name] = f"<{type(exc).__name__}: {exc}>"

    for label, python in (("crowdnav", "/home/abc/miniconda3/envs/crowdnav/bin/python"),
                          ("sicnav", "/home/abc/miniconda3/envs/sicnav/bin/python")):
        try:
            out = subprocess.run(
                [python, "-c",
                 "import sys,numpy,scipy;print(sys.version.split()[0],numpy.__version__,scipy.__version__)"],
                capture_output=True, text=True, timeout=120).stdout.strip()
        except Exception as exc:
            out = f"<{type(exc).__name__}: {exc}>"
        snap["environment"][label] = out

    for label, tool in (("gcc-9", ["/usr/bin/gcc-9", "--version"]),
                        ("cmake", ["cmake", "--version"])):
        try:
            snap["environment"][label] = subprocess.run(
                tool, capture_output=True, text=True, timeout=60).stdout.splitlines()[0]
        except Exception as exc:
            snap["environment"][label] = f"<{type(exc).__name__}: {exc}>"

    target = OUT / "preflight.json"
    target.write_text(json.dumps(snap, indent=1, ensure_ascii=False, sort_keys=True))

    print(f"快照 -> {target}")
    print(f"\n仓库 HEAD 与脏状态：")
    for name, info in snap["repos"].items():
        if "head" not in info:
            print(f"  {name:18s} {info['note']}")
            continue
        dirty = len(info["dirty"].splitlines())
        untracked = len(info["untracked"].splitlines())
        print(f"  {name:18s} {info['head'][:12]}  改动 {dirty} 个文件，未跟踪 {untracked} 个")
    print(f"\n工作副本产物：")
    for name, entry in snap["workspaces"].items():
        missing = [k for k, v in entry["artefacts"].items() if not v["exists"]]
        loaded = len(entry["loaded"] or {})
        print(f"  {name:16s} 产物齐 {len(entry['artefacts'])-len(missing)}/{len(entry['artefacts'])}"
              + (f"，缺 {missing}" if missing else "") + f"，实际加载相关 .so {loaded} 个")
    print(f"\n模块解析：")
    for name, path in snap["modules"].items():
        print(f"  {name:24s} {path}")
    print(f"\n环境：")
    for name, value in snap["environment"].items():
        print(f"  {name:10s} {value}")


if __name__ == "__main__":
    main()
