"""The frozen development-candidate manifest (order section 13.5).

Locked before the first D1 episode is run.  The order forbids filling this in
while looking at results, so the twelve candidates per method family and the
rule that produced them are written down here once and the sweep reads them.

What the twelve have to cover, per the order: risk, the physical prediction
horizon, and the native tracking/terminal weight configuration.  Each family's
set also has to contain the author's recommended configuration adapted to the
common task and actuator box, and the current working point, so that neither is
quietly dropped.

Two things learned during interface acceptance decide the axes:

  * the prediction horizon is fixed at code-generation time, so N=16/24/32
    (4/6/8 s on the common 0.25 s grid) are three separately generated
    workspaces, hashed with their generator configuration;

  * the reference path's continuation past the goal is a real interfacing
    parameter, not a constant.  A traced episode showed 2.0 m carrying the robot
    through the goal region and on to the spline's end (closest approach 0.47 m,
    never inside 0.25 m) while 0.0 m had it stabilise short.  With the native
    contour weight of 0.05 against a lag weight of 0.75 the controller tracks
    its line with several tenths of a metre of lateral error, so which of the
    two is better cannot be settled by inspection.  It is swept.

Risk is expressed in each method's own units.  SH-MPC doubles the configured
value internally (scenario_module/src/config.cpp:27), so the same number means
different things in the two families; the effective value is recorded alongside.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

BUILD = Path("/home/abc/workspace/bayes_occ_mpc/build_modern")
OUT = Path("/home/abc/temp/modern/candidates")

# Each method family's current working point, in its own yaml units, as selected
# by the earlier development sweep.
CURRENT_RISK = {"tmpc": 0.35, "shmpc": 0.42}
# What the shipped settings.yaml carries.
AUTHOR_RISK = 0.05
AUTHOR_CONTOUR, AUTHOR_TERMINAL = 0.05, 10.0


@dataclass(frozen=True)
class Candidate:
    """One development candidate.  `workspace` fixes N and the solver binary;
    everything else is read at runtime."""
    candidate_id: str
    family: str                 # tmpc | shmpc | bayes
    horizon: int
    workspace: Optional[str]    # None for the Bayesian arm
    risk: Optional[float]       # yaml units for the comparators
    contour: Optional[float]
    terminal_contouring: Optional[float]
    path_overshoot: Optional[float]
    # Bayesian arm only.
    risk_point: Optional[int] = None
    population: Optional[int] = None
    iterations: Optional[int] = None
    rationale: str = ""

    @property
    def effective_risk(self) -> Optional[float]:
        if self.risk is None:
            return None
        return self.risk * (2.0 if self.family == "shmpc" else 1.0)


def _comparator_family(family: str) -> Tuple[Candidate, ...]:
    base = family
    n24, n32 = f"{family}_n24", f"{family}_n32"
    own = CURRENT_RISK[family]
    A, C, T = AUTHOR_RISK, AUTHOR_CONTOUR, AUTHOR_TERMINAL
    rows = [
        # id                 N   ws    risk  contour terminal  overshoot  why
        ("c01_author",       16, base,   A,     C,      T,       0.0,
         "author's shipped configuration, adapted only to the common actuator box"),
        ("c02_author_reach", 16, base,   A,     C,      T,       2.0,
         "same, with the reference continued through the goal"),
        ("c03_current",      16, base, own,     C,      T,       2.0,
         "the current working point carried forward from the earlier sweep"),
        ("c04_current_stop", 16, base, own,     C,      T,       0.0,
         "current risk with the reference ending at the goal"),
        ("c05_risk20",       16, base, 0.20,    C,      T,       0.0,
         "midpoint risk, native tracking"),
        ("c06_risk20_reach", 16, base, 0.20,    C,      T,       2.0,
         "midpoint risk, reference through the goal"),
        ("c07_tight",        16, base, 0.20,  0.50,   50.0,      0.0,
         "tracking weighted 10x harder: contour error, not progress, dominates"),
        ("c08_tight_reach",  16, base, 0.20,  0.50,   50.0,      2.0,
         "tight tracking with the reference through the goal"),
        ("c09_verytight",    16, base, 0.20,  2.00,  100.0,      0.0,
         "tracking weighted 40x: the limit of holding the line"),
        ("c10_n24",          24, n24,  0.20,    C,      T,       2.0,
         "6 s horizon, native tracking"),
        ("c11_n24_tight",    24, n24,  0.20,  0.50,   50.0,      2.0,
         "6 s horizon with tight tracking"),
        ("c12_n32_tight",    32, n32,  0.20,  0.50,   50.0,      2.0,
         "8 s horizon with tight tracking"),
    ]
    return tuple(Candidate(cid, family, n, ws, risk, contour, terminal, over,
                           rationale=why)
                 for cid, n, ws, risk, contour, terminal, over, why in rows)


def _bayes_family() -> Tuple[Candidate, ...]:
    """The same budget for our own method, over the parameters the order allows:
    risk operating point, prediction horizon, and solver budget.  The risk
    model, the filter and the cost terms are not touched."""
    rows = [
        # id                 N   point pop  iters  why
        ("c01_p2_n16",       16, 2,    512,  4, "the registered working point"),
        ("c02_p1_n16",       16, 1,    512,  4, "one step more conservative"),
        ("c03_p3_n16",       16, 3,    512,  4, "one step more permissive"),
        ("c04_p0_n16",       16, 0,    512,  4, "most conservative registered point"),
        ("c05_p4_n16",       16, 4,    512,  4, "most permissive registered point"),
        ("c06_p2_n24",       24, 2,    512,  4, "6 s horizon at the working point"),
        ("c07_p1_n24",       24, 1,    512,  4, "6 s horizon, conservative"),
        ("c08_p3_n24",       24, 3,    512,  4, "6 s horizon, permissive"),
        ("c09_p2_n32",       32, 2,    512,  4, "8 s horizon at the working point"),
        ("c10_p2_n16_small", 16, 2,    256,  4, "half the sample budget"),
        ("c11_p2_n16_big",   16, 2,   1024,  4, "double the sample budget"),
        ("c12_p2_n16_iter",  16, 2,    512,  6, "more CEM iterations, same samples"),
    ]
    return tuple(Candidate(cid, "bayes", n, None, None, None, None, None,
                           risk_point=point, population=pop, iterations=iters,
                           rationale=why)
                 for cid, n, point, pop, iters, why in rows)


CANDIDATES = {"tmpc": _comparator_family("tmpc"),
              "shmpc": _comparator_family("shmpc"),
              "bayes": _bayes_family()}


def settings_for(candidate: Candidate) -> Path:
    """Write (once) and return the settings.yaml this candidate runs with.

    The file is derived from its own workspace's shipped settings, so N and the
    generated solver always agree; only the runtime-read fields are edited.
    """
    if candidate.family == "bayes":
        raise ValueError("the Bayesian arm has no yaml settings")
    source = (BUILD / candidate.workspace / "src/mpc_planner"
              / "mpc_planner_jackalsimulator/config/settings.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{candidate.family}_{candidate.candidate_id}.yaml"
    text = source.read_text().splitlines()
    out, section = [], None
    for line in text:
        stripped = line.strip()
        if stripped and not line.startswith((" ", "\t")) and stripped.endswith(":"):
            section = stripped[:-1]
        if section == "probabilistic" and stripped.startswith("risk:"):
            line = f"  risk: {candidate.risk}"
        elif section == "weights" and stripped.startswith("contour:"):
            line = f"  contour: {candidate.contour}"
        elif section == "weights" and stripped.startswith("terminal_contouring:"):
            line = f"  terminal_contouring: {candidate.terminal_contouring}"
        out.append(line)
    body = "\n".join(out) + "\n"
    if not target.exists() or target.read_text() != body:
        target.write_text(body)
    return target


if __name__ == "__main__":
    import hashlib, json
    manifest = {}
    for family, rows in CANDIDATES.items():
        manifest[family] = []
        for c in rows:
            entry = {k: v for k, v in c.__dict__.items()}
            if family != "bayes":
                path = settings_for(c)
                entry["settings"] = str(path)
                entry["settings_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                entry["effective_risk"] = c.effective_risk
            manifest[family].append(entry)
    target = Path("/home/abc/temp/modern/snapshot/candidates.json")
    target.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    for family, rows in manifest.items():
        print(f"\n{family}: {len(rows)} 个候选")
        for e in rows:
            if family == "bayes":
                print(f"  {e['candidate_id']:18s} N={e['horizon']:2d} point={e['risk_point']} "
                      f"pop={e['population']:4d} iters={e['iterations']}  {e['rationale']}")
            else:
                print(f"  {e['candidate_id']:18s} N={e['horizon']:2d} risk={e['risk']:.2f}"
                      f"(有效{e['effective_risk']:.2f}) contour={e['contour']:.2f} "
                      f"term={e['terminal_contouring']:.0f} overshoot={e['path_overshoot']:.1f}")
    print(f"\n清单 -> {target}")
