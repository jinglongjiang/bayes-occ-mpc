"""Expose the acados NLP handles read-only so the bridge can dump a full iterate.

Order section 13.4-C requires a failing step to be saved with its complete
primal, dual and slack values.  Replaying "the current observation" and
asserting the internal state matched is exactly what the order rules out, and
the solver's iterate is private.  These accessors return the existing pointers
without copying or modifying anything, so no solve behaves differently.
"""
import pathlib, sys

MARKER = "nlpOutForDiagnostics"
ANCHOR = """        ocp_nlp_solver *nlpSolverForDiagnostics() const { return _nlp_solver; }"""
ADDITION = """        ocp_nlp_solver *nlpSolverForDiagnostics() const { return _nlp_solver; }

        // Read-only handles for the failing-step dump.  They hand back the
        // solver's own pointers; nothing is copied and nothing is written, so a
        // solve is bit-identical whether or not the dump is enabled.
        ocp_nlp_config *nlpConfigForDiagnostics() const { return _nlp_config; }
        ocp_nlp_dims *nlpDimsForDiagnostics() const { return _nlp_dims; }
        ocp_nlp_out *nlpOutForDiagnostics() const { return _nlp_out; }
        ocp_nlp_in *nlpInForDiagnostics() const { return _nlp_in; }"""

path = pathlib.Path(sys.argv[1])
text = path.read_text()
if MARKER in text:
    print(f"  已有诊断访问器: {path.parent.parent.parent.parent.name}")
    raise SystemExit(0)
if ANCHOR not in text:
    raise SystemExit(f"未找到锚点 in {path}")
path.write_text(text.replace(ANCHOR, ADDITION, 1))
print(f"  已加诊断访问器: {path}")
