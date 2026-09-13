"""Repository-local runtime paths and explicit historical hash verification."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROWD = ROOT / 'vendor/CrowdNav'
OLD_ROOT = '/home/abc/workspace/bayes_occ_mpc_hermite/'
OLD_CROWD = '/home/abc/workspace/nav_data/mamba/camrl/CrowdNav/'


def resolve_path(path):
    value = str(path)
    for prefix, root in ((OLD_ROOT, ROOT), (OLD_CROWD, CROWD)):
        if value.startswith(prefix):
            return root / value[len(prefix):]
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen(path, expected):
    actual = resolve_path(path)
    if sha(actual) == expected:
        return
    manifest = json.loads((ROOT / 'reproducibility/manifest.json').read_text())
    relative = str(actual.relative_to(ROOT))
    change = manifest['path_changes'].get(relative)
    if change and sha(actual) == change['current_sha256']:
        snapshots = {change['original_sha256']: change['original_snapshot']}
        snapshots.update(change.get('historical_snapshots', {}))
        snapshot = snapshots.get(expected)
        if snapshot and sha(ROOT / snapshot) == expected:
            return
    raise RuntimeError('frozen dependency mismatch: ' + str(path))
