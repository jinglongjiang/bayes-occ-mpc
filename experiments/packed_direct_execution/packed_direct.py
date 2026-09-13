"""Opt-in execution backend for Direct at repo commit 82857b3.

No posterior/model/threshold changes.  No parallel reduction within a risk row.
This module was tested on synthetic inputs, not the user's full MPC pipeline.
"""
import ctypes as ct
import hashlib
import os
import subprocess
import threading
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
_LOCK = threading.Lock()
_LIBRARY = None


def library():
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        source = HERE / 'risk_execution.cpp'
        build = HERE / '_build'
        build.mkdir(exist_ok=True)
        flags = ['-O3', '-std=c++11', '-shared', '-fPIC', '-ffp-contract=off', '-fopenmp']
        compiler = os.environ.get('CXX', 'g++')
        version = subprocess.check_output([compiler, '--version'], text=True)
        fingerprint = hashlib.sha256(source.read_bytes() + repr(flags).encode() + version.encode()).hexdigest()
        target, stamp = build / 'risk_execution.so', build / 'sha256.txt'
        if not target.exists() or not stamp.exists() or stamp.read_text().strip() != fingerprint:
            temporary = build / ('risk_execution.%d.so' % os.getpid())
            subprocess.run([compiler] + flags + [str(source), '-o', str(temporary)], check=True)
            os.replace(str(temporary), str(target))
            stamp.write_text(fingerprint + '\n')
        lib = ct.CDLL(str(target))
        p = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        q = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        lib.direct_context_create.argtypes = [p,p,p,p,q,p,p,ct.c_int,ct.c_int,ct.c_double,p,q,p,ct.c_int,ct.c_int]
        lib.direct_context_create.restype = ct.c_void_p
        lib.direct_context_free.argtypes = [ct.c_void_p]
        lib.direct_context_free.restype = None
        lib.direct_query.argtypes = [ct.c_void_p,p,ct.c_int,ct.c_int,p]
        lib.direct_query.restype = ct.c_int
        lib.direct_last_cpus.argtypes = [ct.c_void_p,ct.POINTER(ct.c_int),ct.c_int]
        lib.direct_last_cpus.restype = ct.c_int
        _LIBRARY = lib
        return lib


class QueryContext:
    def __init__(self, evaluator):
        self._pointer = None
        self._library = library()
        h = int(evaluator.cfg.horizon)
        weights = np.ascontiguousarray(evaluator.weights, dtype=np.float64).reshape(-1)
        offsets = np.ascontiguousarray(evaluator.offsets, dtype=np.int64).reshape(-1)
        existence = np.ascontiguousarray(evaluator.r, dtype=np.float64).reshape(-1)
        modes, people = len(weights), len(existence)
        self.horizon, self.modes, self.people = h, modes, people
        if h <= 0 or len(offsets) != people + 1 or offsets[0] != 0 or offsets[-1] != modes:
            raise ValueError('invalid horizon/offsets')
        if people and (np.diff(offsets) <= 0).any():
            raise ValueError('empty per-person mode group')
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('invalid mode weights')
        if not np.isfinite(existence).all() or ((existence < 0) | (existence > 1)).any():
            raise ValueError('invalid existence weights')
        if people == 0:
            return
        table_obj = evaluator.table
        nodes, spacing = len(table_obj.grid), float(table_obj.step)
        arrays = {
            'centers': np.ascontiguousarray(evaluator.centers, dtype=np.float64),
            'variance': np.ascontiguousarray(evaluator.variance, dtype=np.float64),
            'sigma': np.ascontiguousarray(evaluator.sigma, dtype=np.float64),
            'radius': np.ascontiguousarray(evaluator.radius, dtype=np.float64).reshape(-1),
            'component': np.ascontiguousarray(evaluator.component, dtype=np.int64),
            'table': np.ascontiguousarray(table_obj.table, dtype=np.float64).reshape(-1, nodes),
            'derivative': np.ascontiguousarray(table_obj.derivative, dtype=np.float64).reshape(-1, nodes),
        }
        for key in ('centers', 'variance', 'sigma', 'radius', 'table', 'derivative'):
            if not np.isfinite(arrays[key]).all():
                raise ValueError('nonfinite input: ' + key)
        if arrays['centers'].shape != (modes, h, 2):
            raise ValueError('center shape mismatch')
        for key in ('variance', 'sigma', 'component'):
            if arrays[key].shape != (modes, h):
                raise ValueError('shape mismatch: ' + key)
        if arrays['radius'].shape != (modes,) or (arrays['radius'] < 0).any():
            raise ValueError('invalid radii')
        if (arrays['variance'] < 0).any() or (arrays['sigma'] < 0).any():
            raise ValueError('negative variance/sigma')
        if ((arrays['variance'] >= 1e-9) & (arrays['sigma'] <= 0)).any():
            raise ValueError('positive variance with zero sigma')
        if arrays['table'].shape != arrays['derivative'].shape:
            raise ValueError('table/derivative shape mismatch')
        for a, b in zip(offsets[:-1], offsets[1:]):
            if not np.isclose(weights[a:b].sum(), 1., atol=1e-10, rtol=0):
                raise ValueError('per-person weights not normalized')
        self._pointer = self._library.direct_context_create(
            arrays['centers'], arrays['variance'], arrays['sigma'], arrays['radius'],
            arrays['component'], arrays['table'], arrays['derivative'], len(arrays['table']),
            nodes, spacing, weights, offsets, existence, people, h)
        if not self._pointer:
            raise RuntimeError('native context construction failed')
        # The C++ object owns a copy: source arrays may safely go out of scope.

    def query(self, positions, threads=1):
        p = np.ascontiguousarray(positions, dtype=np.float64)
        if p.ndim != 3 or p.shape[1:] != (self.horizon, 2) or not np.isfinite(p).all():
            raise ValueError('invalid robot query positions')
        if not isinstance(threads, (int, np.integer)) or threads < 1:
            raise ValueError('threads must be a positive integer')
        out = np.zeros(p.shape[:2], dtype=np.float64)
        if not self.people or not len(p):
            return out
        if not self._pointer:
            raise RuntimeError('context already closed')
        result = self._library.direct_query(self._pointer, p, len(p), int(threads), out)
        if result != 0 or not np.isfinite(out).all():
            raise FloatingPointError('invalid native direct risk output')
        return out

    def close(self):
        pointer = getattr(self, '_pointer', None)
        if pointer:
            self._library.direct_context_free(pointer)
            self._pointer = None

    def last_cpus(self):
        values=(ct.c_int*256)()
        count=self._library.direct_last_cpus(self._pointer,values,256)
        if count<0:raise RuntimeError('CPU placement output overflow')
        return list(values[:count])

    def __del__(self):
        self.close()


def make_direct_class(base_direct, threads=1):
    """Return an experimental subclass of experiments.intent_risk_query.Direct.

    Build the library BEFORE timed calls. Instantiate this evaluator once per
    current posterior, including packing/validation in complete pipeline timing.
    Preserve the project's ApproxPlanner full-evaluation path.
    """
    library()

    class PackedDirect(base_direct):
        def __init__(self, *args, **kwargs):
            started = time.perf_counter()
            super().__init__(*args, **kwargs)
            self.query_context = QueryContext(self)
            self.build_ms = 1000. * (time.perf_counter() - started)

        def interval(self, positions):
            started = time.perf_counter()
            out = self.query_context.query(positions, threads=threads)
            self.query_ms += 1000. * (time.perf_counter() - started)
            self.counts[0] += len(out) * self.cfg.horizon * len(self.weights)
            # This is the existing point-query API, NOT an enclosure certificate.
            return out, out

    PackedDirect.__name__ = 'PackedDirect_%dThreads' % threads
    return PackedDirect
