"""Opt-in computational backend; no posterior, risk or control model changes."""
import ctypes as ct
import hashlib
import shutil
import subprocess
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import ncx2
from experiments import goal_posterior_probe as old
from experiments import intent_repair_run as repair

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'build_intent_backend'


def build():
    src=ROOT/'vendor/Python-RVO2/src'
    paths=sorted(src.glob('*.h'))+sorted(p for p in src.glob('*.cpp') if p.name!='rvo2.cpp')
    wrapper=ROOT/'experiments/intent_kernels.cpp'
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in paths+[wrapper])).hexdigest()
    target=BUILD/'kernels.so';stamp=BUILD/'source.sha256'
    if target.exists() and stamp.exists() and stamp.read_text()==digest:return target
    BUILD.mkdir(exist_ok=True)
    for p in paths:
        destination=BUILD/p.name
        if p.name in ('Agent.h','RVOSimulator.h','KdTree.h'):
            # Access-only friendship in generated headers; vendored sources stay byte-identical.
            text=p.read_text().replace('private:', 'private:\n        friend class BatchContext;',1)
            destination.write_text(text)
        else:shutil.copy2(p,destination)
    command=['g++','-O3','-std=c++11','-shared','-fPIC','-ffp-contract=off',
        '-I'+str(BUILD),str(wrapper)]+[str(BUILD/n) for n in ('Agent.cpp','KdTree.cpp','Obstacle.cpp','RVOSimulator.cpp')]+['-o',str(target)]
    subprocess.run(command,check=True)
    stamp.write_text(digest)
    return target


_lib=None
def library():
    global _lib
    if _lib is None:
        _lib=ct.CDLL(str(build()))
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        ints=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        _lib.context_create.argtypes=[ptr,ptr,ct.c_int];_lib.context_create.restype=ct.c_void_p
        _lib.context_free.argtypes=[ct.c_void_p];_lib.context_free.restype=None
        _lib.behavior_forward.argtypes=[ct.c_void_p,ptr,ptr,ct.c_int,ptr,ct.c_int,ct.c_int,ct.c_double,ptr]
        _lib.fused_bounds.argtypes=[ptr,ct.c_int,ct.c_int,ptr,ct.c_int,ptr,ptr,ptr,ptr,ints,
            ptr,ptr,ct.c_int,ct.c_double,ct.c_double,ptr,ints,ptr,ct.c_int,ct.c_double,ct.c_double,ptr,ptr]
        _lib.exact_prepare.argtypes=[ptr,ct.c_int,ct.c_int,ptr,ct.c_int,ptr,ptr,ct.c_double,ptr,ptr,ptr,ints]
        _lib.exact_prepare.restype=ct.c_int
        _lib.aggregate_exact.argtypes=[ptr,ct.c_int,ct.c_int,ct.c_int,ptr,ints,ptr,ct.c_int,ptr]
    return _lib


def doubles(value):return np.ascontiguousarray(value,dtype=np.float64)


class Context:
    def __init__(self,state,neighbors):
        self.lib=library();self.pointer=self.lib.context_create(doubles(state),doubles(neighbors),len(neighbors))
    def __del__(self):
        if getattr(self,'pointer',None):self.lib.context_free(self.pointer);self.pointer=None


def neighbors_at(f,k):
    return doubles([[e['px']+k*.25*e['vx'],e['py']+k*.25*e['vy'],e['vx'],e['vy'],e['radius']]
                    for e in f['neighbors']]).reshape(-1,5)


def ttc_ratio(pos,vel,radius,neighbors):
    closest=None
    for o in neighbors:
        px,py=o[:2]-pos;vx,vy=o[2:4]-vel
        a=vx*vx+vy*vy
        if a<1e-8:continue
        combined=radius+o[4]
        b=2.*(px*vx+py*vy);c=px*px+py*py-combined*combined
        disc=b*b-4.*a*c
        if disc<=0.:continue
        root=float(np.sqrt(disc));t1=(-b-root)/(2.*a);t2=(-b+root)/(2.*a)
        if t2<0.:continue
        t=t1 if t1>0. else t2
        if closest is None or t<closest:closest=t
    return 1. if closest is None or closest>=2. else np.clip(closest/2.,0.,1.)


class Behavior:
    def __init__(self):self.cache=OrderedDict();self.hits=0;self.builds=0
    def forward(self,f,goals,steps=1):
        unique,inverse=np.unique(np.asarray(goals).reshape(-1,2),axis=0,return_inverse=True)
        if not len(unique):return np.empty((0,steps,2))
        state=doubles(np.r_[f['pos'],f['vel'],f['radius']]);neighbors=neighbors_at(f,0)
        key=(state.tobytes(),neighbors.tobytes())
        if key not in self.cache:
            self.cache[key]=(Context(state,neighbors),ttc_ratio(state[:2],state[2:4],f['radius'],neighbors))
            self.builds+=1
            if len(self.cache)>4096:self.cache.popitem(last=False)
        else:self.hits+=1;self.cache.move_to_end(key)
        context,ratio=self.cache[key]
        output=np.empty((len(unique),steps,2))
        library().behavior_forward(context.pointer,state,neighbors,len(neighbors),doubles(unique),len(unique),steps,float(ratio),output)
        self.builds+=len(unique)*max(0,steps-1)
        return output[inverse]


class FusedEnvelope(repair.MixtureEnvelope):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        t=self.table
        self.centers=doubles(t.obs.human_segment_end);self.variance=doubles(t.variance)
        self.sigma=doubles(t.s);self.radius=doubles(t.radius.ravel());self.a=doubles(t.a)
        self.component=np.ascontiguousarray(t.component_map,dtype=np.int64)
        self.offsets=np.ascontiguousarray(self.offsets,dtype=np.int64)
        self.r=doubles(self.obs.human_existence).copy()
        if self.cfg.existence_override is not None:self.r[:]=self.cfg.existence_override
    def bounds(self,positions):
        start=time.perf_counter();p=doubles(positions);lo=np.empty(p.shape[:2]);hi=np.empty_like(lo);t=self.table
        library().fused_bounds(p,len(p),p.shape[1],self.centers,len(self.weights),self.variance,
            self.sigma,self.radius,self.a,self.component,t.table,t.derivative,len(t.grid),t.step,t.interpolation_error,
            self.weights,self.offsets,self.r,len(self.r),float(ndtr(-8.)),float(1.-np.exp(-32.)-1e-12),lo,hi)
        self.seconds+=time.perf_counter()-start
        return lo,hi
    def exact(self,positions):
        p=doubles(positions);m=len(self.weights);h=self.cfg.horizon
        chunk=max(1,262144//max(1,m*h));out=[]
        for first in range(0,len(p),chunk):
            block=p[first:first+chunk];size=len(block)*m*h
            q=np.empty(size);scaled=np.empty(size);nc=np.empty(size);slots=np.empty(size,dtype=np.int64)
            used=library().exact_prepare(block,len(block),h,self.centers,m,self.variance,self.radius,float(ndtr(-8.)),q,scaled,nc,slots)
            q[slots[:used]]=ncx2.cdf(scaled[:used],2.,nc[:used])
            if not np.isfinite(q).all():raise FloatingPointError('invalid exact probability')
            result=np.empty((len(block),h))
            library().aggregate_exact(q,len(block),h,m,self.weights,self.offsets,self.r,len(self.r),result)
            out.append(result)
        return np.concatenate(out)


@contextmanager
def enabled(behavior=True,risk=True):
    previous_forward,previous_envelope=old.forward,repair.MixtureEnvelope
    engine=Behavior()
    if behavior:old.forward=engine.forward
    if risk:repair.MixtureEnvelope=FusedEnvelope
    try:yield engine
    finally:old.forward=previous_forward;repair.MixtureEnvelope=previous_envelope
