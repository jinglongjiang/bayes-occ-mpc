#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/abc/miniconda3/envs/crowdnav/bin/python -u - <<'PY'
import concurrent.futures as futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import shlex
import sys
import threading
import time
import numpy as np
import scipy

root = Path.cwd()
resume = os.environ.get("BAYES_RESUME_OUT")
out = Path(resume) if resume else root / "results" / ("matched_safety_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
source = out / "source"
if not resume:
    source.mkdir(parents=True)
    for name in ("continuous_mpc_gate.py","bayesian_rfs.py","test_bayesian_rfs.py",
                 "evaluate_matched_safety.py","analyze_paired.py","run_remaining_suite.sh"):
        shutil.copy2(root/name,source/name)
    shutil.copy2(root/"results/pre_route_upgrade/source/continuous_mpc_gate.py",source/"legacy_search.py")
    crowdnav = Path("/home/abc/workspace/nav_data/mamba/camrl/CrowdNav")
    environment = source/"environment"
    for path in (crowdnav/"crowd_sim").rglob("*.py"):
        target = environment/path.relative_to(crowdnav)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    for name in ("crowd_nav/__init__.py","crowd_nav/contracts.py","crowd_nav/configs/env.config"):
        target = environment/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(crowdnav/name,target)
sys.path.insert(0,str(source))
from evaluate_matched_safety import save,SCENES,METHODS,RISK_SCALES,config,make_report
from bayesian_rfs import RFSConfig
from dataclasses import asdict
started = time.time()
last_note = started
records = []
active = {}
lock = threading.Lock()
stop = threading.Event()
manifest = {
    "started":dt.datetime.now().isoformat(),"python":sys.executable,"scipy":scipy.__version__,"numpy":np.__version__,
    "device":"Local Ryzen 7 5700G CPU; no GPU training, Webots GUI or robot actuation",
    "concurrency":"4 subprocess jobs x 2 episode workers, single-thread BLAS; latency serial",
    "source_sha256":{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()},
    "case_ranges":{"calibration_fit":[3500,3549],"calibration_validation":[3550,3599],
                   "navigation_development":[3600,3699],"formal":[4000,4099],
                   "noise":[4200,4299],"solver_seeds":[4300,4399],"latency":[4400,4419]},
    "scene_protocol":SCENES,"formal_episodes_per_scene_method_point":100,"risk_scales":RISK_SCALES,
    "fixed_radius_grid":[.15,.25,.35,.45,.60],"methods":METHODS,
    "primary_selection":"Within each method, fastest audited penalized time among dev settings with audited collision <=1%; otherwise lowest collision, then fastest. All five settings still evaluated and reported on test.",
    "primary_inference":"Bayes vs static covariance, EWMA, conformal and geometric fixed, DEV-selected. Collision noninferiority margin +1 percentage point and lower audited time, approximate one-sided case-cluster bootstrap bounds Bonferroni-adjusted for 8 endpoints. Insufficient evidence is INCONCLUSIVE, not automatic algorithm death.",
    "ablation":"Old/new sampling plan ONLY x fixed/full covariance; same filter, dynamics, probability integration, numerical environment, 512 samples x4 iterations.",
    "collision_audit":"Original benchmark events unchanged; independent actual pre/post swept-motion overlap per step, report both. No claim of complete physically corrected rollout beyond benchmark terminal event.",
    "prior_exposure":"The design and existing risk parameters have seen earlier 20-person runs. Five-person new fitting does NOT erase that history or establish pristine 5-to-20 generalization.",
    "training":False,"robot_visible":False,"external_baselines":"SARL/LSTM historical references only; no rerun requested.",
    "stop_policy":"Stop only on execution/data-contract failure, not disappointing performance. No post-test parameter selection.",
}
if resume:
    manifest = json.loads((out/"manifest.json").read_text())
    started = dt.datetime.fromisoformat(manifest["started"]).timestamp()
    prior_records = {r["name"]:r for r in json.loads((out/"status.json").read_text())["completed"]}
else:
    save(out/"manifest.json",manifest)
    prior_records = {}

remote_host = os.environ.get("BAYES_REMOTE_HOST")
remote_root = os.environ.get("BAYES_REMOTE_ROOT", "/root/bayes_mpc_matched_20260906")
remote_python = os.environ.get("BAYES_REMOTE_PYTHON", "/root/venvs/bayes_mpc_eval_20260906/bin/python")
ssh = ["ssh","-S","/run/user/1000/bayes-4090-control.sock","-o","BatchMode=yes","-o","ConnectTimeout=15"]
local_slots = threading.BoundedSemaphore(4)
remote_slots = threading.BoundedSemaphore(3 if remote_host else 0)
serial_stage = False
if resume:
    for name,digest in manifest["source_sha256"].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("frozen source changed before resumption: "+name)
    save(out/"execution_migration.json",{
        "time":dt.datetime.now().isoformat(),"dispatcher_sha256":hashlib.sha256((root/"run_remaining_suite.sh").read_bytes()).hexdigest(),
        "resume_existing_outputs":True,"remote_host":remote_host,"remote_python":remote_python,
        "remote_root":remote_root,"local_workers":8,"remote_workers":6 if remote_host else 0,
        "remote_affinity":"2-7, nice 10; no GPU access or modification of ECG process",
        "algorithm_and_protocol_unchanged":True,"latency":"Always local, serial after all distributed jobs complete"})

def note(stage):
    global last_note
    last_note = time.time()
    path = Path("/home/abc/temp/answer.md")
    current = path.read_text(encoding="utf-8")
    save(out/"handoff_snapshot.json",{"time":dt.datetime.now().isoformat(),
        "answer_sha256_before_append":hashlib.sha256(current.encode()).hexdigest(),
        "recent_text_for_review":current[-12000:],
        "meaning":"Recorded for later human/agent review; automatic progress writer does not reason about CC comments."})
    with path.open("a",encoding="utf-8") as stream:
        stream.write("\n\n## "+dt.datetime.now().strftime("%Y-%m-%d %H:%M KST")+"：公平对照队列自动进度\n\n")
        stream.write(f"阶段：{stage}；已完成{len(records)}个任务；已运行{(time.time()-started)/3600:.2f}小时。结果目录：`{out}`。\n")
        stream.write("工作：5人标定/开发集 → 六场景五档完整前沿 → 搜索×协方差消融 → 噪声/种子 → 单进程全管线耗时 → 配对统计与报告。所有回合保留官方判定和实际扫掠碰撞核验。\n")
        stream.write("自动记录不代表已解答CC的新意见；文件新增内容已保留，最近内容备份在handoff_snapshot.json供下一次人工复核。\n")

def job(name,method,point,scene,offset,count=100,search="new",extra=(),workers=2):
    cmd = [sys.executable,source/"evaluate_matched_safety.py","evaluate","--method",method,
           "--point",point,"--scene",scene,"--search",search,"--offset",offset,"--count",count,
           "--workers",workers,"--calibration",out/"calibration.json",*extra,"--output",out/(name+".json")]
    return name,cmd,list(range(offset,offset+count))

def run(item):
    name,cmd,expected = item
    if stop.is_set(): raise RuntimeError("queue cancelled")
    begin = time.time()
    path = out/(name+".json")
    cached = bool(resume and ((expected is not None and path.exists())
                  or (expected is None and name in prior_records)))
    device = "cached"
    if not cached:
        while True:
            if stop.is_set(): raise RuntimeError("queue cancelled")
            if local_slots.acquire(blocking=False):
                slot,device = local_slots,"local"
                break
            if remote_host and not serial_stage and expected is not None and remote_slots.acquire(blocking=False):
                slot,device = remote_slots,"4090-cpu"
                break
            time.sleep(.1)
        actual_cmd = list(map(str,cmd))
        if device == "4090-cpu":
            actual_cmd = [remote_python]+[str(x).replace(str(out),remote_root) for x in cmd[1:]]
            remote_cmd = ["timeout","--kill-after=15s","5400","taskset","-c","2-7","nice","-n","10","env",
                          "OPENBLAS_NUM_THREADS=1","OMP_NUM_THREADS=1","MKL_NUM_THREADS=1","CUDA_VISIBLE_DEVICES=",*actual_cmd]
            actual_cmd = [*ssh,remote_host,shlex.join(remote_cmd)]
        print("START",name,device,flush=True)
        try:
            with (out/(name+".log")).open("w") as log:
                process = subprocess.Popen(actual_cmd,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with lock: active[name] = process
                try:
                    code = process.wait(timeout=5500)
                    if code: raise RuntimeError(f"{name} exited {code}: see {name}.log")
                finally:
                    with lock: active.pop(name,None)
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        process.wait()
            if device == "4090-cpu":
                temporary = str(path)+".remote"
                with open(temporary,"wb") as stream:
                    subprocess.run([*ssh,remote_host,"cat",f"{remote_root}/{name}.json"],stdout=stream,check=True)
                os.replace(temporary,path)
            save(out/"execution"/(name+".json"),{"device":device,"seconds":time.time()-begin,
                "host":remote_host if device=="4090-cpu" else os.uname().nodename})
        finally:
            slot.release()
    if expected is not None:
        payload = json.loads(path.read_text())
        if [r["case_id"] for r in payload["episodes"]] != expected:
            raise RuntimeError("incomplete/duplicate/misaligned episodes: "+name)
        if payload["protocol"]["robot_visible"]:
            raise RuntimeError("visibility protocol violated")
        for filename,digest in payload["protocol"]["source_sha256"].items():
            if digest != manifest["source_sha256"][filename]:
                raise RuntimeError("worker code hash mismatch: "+name+":"+filename)
        if payload["protocol"]["calibration_sha256"] != hashlib.sha256((out/"calibration.json").read_bytes()).hexdigest():
            raise RuntimeError("worker calibration mismatch: "+name)
    if cached and name in prior_records:
        return {**prior_records[name],"reused_on_resume":True}
    return {"name":name,"command":list(map(str,cmd)),"device":device,"seconds":time.time()-begin}

def stage(name,jobs,workers=4):
    global serial_stage
    serial_stage = workers==1
    if remote_host and workers>1: workers += 3
    print(dt.datetime.now().isoformat(),name,len(jobs),"jobs",flush=True)
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run,item):item[0] for item in jobs}
        try:
            while pending:
                done,_ = futures.wait(pending,timeout=30,return_when=futures.FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    record = future.result()
                    records.append(record)
                    print("DONE",record["name"],round(record["seconds"],1),"seconds",flush=True)
                save(out/"status.json",{"stage":name,"completed":records,"pending":list(pending.values()),"elapsed_seconds":time.time()-started})
                if time.time()-last_note >= 5400: note(name)
        except BaseException:
            stop.set()
            with lock:
                for process in active.values():
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
            for future in pending: future.cancel()
            raise

try:
    note("冻结协议、源码和环境，准备自检")
    stage("unit_contracts",[("selfcheck",[sys.executable,"-m","unittest","-v","test_bayesian_rfs.py"],None)],1)
    stage("five_person_calibration",[("calibration",[sys.executable,source/"evaluate_matched_safety.py","calibrate",
        "--offset",3500,"--count",100,"--workers",8,"--output",out/"calibration.json"],None)],1)
    # Production-noise, both search implementations, and every adapter get a
    # bounded integration smoke before the large grid. These cases are never formal.
    stage("integration_smoke",[job("smoke_"+method,method,2,3,3993,count=2,
          extra=("--position-noise",.1,"--velocity-noise",.2,"--detection-probability",.8)) for method in METHODS]
          +[job("smoke_old", "static_cov",2,3,3993,count=2,search="old")])
    stage("five_person_navigation_development",[job(f"dev_{method}_p{point}",method,point,0,3600)
          for method in METHODS for point in range(5)])
    points,dev = {},{}
    for method in METHODS:
        values = [json.loads((out/f"dev_{method}_p{point}.json").read_text())["summary"] for point in range(5)]
        feasible = [i for i,v in enumerate(values) if v["collision_union"] <= .01]
        if feasible:
            point = min(feasible,key=lambda i:(values[i]["audited_penalized_time"],-values[i]["success_without_overlap"],i))
        else:
            point = min(range(5),key=lambda i:(values[i]["collision_union"],values[i]["audited_penalized_time"],i))
        points[method] = point
        dev[method] = values
    selection_path = out/"frozen_selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text())["points"] != points:
            raise RuntimeError("resumed development selection differs from frozen selection")
    else:
        save(selection_path,{"points":points,"development":dev,"formal_not_started":True})
    bundle = {"schema":1,"controller":"ContinuousCEMMPC","belief":"Bernoulli-Gaussian detected tracks",
        "planner":asdict(config(points["bayes"])),"filter":asdict(RFSConfig()),
        "learned_weights":None,"runtime_state":"Tracks and previous CEM trajectory reset on a new task; not a pretrained neural checkpoint",
        "runtime":{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__},
        "source_sha256":manifest["source_sha256"],
        "deployment_status":"Holonomic CrowdSim bundle only. NOT validated for Webots/TurtleBot differential drive, real sensor association or real-time actuation."}
    if not (out/"model_bundle.json").exists(): save(out/"model_bundle.json",bundle)
    stage("fresh_six_scene_complete_operating_grid",[job(f"formal_{method}_p{point}_{scene}",method,point,scene,4000)
          for method in METHODS for point in range(5) for scene in range(6)])
    make_report(out)
    stage("search_covariance_factorial_and_mean",[job(f"formal_{method}_old_{scene}",method,2,scene,4000,search="old")
          for method in ("bayes","static_cov") for scene in range(6)]
          +[job(f"formal_mean_{scene}","posterior_mean",2,scene,4000) for scene in range(6)])
    make_report(out)
    extra_jobs = []
    for level,pos,vel,detect in (("moderate",.05,.1,.9),("severe",.1,.2,.8)):
        for method in ("bayes","static_cov","ewma","conformal"):
            extra_jobs.append(job(f"noise_{level}_{method}",method,points[method],3,4200,
                extra=("--position-noise",pos,"--velocity-noise",vel,"--detection-probability",detect)))
    extra_jobs += [job(f"seed_{seed}","bayes",points["bayes"],3,4300,extra=("--seed",seed)) for seed in (0,1,2)]
    stage("noise_and_solver_seeds",extra_jobs)
    make_report(out)
    stage("serial_pipeline_latency",[job("latency_"+method,method,points[method],3,4400,count=20,workers=1)
          for method in METHODS],1)
    make_report(out)
    for name,digest in manifest["source_sha256"].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError("frozen source changed during run: "+name)
    save(out/"complete.json",{"jobs":len(records),"elapsed_seconds":time.time()-started})
    note("全部完成，report.md与analysis.json已更新；未启动实体控制")
    print("COMPLETE",out,flush=True)
except BaseException as error:
    save(out/"failed.json",{"error":repr(error),"elapsed_seconds":time.time()-started})
    note("执行异常，停止队列；详情见failed.json，不当作算法失败")
    raise
PY
