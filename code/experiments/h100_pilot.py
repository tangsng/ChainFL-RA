# -*- coding: utf-8 -*-
"""Parallel experiment driver for H100: runs a job list of run_fl invocations
with N workers, round-robin over GPUs. Each job is one (config, cell, seed).
Usage: python3 h100_pilot.py <jobs.txt> <n_workers>
jobs.txt: one arg-string per line, e.g.
  --algo chainfl --dataset mnist ... --wcap 0.15 --out ../results/raw_v2/xxx.csv
"""
import subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_one(idx_job):
    idx, argstr = idx_job
    gpu = idx % 2
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log = argstr.split("--out")[1].split()[0].strip()
    log = log.replace(".csv", ".log").replace("/raw_v2/", "/raw_v2/logs/")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as lf:
        p = subprocess.run([sys.executable, "-m", "experiments.run_fl"] + argstr.split(),
                           stdout=lf, stderr=subprocess.STDOUT, env=env)
    return idx, p.returncode, argstr

def main():
    jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(f"{len(jobs)} jobs, {nw} workers")
    failed = []
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(run_one, (i, j)) for i, j in enumerate(jobs)]
        done = 0
        for f in as_completed(futs):
            idx, rc, argstr = f.result()
            done += 1
            tag = "OK " if rc == 0 else "FAIL"
            if rc != 0:
                failed.append(argstr)
            out = argstr.split("--out")[1].split()[0]
            print(f"[{done}/{len(jobs)}] {tag} {os.path.basename(out)}", flush=True)
    print("FAILED:", len(failed))
    for a in failed:
        print("  ", a)

if __name__ == "__main__":
    main()
