#!/usr/bin/env python3
"""Fixed benchmark for the autoresearch loop on Option B (ControlNet) on Seamless. Do not modify.

Plays the role of prepare.py in karpathy/autoresearch: the training budget, the val pairs and
the evaluation are fixed here, so every run is comparable. The loop edits the model
(framework/model/model_dualtalk_controlnet.py, configs/model/model_dualtalk_controlnet.yaml),
never this file, val300.json, evaluate_seamless.py or seamless_metrics.py. Every metric is
lower-is-better; L_FDD is the headline number. The test split is never used.

One run:
    1. train Option B with train_all.py (as run_seamless_controlnet.sh) from the fixed Stage-1
       (1e0b2d6h, best-epoch=199) and speaking model (2rufbto2, latest-epoch=200):
       10 epochs x 600 batches, seed 1234, DDP on GPUs 0,1, OMP / MKL threads 8
    2. generate speaker A of the 300 fixed val pairs (val300.json) from ckpt=latest with
       evaluate_seamless.py, one shard per GPU
    3. score them: seamless_metrics.py --split val --pairs val300.json
    Any failed step, or the whole run going past 60 min, kills everything: "L_FDD: crash".

    cd /home/mpate169/Mayur/LiveFace_listen_speak/project
    /home/mpate169/miniconda3/envs/liveface/bin/python3.9 autoresearch/bench_controlnet.py > run.log 2>&1
    grep "^L_FDD:\|^train_min:" run.log

stdout is exactly
    L_FDD: 1.2345   S_LVE: 1.2345   S_FDD: 1.2345   L_PDD: 1.2345   L_FID: 1.2345
    train_min: 20.1   eval_min: 17.3
(train_min = wall clock of step 1, eval_min = steps 2 + 3) or "L_FDD: crash", with the reason
and the log to read on stderr. Logs (bench_*.log), config and tensorboard of a run:
outputs/Seamless/autoresearch_controlnet/<run_id>/; predictions and seamless_metrics.json:
results/seamless/autoresearch_controlnet/<run_id>/val_bench300/. The run's checkpoints (~2 GB)
are deleted when it ends: /home is a shared disk that is nearly full.

val300.json was made once with --make-pairs: 300 of the 543 val pairs longer than 101 frames
(the metric's rule), random.Random(0).sample, sorted.
"""
import argparse
import fcntl
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # project/
ENVS = Path("/home/mpate169/miniconda3/envs")
PY = ENVS / "liveface/bin/python3.9"                 # training + generation
PY_METRICS = ENVS / "ARTalk/bin/python"              # seamless_metrics.py, as run_seamless_eval.sh

# ---- fixed benchmark constants -------------------------------------------------------------
STAGE1 = ROOT / "outputs/Seamless/seamless_vqvae_prior/1e0b2d6h"
STAGE1_CKPT = STAGE1 / "tensorboard/1e0b2d6h/checkpoints/best-epoch=199.ckpt"
SPEAK_CKPT = ROOT / "outputs/Seamless/seamless_speak/2rufbto2/tensorboard/2rufbto2/checkpoints/latest-epoch=200.ckpt"
EPOCHS = 10
BATCHES_PER_EPOCH = 600
SEED = 1234
GPUS = (0, 1)                                        # 2 and 3 belong to other users
THREADS = 8
TIME_LIMIT_MIN = 60
PAIRS_FILE = HERE / "val300.json"
N_PAIRS = 300
PAIRS_SEED = 0
MIN_LEN = 101                                        # seamless_metrics.MIN_LEN
EXPERIMENT = "autoresearch_controlnet"
TAG = "_bench300"
REPORT = ("L_FDD", "S_LVE", "S_FDD", "L_PDD", "L_FID")
SEAMLESS_PAIRS = ROOT / "datasets/unilstalk/SeamlessInteractionTalk/seamless_interaction_pairs.json"

ENV = {**os.environ, "PATH": f"{PY.parent}:{os.environ.get('PATH', '')}",
       **{k: str(THREADS) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                    "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}}
LIVE = []                                            # every process group this run started


class Crash(Exception):
    pass


def make_pairs():
    if PAIRS_FILE.exists():
        sys.exit(f"{PAIRS_FILE} already exists: the benchmark pairs are fixed")
    meta = json.load(open(SEAMLESS_PAIRS))["val"]
    val = sorted([Path(a).name, Path(b).name, n] for a, b, n in meta.values() if n > MIN_LEN)
    pick = sorted(random.Random(PAIRS_SEED).sample(val, N_PAIRS))
    rows = ",\n  ".join(json.dumps(p) for p in pick)
    PAIRS_FILE.write_text(f'{{"split": "val", "seed": {PAIRS_SEED}, "of": {len(val)}, "pairs": [\n  {rows}\n]}}\n')
    assert json.load(open(PAIRS_FILE))["pairs"] == pick
    print(f"{PAIRS_FILE}: {N_PAIRS} of {len(val)} val pairs")


def load_pairs():
    pairs = json.load(open(PAIRS_FILE))
    keys = [p[0] for p in pairs["pairs"]]
    meta = json.load(open(SEAMLESS_PAIRS))["val"]
    val = {Path(a).name for a, _, n in meta.values() if n > MIN_LEN}
    if pairs["split"] != "val" or len(set(keys)) != N_PAIRS or not set(keys) <= val:
        raise Crash(f"{PAIRS_FILE} is not {N_PAIRS} distinct val pairs")
    return keys


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run(jobs, deadline):
    """Start [(cmd, extra env, log)] together; return once all exit 0."""
    procs = []
    for cmd, extra, log in jobs:
        p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, env={**ENV, **extra},
                             stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((p, log))
        LIVE.append(p)
    while True:
        codes = [p.poll() for p, _ in procs]
        for c, (_, log) in zip(codes, procs):
            if c not in (None, 0):
                raise Crash(f"exit code {c}, see {log}")
        if all(c == 0 for c in codes):
            return
        if time.time() > deadline:
            raise Crash(f"over the {TIME_LIMIT_MIN} min limit, killed (see {procs[0][1]})")
        time.sleep(5)


def kill_all():
    """SIGTERM, then SIGKILL, every process group started (DDP rank 1 is in rank 0's group)."""
    def alive(p):
        p.poll()
        try:
            os.killpg(p.pid, 0)
            return True
        except ProcessLookupError:
            return False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for p in LIVE:
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                pass
        end = time.time() + 30
        while sig == signal.SIGTERM and time.time() < end and any(alive(p) for p in LIVE):
            time.sleep(1)


def check_predictions(pred_dir, keys):
    import numpy as np
    for k in keys:
        f = pred_dir / f"{k}.npy"
        if not f.exists():
            raise Crash(f"missing prediction {f}")
        if not np.isfinite(np.load(f)).all():
            raise Crash(f"non-finite values in {f}")


def bench():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT_MIN * 60
    keys = load_pairs()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "outputs/Seamless" / EXPERIMENT / run_id
    pred_dir = ROOT / "results/seamless" / EXPERIMENT / run_id / f"val{TAG}"
    run_dir.mkdir(parents=True)
    try:
        # 1. train (same overrides as run_seamless_controlnet.sh, plus the fixed budget and seed)
        run([([PY, "train_all.py",
               f"experiment={EXPERIMENT}", f"run_id={run_id}", "state=new", f"seed={SEED}",
               "data=seamless_pred_moshi_text", "data.load_partner=true",
               "model=model_dualtalk_controlnet",
               f"model.duplex.speak_ckpt_path='{SPEAK_CKPT}'",
               f"model.folder_prior={STAGE1}", "model.version_prior=0",
               f"model.ckpt_path_prior='{STAGE1_CKPT}'",
               "callback.best_epoch_ckpt.monitor=total/val", "~callback.early_stopping",
               "model.loss.exp_dims=100",
               f"trainer.devices={len(GPUS)}", f"trainer.max_epochs={EPOCHS}",
               f"+trainer.limit_train_batches={BATCHES_PER_EPOCH}",
               "+trainer.strategy=ddp_find_unused_parameters_true"],
              {"CUDA_VISIBLE_DEVICES": ",".join(map(str, GPUS)), "MASTER_PORT": str(free_port())},
              run_dir / "bench_train.log")], deadline)
        t1 = time.time()
        latest = run_dir / "tensorboard" / run_id / "checkpoints" / f"latest-epoch={EPOCHS - 1}.ckpt"
        if not latest.exists():
            raise Crash(f"training ended without {latest.name}, see {run_dir / 'bench_train.log'}")

        # 2. generate the fixed val pairs, one shard per GPU
        run([([PY, "evaluate_seamless.py", f"folder={run_dir}", "ckpt=latest", "split=val",
               f"pairs={PAIRS_FILE}", f"tag={TAG}", "device=0",
               f"shard={s}", f"num_shards={len(GPUS)}"],
              {"CUDA_VISIBLE_DEVICES": str(g)}, run_dir / f"bench_generate{s}.log")
             for s, g in enumerate(GPUS)], deadline)
        check_predictions(pred_dir, keys)

        # 3. score
        run([([PY_METRICS, "seamless_metrics.py", "--split", "val", "--pairs", PAIRS_FILE,
               "--pred", pred_dir],
              {"CUDA_VISIBLE_DEVICES": str(GPUS[0])}, run_dir / "bench_metrics.log")], deadline)
        m = json.load(open(pred_dir / "seamless_metrics.json"))
        if m["pairs"] != N_PAIRS:
            raise Crash(f"scored {m['pairs']} pairs, expected {N_PAIRS}")
        t2 = time.time()
        if t2 > deadline:
            raise Crash(f"over the {TIME_LIMIT_MIN} min limit")
    finally:
        kill_all()
        for f in (run_dir / "tensorboard").glob("*/checkpoints/*.ckpt"):
            f.unlink()
    print("   ".join(f"{k}: {m[k]:.4f}" for k in REPORT))
    print(f"train_min: {(t1 - t0) / 60:.1f}   eval_min: {(t2 - t1) / 60:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--make-pairs", action="store_true", help=f"write {PAIRS_FILE.name} (once)")
    if ap.parse_args().make_pairs:
        return make_pairs()

    def on_signal(signum, _):
        raise Crash(f"received signal {signum}")
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        lock = open(HERE / ".bench.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Crash(f"another benchmark run is using GPUs {GPUS}")
        bench()
    except Exception as e:
        print("L_FDD: crash", flush=True)
        print(f"crash: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
