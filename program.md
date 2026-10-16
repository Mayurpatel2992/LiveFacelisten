# autoresearch: Option B (ControlNet) listen+speak on Seamless

This is an experiment to have the LLM do its own research, on top of karpathy/autoresearch
(`../autoresearch/program.md`). The objective here is to improve Option B, the ControlNet
listen+speak model: a frozen speaking model, plus a trainable listening branch driven by the
partner's audio, gated by an own-voice classifier. It's trained and evaluated on Seamless
Interaction under the UniLS / FacePlex protocol.

All paths are relative to `project/`. `PY=/home/mpate169/miniconda3/envs/liveface/bin/python3.9`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `oct16`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Check the starting point**: `git status` on `main` must be clean, with `program.md` and `.gitignore` committed. Otherwise, a `git reset --hard` in the loop can delete them. If the tree is not clean, ask the human.
3. **Create the branch**: `git checkout -b autoresearch/<tag>` from `main`.
4. **Read the in-scope files**:
   - `README_listen_speak.md`: the Seamless section (data, protocol, the three models).
   - `autoresearch/bench_controlnet.py`: the fixed benchmark. Read it, never modify it.
   - `framework/model/model_dualtalk_controlnet.py`: Option B (the ControlNet branch, its zero-init connections, the gate). Editable.
   - `framework/model/model_dualtalk_duplex.py`: the shared listen+speak base (partner front-end, training step, cached streaming generation, causality test). Editable.
   - `configs/model/model_dualtalk_controlnet.yaml`: Option B's config. It inherits `model_dualtalk_speak.yaml`; put overrides here. Editable.
   - `framework/model/model_vqvae_arflow_pred.py`: the speaking model that Option B wraps. Read-only context.
5. **Verify the fixed inputs exist**: the Stage-1 and speaking-model checkpoints named in `autoresearch/bench_controlnet.py`, and `autoresearch/val300.json`.
6. **Check results.tsv**: it lives at `project/results.tsv`, is git-ignored, and already holds the `baseline` rows from the noise runs (see "Deciding keep / discard"). Do not delete it.
7. **Confirm and go**: confirm the setup looks good.

Once you get confirmation, kick off the experimentation.

## The benchmark (fixed)

One run of `autoresearch/bench_controlnet.py`:

1. Trains Option B from the fixed Stage-1 (`1e0b2d6h`) and speaking model (`2rufbto2`, latest checkpoint) for exactly **10 epochs × 600 batches**, seed 1234, with DDP on **GPUs 0,1**. The harness sets the OMP/MKL threads to 8.
2. Generates speaker A of the 300 fixed val pairs (`autoresearch/val300.json`) from the final checkpoint, one shard per GPU.
3. Scores them with `seamless_metrics.py --split val` (UniLS's own metric code).

It takes about 40 min. The harness kills it at **60 min** wall clock and prints `L_FDD: crash`. Launch it as:

```
$PY autoresearch/bench_controlnet.py > run.log 2>&1
```

The run takes ~40 min, so start it in the background and wait for it to exit. A foreground shell call will time out, and repeatedly reading `run.log` floods your context. When it finishes, it prints exactly:

```
L_FDD: 12.3456   S_LVE: 12.3456   S_FDD: 12.3456   L_PDD: 12.3456   L_FID: 12.3456
train_min: 20.0   eval_min: 18.0
```

or `L_FDD: crash`. The reason and the log to read are on the next line, `crash: ...`. Extract with:

```
grep "^L_FDD:\|^train_min:\|^crash:" run.log
```

Per-run logs (`bench_train.log`, `bench_generate0.log`, ...) are in `outputs/Seamless/autoresearch_controlnet/<run_id>/`. The run's checkpoints are deleted when it ends.

The budget is in batches, not seconds: a larger batch size sees more data per run but takes longer, and the 60-min limit still applies.

**The goal is simple: get the lowest L_FDD** (listening-frame facial dynamics deviation, lower is better). That must happen without hurting speaking: S_LVE (speaking lip vertex error) is a guard rail, not a target.

## What you CAN do

Edit only these files. Everything in them is fair game: architecture of the listening branch and gate, how the branch is injected, losses and their weights, optimizer, learning rate and schedule, regularisation, and other training hyperparameters.

- `framework/model/model_dualtalk_controlnet.py`
- `framework/model/model_dualtalk_duplex.py`. Option A (`model_dualtalk_unils.py`) shares this base, so it may change too. That is fine on this branch.
- `configs/model/model_dualtalk_controlnet.yaml`
- `configs/data/seamless_pred_moshi_text.yaml`, only the fields that shape training batches (e.g. batch size). Not `root`, the splits, or which streams are loaded.
- `configs/trainer/gpu.yaml`, e.g. gradient clipping, precision, gradient accumulation. `max_epochs`, `limit_train_batches` and `devices` are set by the benchmark on the command line, so editing them here has no effect.

## What you CANNOT do

- Modify the read-only files. These play the role of `prepare.py`: `autoresearch/bench_controlnet.py`, `autoresearch/val300.json`, `seamless_metrics.py`, `evaluate_seamless.py`, `configs/evaluation_seamless.yaml`, `test_duplex.py`, `train_all.py`, `deps/` (incl. `deps/UniLS`), `datasets/` (incl. the split, index and norm files), `pre_process/`, and the Stage-1 and speaking-model checkpoints and their run folders under `outputs/Seamless/`. Also leave `configs/model/model_dualtalk_speak.yaml` and `framework/model/model_vqvae_arflow_pred.py` alone: they define the frozen speaking model the checkpoint is loaded into.
- **Use the test split, in any way.** Do not generate it, score it, or read existing test results to choose what to try. Do not train on val either: the 300 benchmark pairs are val.
- **Give the model ground truth at inference.** Generation may use only the own and partner audio streams (Mimi codes, Moshi text tokens) and the clip length. Ground-truth motion and the speech masks (`vad`, `partner_vad`) are training labels only, because the metric splits speaking and listening frames with that same mask. So `duplex.gate_infer_source` must stay `pred` (or `ones`); `vad` is the oracle. Never special-case benchmark tracks or keys.
- Install new packages or add dependencies.

## Constraints

- **Stay causal.** The causality test must say `PASSED` before every benchmark run.
- **Trainable parameters ≤ 12,000,000.** The baseline has 8,961,409. Both checks come from one ~10 s CPU pre-flight run:

  ```
  O=$PWD/outputs/Seamless
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= $PY test_duplex.py \
    experiment=test_duplex state=new \
    data=seamless_pred_moshi_text data.load_partner=true data.tiny=true \
    model=model_dualtalk_controlnet \
    "model.duplex.speak_ckpt_path='$O/seamless_speak/2rufbto2/tensorboard/2rufbto2/checkpoints/latest-epoch=200.ckpt'" \
    model.folder_prior=$O/seamless_vqvae_prior/1e0b2d6h model.version_prior=0 \
    "model.ckpt_path_prior='$O/seamless_vqvae_prior/1e0b2d6h/tensorboard/1e0b2d6h/checkpoints/best-epoch=199.ckpt'" \
    > preflight.log 2>&1
  grep "DUPLEX CAUSALITY" preflight.log                                    # PASSED
  grep -a "\] - trainable " "$(ls -td outputs/Seamless/test_duplex/*/ | head -1)logs.out" | tail -1
  ```

  Use the **last** `trainable` line. The first one is the speaking model before it is frozen.
- **GPUs 0,1 only.** GPUs 2 and 3 belong to other users; the benchmark is fixed to 0,1. Run only one benchmark at a time. The harness takes a lock, and a second run prints `L_FDD: crash`.
- **Threads capped.** The harness sets them to 8. Export `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8` for anything else you run.
- **Check the load before each launch.** This node is shared: CPU overload has killed every job on it before. Run `uptime` and `nvidia-smi --query-gpu=index,memory.used --format=csv -i 0,1` first. If the 1-minute load average is above 64 (half of the 128 cores), or another process holds memory on GPU 0 or 1, wait 10 minutes and check again.
- **VRAM** is a soft constraint (A6000, 48 GB). Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

## Deciding keep / discard

Training is seeded, but it is not bit-exact on GPU, so the same code gives slightly different numbers. The noise was measured by running the unchanged baseline several times. Those runs are the `baseline` rows in `results.tsv`.

- **noise**: max − min of L_FDD over the `baseline` rows. Measured: _pending (filled in when the baseline runs finish)_.
- **best**: the lowest bar to beat. It starts at the mean L_FDD of the `baseline` rows and becomes the L_FDD of each new `keep`.
- **S_LVE limit**: 1.02 × the mean S_LVE of the `baseline` rows.

A change is **keep** only if **L_FDD < best − noise** and **S_LVE ≤ S_LVE limit**: a gate that hurts speaking is not a win. Anything else is **discard**. A run that prints `L_FDD: crash`, or fails the pre-flight, is **crash**.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. Removing code for an L_FDD within the noise? Keep. A gain just past the noise that adds 50 lines of hacks? Probably discard.

## Logging results

When an experiment is done, log it to `results.tsv`. It is tab-separated, NOT comma-separated, because commas break in descriptions. It is git-ignored: never `git add` it.

The TSV has a header row and 6 columns:

```
commit	L_FDD	S_LVE	L_FID	status	description
```

1. git commit hash (short, 7 chars)
2. L_FDD (4 decimals); use 0.0000 for crashes
3. S_LVE (4 decimals); use 0.0000 for crashes
4. L_FID (4 decimals); use 0.0000 for crashes
5. status: `keep`, `discard`, `crash` (or `baseline` for the noise runs)
6. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/oct16`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Implement **one** experimental idea by editing the editable files.
3. `git diff --name-only` must list only editable files. Stage them by name (`git add <files>`, never `git add -A`), then `git commit -m "<idea>"`.
4. Run the pre-flight (causality + parameter count). If it fails, fix it if the fix is trivial; otherwise log `crash` and `git reset --hard HEAD~1`.
5. Check the load average and GPUs 0,1, as above.
6. Run the benchmark: `$PY autoresearch/bench_controlnet.py > run.log 2>&1` (in the background; wait for it to exit).
7. Read out the results: `grep "^L_FDD:\|^train_min:\|^crash:" run.log`.
8. If it says `L_FDD: crash`, read the log named on the `crash:` line (`tail -n 50`) for the stack trace and attempt a fix. If you can't get things to work after a few attempts, give up on the idea.
9. Record the results in `results.tsv`.
10. If it is a **keep**, you "advance" the branch: the commit stays and its L_FDD is the new best.
11. If it is a **discard** or **crash**, `git reset --hard HEAD~1`.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: the harness enforces the 60-min limit itself. A killed run is a `crash`: discard and revert.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, look at the per-run loss curves in `outputs/Seamless/autoresearch_controlnet/<run_id>/`, try combining previous near-misses, try more radical changes to the listening branch or the gate. The loop runs until the human interrupts you, period.

Each experiment takes about 40 minutes, so expect roughly 1.5 per hour: about a dozen over a night.
