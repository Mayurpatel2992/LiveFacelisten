# Listen + speak (DualTalk) experiment

Separate workspace: code was copied from `LiveFace_train_moshi_abl_exps_20261012`
on 2026-10-13; nothing in that folder is changed or imported from here
(shared read-only assets in `datasets/` are symlinks).

## Pipeline (tmux sessions)

| step | what | session | output |
|---|---|---|---|
| 0 | DualTalk download + unzip (HF `ZiqiaoPeng/DualTalk_Dataset`) | - | `datasets/dualtalk/raw/{train,test,ood}` |
| 0 | splits, own-track VAD, stats (`pre_process/dualtalk_prepare.py`) | - | `datasets/dualtalk/{splits,vad,stats.json}` |
| 0 | Mimi codes (`pre_process/dualtalk_mimi_codes.py`, miburi env) | `ls_mimi0/1` | `datasets/dualtalk/mimi_codes` |
| 0 | Whisper small.en word timings (`pre_process/dualtalk_whisper.py`) | `ls_whisper0/1` | `datasets/dualtalk/whisper` |
| 0 | Moshi text tokens (`pre_process/dualtalk_text_tokens.py`) | `ls_pipeline` | `datasets/dualtalk/moshi_text_codes` |
| 1 | Stage-1 VQ-VAE, 53-D (`run_dualtalk_stage1.sh`) | `ls_stage1` | `outputs/DualTalk/dualtalk_vqvae_prior/<id>` |
| 2 | speaking model, own track (`run_dualtalk_speak.sh`) | `ls_pipeline` | `outputs/DualTalk/dualtalk_speak/<id>` |
| 3 | Option A UniLS-style (`run_dualtalk_unils.sh`) | `ls_unils` | `outputs/DualTalk/dualtalk_unils/<id>` |
| 3 | Option B ControlNet (`run_dualtalk_controlnet.sh`) | `ls_controlnet` | `outputs/DualTalk/dualtalk_controlnet/<id>` |

`run_dualtalk_pipeline.sh` (tmux `ls_pipeline`, log `outputs/dualtalk_pipeline.log`)
chains steps 0 → 3: text tokens once Mimi + Whisper finish, the speaking model
once Stage-1 ends, A and B once the speaking model ends. Stage-1 stops by early
stopping (patience 30) or 200 epochs; to stop it earlier, `tmux kill-session -t
ls_stage1` and the pipeline continues from its best checkpoint. The speaking
model runs up to 201 epochs (Exp-5 setting); stop it the same way
(`tmux kill-session -t ls_pipeline` then launch A / B by hand with STAGE1= SPEAK=).

## Data

DualTalk: 4853 train / 533 test / 377 OOD conversations, two separated tracks
each, FLAME (DECA/Spectre) exp 50 + pose 6 at 25 fps, 16 kHz audio, <= 30 s.
- val = 5% of train conversations (243). DualTalk's test shares videos with
  train (282/284), OOD shares none.
- Each track is one item: own = that speaker, partner = the other.
- Motion 53-D = exp 50 + jaw 3 (pose[:, 3:6]); head pose is not modelled.
  Min-max ranges exp [-4.5, 4.5], jaw [-0.2, 0.4] (train 0.1-99.9 percentiles).
- VAD: 40 ms RMS referenced to the louder track of the conversation, own
  track >= ref-25 dB and >= partner+6 dB, 7-frame median. Speech time vs
  transcript word count r = 0.84 (per-track reference: r = 0.17). Speakers talk
  ~32% of frames.
- Mimi input: both tracks DC-removed and divided by their joint peak (MEAD's
  per-clip min-max would amplify a silent listener's bleed to full scale).
- Text tokens: Whisper words that fall outside the track's VAD and are not in
  its DualTalk transcript are dropped as partner bleed (~6%).

## Models

- `model_dualtalk_speak` = Exp-5 recipe with `arflow.use_prosody: false`
  (whole-clip DisVoice prosody is not causal) and `arflow.style_mode: constant`
  (no identity/emotion labels). Both switches are new in this copy of
  `model_vqvae_arflow_pred.py`; defaults keep MEAD behaviour.
- `model_dualtalk_duplex.py`: shared base for A/B. Loads + freezes the speaking
  model, adds a trainable partner audio front-end (copy of the own one), joint
  random crop of own/partner/VAD, cached duplex AR loop, `duplex_leakage_test`.
- Option A `model_dualtalk_unils.py`: per block SelfAttn(LoRA) -> CrossAttn(own)
  -> CrossAttn(partner) -> FFN(LoRA); causal RoPE cross-attention, zero-init
  outputs; AR/flow heads fine-tuned. 9.0 M trainable.
- Option B `model_dualtalk_controlnet.py`: frozen speaking model; trainable
  backbone copy fed main input + Z_in(partner tokens); h_i += g_t Z_i(c_i);
  g_t = 1 - P(own voice) from a classifier on own audio tokens (BCE on VAD).
  Train gate = VAD; inference gate = classifier (`gate=vad` oracle, `gate=ones`
  no switch). 9.0 M trainable.
- Both pass `test_duplex.py`: no future leakage (bit-exact), streaming loop ==
  parallel pass to 1.5e-6 in fp32. (Training uses `set_float32_matmul_precision
  ('medium')` from prepare.py; under it the same comparison is ~1e-3, also for
  the original speaking backbone.)

## Evaluation

    python3.9 evaluate_dualtalk.py folder=outputs/DualTalk/dualtalk_controlnet/<id> split=test   # + gate=vad|ones
    python3.9 dualtalk_metrics.py --pred results/dualtalk/dualtalk_controlnet/<id>/test --split test

`dualtalk_metrics.py` ports DualTalk's metric/metric.py (FD, P-FD, MSE, SID,
STS, diversity, variance, rPCC) for EXP and JAW, plus MSE on speaking /
listening frames and a pooled listening FD. It also scores the DualTalk
baseline's own `result_DualTalk/*.npy` (columns 0:53).

## Not done yet

- DualTalk baseline numbers (pretrained model: HF `ZiqiaoPeng/DualTalk`; its
  env is py3.8 / torch 1.12; run test.py, then dualtalk_metrics.py).
- Vertex-space metrics (LVE / FDD) on DualTalk tracks.
- ARTalk on MEAD / BIWI / VOCASET (talking-head-only table).

# Seamless Interaction (UniLS / FacePlex protocol)

Started 2026-10-14 on the 8x A6000 machine (`cci-csgpu6`), to compare with UniLS
(arXiv 2512.09327) and FacePlex (arXiv 2606.30145, Table 1) on the same data and
metric code.

## Envs (conda)

| env | for |
|---|---|
| `liveface` | training / generation / Whisper / text tokens: clone of `probtalk3d` (py3.9, torch 2.1.1, lightning 2.1.4) + lmdb, sentencepiece, openai-whisper 20250625 |
| `mimi` | Mimi codes (transformers 4.46 `MimiModel`) and the dataset download (hf_xet) |
| `unils` | the released UniLS baseline (py3.13, torch 2.10, transformers 5.3, as in `deps/UniLS/environment.yml`; pytorch3d stubbed) |
| `ARTalk` (existing) | `seamless_metrics.py` (torch + lmdb) |

## Data

UniLS-Talk, Seamless part only (`pre_process/download_unilstalk.sh`, 130 GB LMDB,
SHA-256 checked; the 45 GB TalkMotionOnly part is UniLS's stage-1 pretraining data
and is skipped). Splits = `seamless_interaction_pairs.json`, pairs > 101 frames as in
UniLS: train 71,242 pairs / 620 h of tracks, val 543, test 3457 (the evaluation set).
A pair is one 2-16 s window, tracks A and B; each track is an item, own = it,
partner = the other. Motion 108-D = exp 100 | head rot 3 (zero-centred per clip) |
jaw 1 | gaze 4, min-max scaled per dimension with the train 0.1..99.9 percentiles.
VAD = the dataset's `speech_mask` (the same mask the metrics split speaking /
listening with). Speakers talk ~48% of frames.

| step | script | env | output (`datasets/seamless/`) |
|---|---|---|---|
| motion + VAD, packed | `pre_process/seamless_prepare.py` | liveface | `motion_/vad_/index_<split>`, `norm.json` |
| Mimi codes (joint-peak level, as DualTalk) | `pre_process/seamless_mimi_codes.py` | mimi | `mimi_codes/<split>/` |
| Whisper small.en words | `pre_process/seamless_whisper.py` | liveface | `whisper/<split>/` |
| text tokens + packing | `pre_process/seamless_pack_codes.py` | liveface | `mimi_/text_/codes_<split>` |

`pre_process/run_seamless_audio_features.sh` runs the Mimi + Whisper shards
(tmux `si_mimi*`, `si_whisper*`). Whisper skips tracks without own speech (16.6k of
142k train tracks). Text bleed filter: words with no own-speech frame within
+-200 ms are dropped (30% of val words; 88% of those also appear in the partner's
transcript at the same time, i.e. real bleed).

## Training

Same recipe as DualTalk; `exp_dims=100` puts the 100 expression dims in the exp
loss and the 8 pose dims in the jaw loss (Stage-1 `model.losses.exp_dims`, stage-2
`model.loss.exp_dims`; default 50 keeps MEAD / DualTalk unchanged). Epochs are
capped (`limit_train_batches`: 1000 steps Stage-1, 600 stage-2) so they stay
DualTalk-sized and the epoch-based schedules (scheduled sampling, uncertainty
masking, early stopping) keep their meaning; each epoch draws a fresh random subset.

| step | script | GPUs |
|---|---|---|
| Stage-1 VQ-VAE | `run_seamless_stage1.sh` (tmux `si_stage1`) | 0,1 |
| speaking model | `run_seamless_speak.sh` | 0,1 |
| Option A / B | `run_seamless_unils.sh` / `run_seamless_controlnet.sh` | 0,1 / 4,5 |

`run_seamless_pipeline.sh` (tmux `si_pipeline`, log `outputs/seamless_pipeline.log`)
chains: features -> packing -> wait for Stage-1 -> speaking model -> A + B -> test
generation and metrics. GPUs 2-3 are left to other users.

## Evaluation

    ./run_seamless_eval.sh outputs/Seamless/seamless_controlnet/<id> "0 1 4 5"

= `evaluate_seamless.py` (speaker A of the 3457 test pairs, free-running, raw 108-D
-> `results/seamless/...`) + `seamless_metrics.py`, which imports UniLS's own FLAME
and metric functions (`deps/UniLS` @ 9cad596; FLAME 2020 converted to
`deps/UniLS/assets/flame_2020.pt`): S_LVE S_MHD S_FDD S_PDD S_JDD L_FDD L_PDD L_JDD,
mean over pairs (and std), plus FacePlex's L-FID as we read it (Frechet distance of
all pooled raw 108-D listening frames; FacePlex gives no formula). GT vs GT gives 0.

UniLS baseline: `baselines/unils_generate.py` (released `UniLSLoRAGen.pt`, its
eval.py loop) saves predictions and UniLS's own per-clip numbers, so the port can be
checked against eval.py on identical outputs.

Caveat for the paper table: FacePlex's numbers for UniLS (S-LVE 7.691, S-FDD 23.45)
differ from UniLS's own (5.83, 18.41), so FacePlex's setup differs somewhere and its
code is not released. Compare against baselines re-run here with the same script;
quote FacePlex / Moshi-Face from their paper with a footnote.
