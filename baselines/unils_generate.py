"""Released UniLS checkpoint on the Seamless test split, the way its own eval.py runs it.

Mirrors deps/UniLS/eval.py: InferEngine(checkpoint), the checkpoint's dataset config with
our LMDB paths, test split, batch_size 1, seed 42, infer_engine.inference(tau, cfg) and
infer_engine._calc_metrics per clip (nanmean at the end, printed like eval.py). Also saves
every prediction (speaker A, [T, 108]) so seamless_metrics.py can score the same outputs:
both must agree, which checks our port of the protocol.

Differences to eval.py: no accelerate (one process per shard; --shard / --num_shards), and
pytorch3d (imported only for rendering) is replaced by a stub.

    cd deps/UniLS && python ../../baselines/unils_generate.py -r checkpoints/UniLSLoRAGen.pt --shard 0 --num_shards 4
Run in the `unils` env. Outputs: results/seamless/unils_released/<ckpt>/test/<trackA>.npy
                                 + per-clip UniLS metrics in .../test/unils_metrics_shard<i>.json
"""
import argparse
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNILS = ROOT / "deps/UniLS"
DATA = ROOT / "datasets/unilstalk/SeamlessInteractionTalk"


def _stub_pytorch3d():
    class Dummy:   # InferEngine builds a RenderMesh before scoring; nothing is rendered
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return Dummy()

        def __getattr__(self, attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return Dummy()

    def placeholder(attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return Dummy

    def mod(name):
        m = types.ModuleType(name)
        m.__getattr__ = placeholder
        sys.modules[name] = m
        return m
    for n in ["pytorch3d", "pytorch3d.io", "pytorch3d.structures", "pytorch3d.renderer",
              "pytorch3d.renderer.mesh", "pytorch3d.renderer.mesh.shader"]:
        mod(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--resume_path", required=True)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_clips", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(UNILS))
    _stub_pytorch3d()
    import accelerate
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from core.data import build_dataset
    from core.trainer.inferencer import InferEngine

    torch.set_float32_matmul_precision("high")
    accelerate.utils.set_seed(42 + args.shard)
    engine = InferEngine(args.resume_path, device="cuda:0")
    dcfg = OmegaConf.create(OmegaConf.to_container(engine.meta_cfg.DATASET))   # read-only in the ckpt
    dcfg.DATA_PATH, dcfg.META_PATH = str(DATA / "data_lmdb"), str(DATA / "seamless_interaction_pairs.json")
    dataset = build_dataset(data_cfg=dcfg, split="test")
    print(f"dataset config: {dict(dcfg)}\ntest clips: {len(dataset)}", flush=True)

    out_dir = ROOT / "results/seamless/unils_released" / Path(args.resume_path).stem / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = list(range(len(dataset)))[args.shard::args.num_shards][:args.max_clips]
    rows = {}
    for n, i in enumerate(todo):
        batch = torch.utils.data.default_collate([dataset[i]])
        key = batch["infos"]["audio_key"][0][0]
        res = engine.inference(batch, tau=args.tau, cfg=args.cfg)
        m = engine._calc_metrics(res, batch)
        rows[key] = {k: v.item() for k, v in m.items()}
        np.save(out_dir / f"{key}.npy", res["pred_motion_code"][0].float().cpu().numpy())
        if (n + 1) % 100 == 0:
            print(f"[{n + 1}/{len(todo)}]", flush=True)
    json.dump({"resume_path": args.resume_path, "tau": args.tau, "cfg": args.cfg, "clips": rows},
              open(out_dir / f"unils_metrics_shard{args.shard}.json", "w"))
    keys = next(iter(rows.values())).keys()
    print(f"[shard {args.shard}, {len(rows)} clips, tau={args.tau}, cfg={args.cfg}] " +
          ", ".join(f"{k}={np.nanmean([r[k] for r in rows.values()]):.4f}" for k in keys))


if __name__ == "__main__":
    main()
