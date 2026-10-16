"""Causality check for the listen+speak models (Option A / B).

Builds the model exactly like train_all.py and runs ``duplex_leakage_test``:
outputs at frame <= t0 must be bit-exact when own audio, partner audio or
motion after t0 are perturbed, and the cached streaming loop must match the
parallel training pass.

    python3.9 test_duplex.py experiment=test_duplex state=new \
        data=dualtalk_pred_moshi_text data.load_partner=true data.tiny=true \
        model=model_dualtalk_controlnet model.duplex.speak_ckpt_path=... \
        model.folder_prior=... model.version_prior=0 model.ckpt_path_prior=...
"""
import sys

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import framework.launch.prepare  # noqa


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    data_module = instantiate(cfg.data)
    model = instantiate(cfg.model, nfeats=data_module.nfeats, split_path=data_module.split_path,
                        one_hot_dim=data_module.one_hot_dim, resumed_training=False,
                        _recursive_=False)
    # prepare.py sets 'medium' (bf16/TF32 matmuls) for training; that alone moves
    # cached-vs-parallel outputs by ~1e-3, so the exactness check runs in full fp32.
    torch.set_float32_matmul_precision("highest")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    for name, m in model.named_modules():                 # randomise zero-init paths so
        if name.endswith(("proj", "lora_b", "ctrl_in")) or name.startswith("ctrl_out."):
            for p in m.parameters(recurse=False):        # the test exercises them
                torch.nn.init.normal_(p, std=0.02)
    ok = model.duplex_leakage_test(device=device)
    print("DUPLEX CAUSALITY", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
