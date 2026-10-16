"""Seamless Interaction (UniLS-Talk, UniLS / FacePlex protocol) for listen+speak training.

Built by pre_process/seamless_prepare.py (+ seamless_mimi_codes.py, seamless_whisper.py,
seamless_pack_codes.py). A pair is one 2-16 s window of a dyadic conversation with two
tracks A / B; each track is one item (own = that track, partner = the other), like
DualTalk's speaker1 / speaker2. keyid = track name, e.g. V00_S1320_I00000211_P0252_0080_0096.

Motion is 108-D: exp 100 | head rot 3 (zero-centred per clip) | jaw 1 | gaze 4, min-max
scaled per dimension with the train 0.1..99.9 percentiles (datasets/seamless/norm.json).

Fields (same as framework/data/dualtalk.py):
    motion                  [1, T, 108] float32, scaled
    mimi_codes              [8, T_mimi] own track, 12.5 Hz
    moshi_text_codes        [1, T_mimi] own track
    partner_mimi_codes      [8, T_mimi] (load_partner=True)
    partner_moshi_text_codes[1, T_mimi]
    vad / partner_vad       [T] float32, 25 fps, the dataset's speech_mask
    audio, prosody          None
Arrays are memory-mapped, so DDP ranks share the page cache.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BaseDataModule


class SeamlessDataModule(BaseDataModule):
    def __init__(self, batch_size: int, num_workers: int, **kwargs):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.save_hyperparameters(logger=False)
        self.split_path = self.hparams.root
        self.one_hot_dim = self.hparams.one_hot_dim
        self.Dataset = Seamless
        sample_overrides = {"split": "val", "tiny": True, "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        self.nfeats = self._sample_set.nfeats


class Seamless(Dataset):
    def __init__(self, data_name: str, root: str, load_audio: bool, split: str,
                 tiny: bool, progress_bar: bool, debug: bool, **kwargs):
        super().__init__()
        self.data_name = data_name
        self.split = split
        self.load_audio = load_audio
        self.load_partner = bool(kwargs.get("load_partner", False))
        root = Path(root)

        self.index = json.load(open(root / f"index_{split}.json"))
        keys = sorted(self.index)
        if tiny or debug:
            pairs = sorted(k for k in keys if self.index[k][3] == 1)[:1 if tiny else 8]
            keys = sorted(pairs + [self.index[k][2] for k in pairs])
        self.keyids = keys
        self.motion = np.load(root / f"motion_{split}.npy", mmap_mode="r")
        self.vad = np.load(root / f"vad_{split}.npy", mmap_mode="r")
        norm = json.load(open(root / "norm.json"))
        self.lo = np.asarray(norm["lo"], dtype=np.float32)
        self.scale = np.maximum(np.asarray(norm["hi"], dtype=np.float32) - self.lo, 1e-6)

        if load_audio:
            self.codes = json.load(open(root / f"codes_{split}.json"))
            self.mimi = np.load(root / f"mimi_{split}.npy", mmap_mode="r")
            self.text = (np.load(root / f"text_{split}.npy", mmap_mode="r")
                         if kwargs.get("load_moshi_text_codes", False) else None)
        self.nfeats = self.motion.shape[1]
        print(f"The number of loaded data pair is: {len(self.keyids)}")
        print(f"Number of features of a motion frame: {self.nfeats}")

    def _vad(self, key):
        off, T = self.index[key][:2]
        return torch.from_numpy(self.vad[off:off + T].astype(np.float32))

    def _mimi(self, key):
        off, T = self.codes[key]
        return torch.from_numpy(self.mimi[:, off:off + T].astype(np.int64))

    def _text(self, key):
        off, T = self.codes[key]
        return torch.from_numpy(self.text[off:off + T].astype(np.int64))[None]

    def __getitem__(self, index):
        key = self.keyids[index]
        off, T, partner, _ = self.index[key]
        motion = (self.motion[off:off + T] - self.lo) / self.scale
        element = {"motion": torch.from_numpy(motion.astype(np.float32))[None], "keyid": key}
        if not self.load_audio:
            return element
        element.update({"audio": None, "prosody": None, "vad": self._vad(key),
                        "mimi_codes": self._mimi(key)})
        if self.text is not None:
            element["moshi_text_codes"] = self._text(key)
        if self.load_partner:
            element["partner_vad"] = self._vad(partner)
            element["partner_mimi_codes"] = self._mimi(partner)
            if self.text is not None:
                element["partner_moshi_text_codes"] = self._text(partner)
        return element

    def __len__(self):
        return len(self.keyids)

    def __repr__(self):
        return f"{self.data_name} dataset: ({len(self)}, _, ..)"
