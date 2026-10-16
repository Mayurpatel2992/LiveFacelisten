"""DualTalk (CVPR 2025) dyadic conversations for listen+speak training.

Every conversation <conv> has two separated tracks, speaker1 / speaker2. Each
track becomes one item whose *own* stream is that speaker and whose *partner*
stream is the other one, so a conversation yields two items:

    keyid  = <conv>_speaker1      (own = speaker1, partner = speaker2)
    keyid  = <conv>_speaker2      (own = speaker2, partner = speaker1)

Motion is 53-D like MEAD (exp 50 + jaw 3, FLAME pose[:, 3:6]); head pose is
not modelled. Values are min-max scaled with fixed ranges chosen from the
DualTalk train percentiles (0.1..99.9: exp -4.66..4.04, jaw -0.19..0.30).

Fields (stage 2, load_audio=True):
    motion                  [1, T, 53]  float32, scaled
    mimi_codes              [K, T_mimi] own track, 12.5 Hz
    moshi_text_codes        [1, T_mimi] own track
    partner_mimi_codes      [K, T_mimi] (load_partner=True)
    partner_moshi_text_codes[1, T_mimi]
    vad / partner_vad       [T] uint8, 25 fps, 1 = that speaker is talking
    audio                   None   (Moshi conditioning only; no waveform)
    prosody                 None   (whole-clip DisVoice prosody is not causal)
"""
import json
import logging
from pathlib import Path

import numpy as np
import torch
from rich.progress import track
from torch.utils.data import Dataset

from .base import BaseDataModule
from .utils import get_split_keyids

logger = logging.getLogger(__name__)


class DualTalkDataModule(BaseDataModule):
    def __init__(self, batch_size: int, num_workers: int, **kwargs):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.save_hyperparameters(logger=False)
        self.split_path = self.hparams.split_path
        self.one_hot_dim = self.hparams.one_hot_dim
        self.Dataset = DualTalk
        sample_overrides = {"split": "train", "tiny": True, "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        self.nfeats = self._sample_set.nfeats


def minmax(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (x - lo) / (hi - lo)


def _load_codes(path: Path):
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return torch.from_numpy(arr.astype(np.int64))


class DualTalk(Dataset):
    def __init__(self, data_name: str,
                 raw_path: str,
                 split_path: str,
                 vad_path: str,
                 load_audio: bool,
                 split: str,
                 tiny: bool,
                 progress_bar: bool,
                 debug: bool,
                 exp_range=(-4.5, 4.5),
                 jaw_range=(-0.2, 0.4),
                 **kwargs):
        super().__init__()
        self.data_name = data_name
        self.load_audio = load_audio
        self.load_mimi_codes = bool(kwargs.get("load_mimi_codes", False))
        self.load_moshi_text_codes = bool(kwargs.get("load_moshi_text_codes", False))
        self.load_partner = bool(kwargs.get("load_partner", False))
        self.exp_range = tuple(exp_range)
        self.jaw_range = tuple(jaw_range)

        raw_path = Path(raw_path)
        vad_path = Path(vad_path)
        with open(Path(split_path) / "source_dirs.json") as f:
            source_dir = raw_path / json.load(f)[split]
        convs = get_split_keyids(path=split_path, split=split)
        if tiny:
            convs = convs[:1]
        elif debug:
            convs = convs[:4]

        it = track(convs, f"Loading {data_name} {split}") if progress_bar else convs
        self.motion_data, self.vad_data = {}, {}
        for conv in it:
            for s in ("speaker1", "speaker2"):
                key = f"{conv}_{s}"
                m = np.load(source_dir / f"{key}.npz")
                exp = minmax(m["exp"][:, :50], *self.exp_range)
                jaw = minmax(m["pose"][:, 3:6], *self.jaw_range)
                motion = np.concatenate([exp, jaw], axis=1).astype(np.float32)
                self.motion_data[key] = torch.from_numpy(motion).unsqueeze(0)   # [1,T,53]
                if load_audio:
                    self.vad_data[key] = torch.from_numpy(np.load(vad_path / f"{key}.npy").astype(np.float32))
        self.keyids = list(self.motion_data.keys())

        if load_audio:
            self.mimi_codes_data = self._load_all(kwargs.get("mimi_codes_path"), self.load_mimi_codes, "Mimi codes")
            self.moshi_text_codes_data = self._load_all(kwargs.get("moshi_text_codes_path"),
                                                        self.load_moshi_text_codes, "Moshi text codes")

        self.nfeats = self.motion_data[self.keyids[0]].shape[2]
        print(f"The number of loaded data pair is: {len(self.keyids)}")
        print(f"Number of features of a motion frame: {self.nfeats}")

    def _load_all(self, path, enabled: bool, what: str):
        if not enabled:
            return None
        if not path:
            raise ValueError(f"{what} requested but no path given")
        path = Path(path)
        out = {k: _load_codes(path / f"{k}.npy") for k in self.keyids if (path / f"{k}.npy").exists()}
        missing = [k for k in self.keyids if k not in out]
        if missing:
            raise FileNotFoundError(f"{len(missing)}/{len(self.keyids)} tracks have no {what} under "
                                    f"{path} (first: {missing[:3]})")
        return out

    @staticmethod
    def partner_of(keyid: str) -> str:
        conv, s = keyid.rsplit("_", 1)
        return f"{conv}_{'speaker2' if s == 'speaker1' else 'speaker1'}"

    def __getitem__(self, index):
        keyid = self.keyids[index]
        element = {"motion": self.motion_data[keyid], "keyid": keyid}
        if not self.load_audio:
            return element
        partner = self.partner_of(keyid)
        element.update({"audio": None, "prosody": None, "vad": self.vad_data[keyid]})
        if self.mimi_codes_data is not None:
            element["mimi_codes"] = self.mimi_codes_data[keyid]
        if self.moshi_text_codes_data is not None:
            element["moshi_text_codes"] = self.moshi_text_codes_data[keyid]
        if self.load_partner:
            element["partner_vad"] = self.vad_data[partner]
            if self.mimi_codes_data is not None:
                element["partner_mimi_codes"] = self.mimi_codes_data[partner]
            if self.moshi_text_codes_data is not None:
                element["partner_moshi_text_codes"] = self.moshi_text_codes_data[partner]
        return element

    def __len__(self):
        return len(self.keyids)

    def __repr__(self):
        return f"{self.data_name} dataset: ({len(self)}, _, ..)"
