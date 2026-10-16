import logging
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from rich.progress import track

from .base import BaseDataModule
from .utils import get_split_keyids
from framework.data.tools.collate import audio_normalize
from framework.data.motion_normalization import normalize_motion

logger = logging.getLogger(__name__)


class MeadDataModule(BaseDataModule):
    def __init__(self, batch_size: int,
                 num_workers: int,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)
        self.save_hyperparameters(logger=False)

        self.split_path = self.hparams.split_path
        self.one_hot_dim = self.hparams.one_hot_dim
        self.Dataset = MEAD
        # Get additional info of the dataset
        sample_overrides = {"split": "train", "tiny": True,
                            "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        self.nfeats = self._sample_set.nfeats


class MEAD(Dataset):
    def __init__(self, data_name: str,
                 motion_path: str,
                 audio_path: str,
                 prosody_path: str,
                 split_path: str,
                 load_audio: str,
                 split: str,
                 tiny: bool,
                 progress_bar: bool,
                 debug: bool,
                 **kwargs):
        super().__init__()
        self.data_name = data_name
        self.load_audio = load_audio

        ids = get_split_keyids(path=split_path, split=split)
        if progress_bar:
            enumerator = enumerate(track(ids, f"Loading {data_name} {split} dataset"))
        else:
            enumerator = enumerate(ids)

        if tiny:
            max_data = 2
        elif not tiny and debug:
            max_data = 8
        else:
            max_data = np.inf

        motion_data_all = {}
        shape_data_all = {}
        audio_data_all = {}
        prosody_data_all = {}

        if load_audio:
            for i, id in enumerator:
                if len(motion_data_all) >= max_data:
                    break
                # load 3DMEAD dataset
                key, motion_data, shape_data, audio_data, prosody_data = load_data(keyid=id,
                                                                     motion_path=Path(motion_path),
                                                                     audio_path=Path(audio_path),
                                                                     prosody_path=Path(prosody_path),
                                                                     max_data=max_data,
                                                                     load_audio=load_audio,
                                                                     split=split)
                motion_data_all.update(dict(zip(key, motion_data)))
                shape_data_all.update(dict(zip(key, shape_data)))
                audio_data_all.update(dict(zip(key, audio_data)))
                prosody_data_all.update(dict(zip(key, prosody_data)))
        else:
            for i, id in enumerator:
                if len(motion_data_all) >= max_data:
                    break
                # load 3DMEAD dataset
                key, motion_data, shape_data = load_data(keyid=id,
                                                         motion_path=Path(motion_path),
                                                         audio_path=Path(audio_path),
                                                         prosody_path=Path(prosody_path),
                                                         max_data=max_data,
                                                         load_audio=load_audio,
                                                         split=None)
                motion_data_all.update(dict(zip(key, motion_data)))
                shape_data_all.update(dict(zip(key, shape_data)))

        self.motion_data = motion_data_all
        self.shape_data = shape_data_all
        if load_audio:
            self.audio_data = audio_data_all
            self.prosody_data = prosody_data_all
        self.keyids = list(motion_data_all.keys())  # file name
        self.nfeats = self[0]["motion"].shape[2]    # number of feature
        print(f"The number of loaded data pair is: {len(self.motion_data)}")
        print(f"Number of features of a motion frame: {self.nfeats}")

    def load_keyid(self, keyid):
        if self.load_audio:
            element = {"motion": self.motion_data[keyid], "shape": self.shape_data[keyid], "audio": self.audio_data[keyid], "prosody": self.prosody_data[keyid],
                       "keyid": keyid}
        else:
            element = {"motion": self.motion_data[keyid], "shape": self.shape_data[keyid], "keyid": keyid}
        return element

    def __getitem__(self, index):
        keyid = self.keyids[index]
        element = self.load_keyid(keyid)
        return element

    def __len__(self):
        return len(self.keyids)

    def __repr__(self):
        return f"{self.data_name} dataset: ({len(self)}, _, ..)"


def load_data(keyid, motion_path, audio_path, prosody_path, max_data, load_audio, split):
    try:
        motion_dir = list(motion_path.glob(f"{keyid}*.npy"))
        motion_key = [directory.stem for directory in motion_dir]
        audio_dir = list(audio_path.glob(f"{keyid}*.wav"))
        audio_key = [directory.stem for directory in audio_dir]
        prosody_dir = list(prosody_path.glob(f"{keyid}*.npy"))
        prosody_key = [directory.stem for directory in prosody_dir]
    except FileNotFoundError:
        return None

    keys = []
    motion_data = []
    shape_data = []
    audio_data = []
    prosody_data = []
    # File naming example: "M003_002_0_0", where M003 is the identity
    identities_list = [
        "W019", "W021", "W023", "W024", "W025", "W026", "W028", "W029",
        "M003", "M005", "M007", "M009", "M011", "M012", "M013", "M019",
        "M022", "M023", "M024", "M025", "M026", "M027", "M028", "M029",
        "M030", "M031", "W009", "W011", "W014", "W015", "W016", "W018",
    ]

    if load_audio:  # second stage
        for key in motion_key:
            if len(keys) >= max_data:
                keys = keys[:max_data]
                motion_data = motion_data[:max_data]
                shape_data = shape_data[:max_data]
                audio_data = audio_data[:max_data]
                prosody_data = prosody_data[:max_data]
                break

            if key in audio_key:
                key_split = key.split("_")
                identity = key_split[0]
                load_key = None
                num_ids = len(identities_list)

                # Split by identity position in identities_list (same as mead_new_split_strategy)
                if split == "train":
                    if identity in identities_list[:int(num_ids * 0.7)]:
                        load_key = key
                        keys.append(key)
                elif split == "val":
                    if identity in identities_list[int(num_ids * 0.7):int(num_ids * 0.85)]:
                        load_key = key
                        keys.append(key)
                elif split == "test":
                    if identity in identities_list[int(num_ids * 0.85):]:
                        load_key = key
                        keys.append(key)

                if load_key is not None:
                    m_index = motion_key.index(load_key)
                    m_dir = motion_dir[m_index]
                    m_npy = np.load(m_dir)

                    # save motion data
                    exp = np.squeeze(m_npy[:, :, 300:350])
                    jaw = np.squeeze(m_npy[:, :, 400:])
                    normalized_exp, normalized_jaw = normalize_motion(exp, jaw)
                    normalized_exp_jaw = np.concatenate((normalized_exp, normalized_jaw), axis=1)
                    motion_data.append(torch.from_numpy(normalized_exp_jaw).unsqueeze(0))
                    # save shape data
                    shape_data.append(torch.from_numpy(m_npy[:, :, :300]))
                    # save audio data
                    a_index = audio_key.index(key)
                    a_dir = audio_dir[a_index]

                    speech_array, _ = librosa.load(a_dir, sr=16000)
                    speech_array = audio_normalize(speech_array)
                    audio_data.append(speech_array)

                    p_index = prosody_key.index(key)
                    p_dir = prosody_dir[p_index]
                    p_npy = np.load(p_dir)
                    if np.isnan(p_npy).any():
                        p_npy = np.nan_to_num(p_npy)
                    p_torch = torch.from_numpy(p_npy).unsqueeze(0)
                    p_torch = p_torch.to(torch.float32)
                    prosody_data.append(p_torch)
                else:
                    # print(f"Pass {key}")
                    pass
            else:
                print(f"No matching audio file for {key}")
                pass

        return keys, motion_data, shape_data, audio_data, prosody_data
    else:     # first stage
        for dir in motion_dir:
            if len(keys) >= max_data:
                keys = keys[:max_data]
                motion_data = motion_data[:max_data]
                shape_data = shape_data[:max_data]
                break

            m_npy = np.load(dir)
            # save file name as key
            keys.append(dir.stem)
            # save motion data
            exp = np.squeeze(m_npy[:, :, 300:350])
            jaw = np.squeeze(m_npy[:, :, 400:])
            normalized_exp, normalized_jaw = normalize_motion(exp, jaw)
            normalized_exp_jaw = np.concatenate((normalized_exp, normalized_jaw), axis=1)
            motion_data.append(torch.from_numpy(normalized_exp_jaw).unsqueeze(0))
            # save shape data
            shape_data.append(torch.from_numpy(m_npy[:, :, :300]))
        return keys, motion_data, shape_data




