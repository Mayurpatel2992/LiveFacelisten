from typing import List, Dict
from sklearn.preprocessing import minmax_scale
import torch
from torch import Tensor


def pad_motion_list_to_max(motion_list: List[Tensor]):
    """Pad a list of motion tensors [1, T_i, F] to the max T in the batch.

    Returns:
        padded: [B, T_max, F] with zero-padding on the time dimension.
        lengths: List[int] of the original (pre-pad) frame counts per sample.
    """
    lengths = [m.shape[1] for m in motion_list]
    max_len = max(lengths)
    nfeats = motion_list[0].shape[2]
    padded = []
    for m in motion_list:
        t = m.shape[1]
        if t < max_len:
            pad = torch.zeros(1, max_len - t, nfeats, dtype=m.dtype, device=m.device)
            m = torch.cat([m, pad], dim=1)
        padded.append(m)
    return torch.cat(padded, dim=0), lengths  # [B, T_max, F], List[int]


def pad_audio_motion_to_max(
    audio_list: List[Tensor], motion_list: List[Tensor]
) -> tuple:
    """Pad a list of resampled audio [1, T_i, D_a] and motion [1, T_i, F]
    tensors to the max T across the batch.

    Returns:
        audio_padded:  [B, T_max, D_a]
        motion_padded: [B, T_max, F]
        lengths:       List[int] of the original (pre-pad) frame counts per sample.
    """
    lengths = [a.shape[1] for a in audio_list]
    max_len = max(lengths)
    audio_dim = audio_list[0].shape[2]
    motion_dim = motion_list[0].shape[2]

    padded_audio = []
    padded_motion = []
    for a, m in zip(audio_list, motion_list):
        t = a.shape[1]
        if t < max_len:
            a_pad = torch.zeros(1, max_len - t, audio_dim, dtype=a.dtype, device=a.device)
            m_pad = torch.zeros(1, max_len - t, motion_dim, dtype=m.dtype, device=m.device)
            a = torch.cat([a, a_pad], dim=1)
            m = torch.cat([m, m_pad], dim=1)
        padded_audio.append(a)
        padded_motion.append(m)

    return torch.cat(padded_audio, dim=0), torch.cat(padded_motion, dim=0), lengths


# For training motion prediction
def collate_motion_and_audio(lst_elements: List) -> Dict:
    batch = {
        # Keep motion as list; individual tensors are [1, T_i, F].
        # Padding to max length happens inside the model's training step
        # after audio/motion resampling (so lengths are matched first).
        "motion": [x["motion"] for x in lst_elements],
        # Audio stays as a list of raw waveforms (processed per-item by HuBERT).
        "audio": [x["audio"] for x in lst_elements]}

    # add other keys, such as keyid
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch


# For training motion prior
def collate_motion(lst_elements: List) -> Dict:
    # Motion tensors are [1, T_i, F]; pad to max length for batch_size > 1.
    motion_list = [x["motion"] for x in lst_elements]
    padded_motion, lengths = pad_motion_list_to_max(motion_list)
    batch = {"motion": padded_motion, "lengths": lengths}  # [B, T_max, F]

    # add other keys, such as keyid
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]

    return batch


def audio_normalize(data):
    normalized_data = minmax_scale(data, feature_range=(-1, 1))
    return normalized_data


def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
