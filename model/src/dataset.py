"""
PyTorch Dataset implementations for the SSL (waveform-based) pipeline.
"""

import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from preprocess import audio_file_to_waveform

LABEL_MAP = {"bonafide": 0, "spoof": 1}   # 0 = real, 1 = cloned/synthetic


class ASVspoofWaveformDataset(Dataset):
    """
    Expects the standard ASVspoof2019 LA layout:
      root/ASVspoof2019_LA_<split>/flac/<file_id>.flac
      root/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.<split>.trn|trl.txt
    """

    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root)
        self.split = split
        protocol_suffix = "trn" if split == "train" else "trl"
        protocol_path = (
            self.root
            / "ASVspoof2019_LA_cm_protocols"
            / f"ASVspoof2019.LA.cm.{split}.{protocol_suffix}.txt"
        )
        audio_dir = self.root / f"ASVspoof2019_LA_{split}" / "flac"

        self.items: List[Tuple[str, int]] = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                file_id, label_str = parts[1], parts[-1]
                audio_path = audio_dir / f"{file_id}.flac"
                self.items.append((str(audio_path), LABEL_MAP[label_str]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        waveform = audio_file_to_waveform(path)
        return torch.from_numpy(waveform), torch.tensor(label, dtype=torch.long)


class CustomFolderWaveformDataset(Dataset):
    """
    Expects:
      root/real/*.wav|flac
      root/cloned/*.wav|flac
    Use this for testing on your own recorded + cloned voice samples.
    """

    VALID_EXT = {".wav", ".flac", ".mp3", ".ogg"}

    def __init__(self, root: str):
        self.items: List[Tuple[str, int]] = []
        for label_name, label in (("real", 0), ("cloned", 1)):
            folder = Path(root) / label_name
            if not folder.exists():
                continue
            for f in folder.iterdir():
                if f.suffix.lower() in self.VALID_EXT:
                    self.items.append((str(f), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        waveform = audio_file_to_waveform(path)
        return torch.from_numpy(waveform), torch.tensor(label, dtype=torch.long)
