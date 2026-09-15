"""
SSL (Wav2Vec2) based spoof-detection model — matches the architecture
trained in the Kaggle notebook, so `ssl_spoof.pt` loads directly into this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class AttentivePooling(nn.Module):
    """Weights each timestep's importance, then computes weighted mean + std."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # x: (B, T, D)
        weights = torch.softmax(self.attn(x), dim=1)          # (B, T, 1)
        mean = torch.sum(x * weights, dim=1)                   # (B, D)
        var = torch.sum(weights * (x - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var + 1e-6)
        return torch.cat([mean, std], dim=1)                   # (B, 2D)


class SSLSpoofDetector(nn.Module):
    def __init__(self, num_classes: int = 2, num_trainable_layers: int = 2):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec2.feature_extractor._freeze_parameters()

        total_layers = len(self.wav2vec2.encoder.layers)
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            trainable = i >= total_layers - num_trainable_layers
            for p in layer.parameters():
                p.requires_grad = trainable

        hidden_size = self.wav2vec2.config.hidden_size  # 768 for wav2vec2-base
        self.pool = AttentivePooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, waveforms):  # waveforms: (B, T)
        hidden_states = self.wav2vec2(waveforms).last_hidden_state  # (B, T', 768)
        pooled = self.pool(hidden_states)
        return self.classifier(pooled)

    def predict_spoof_prob(self, x):
        """Returns probability the input is a spoof/clone (class 1)."""
        with torch.no_grad():
            probs = F.softmax(self.forward(x), dim=1)
            return probs[:, 1]
