"""Stage 5: loaders for the two pretrained dimensional speech-emotion (SER) models.

Both were trained by others on MSP-Podcast (natural podcast speech rated by listeners for arousal, dominance and
valence). We do not train or fine-tune anything: the models are used as fixed measuring instruments.

  audeering  audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim   (wav2vec2-large, 12 layers; CC-BY-NC-SA 4.0)
  odyssey    3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes        (WavLM-large; Odyssey 2024 challenge baseline)

predict(model, windows) -> array (n, 3) with columns ARousal, DOMinance, VALence, each roughly in [0, 1].
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

ATTRIBUTES = ("arousal", "dominance", "valence")
MODEL_IDS = {"audeering": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
             "odyssey": "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"}


def _audeering_classes():
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):  # from the model card
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            x = torch.tanh(self.dense(self.dropout(x)))
            return self.out_proj(self.dropout(x))

    class EmotionModel(Wav2Vec2PreTrainedModel):  # from the model card
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return hidden, self.classifier(hidden)

    return EmotionModel


class Loaded:
    def __init__(self, name, model, device, order, mean=None, std=None):
        self.name, self.model, self.device, self.order, self.mean, self.std = name, model, device, order, mean, std


def load(name: str, device: str = "cuda") -> Loaded:
    if name == "audeering":
        model = _audeering_classes().from_pretrained(MODEL_IDS[name]).to(device).eval()
        return Loaded(name, model, device, order=[0, 1, 2])  # card: arousal, dominance, valence
    if name == "odyssey":
        from transformers import AutoModelForAudioClassification
        model = AutoModelForAudioClassification.from_pretrained(MODEL_IDS[name], trust_remote_code=True).to(device).eval()
        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        order = [next(i for i, l in id2label.items() if l == a) for a in ATTRIBUTES]
        return Loaded(name, model, device, order, mean=float(model.config.mean), std=float(model.config.std))
    raise ValueError(name)


@torch.no_grad()
def predict(m: Loaded, windows: np.ndarray) -> np.ndarray:
    """windows: (n, samples) float32 at 16 kHz, equal length."""
    x = windows.astype(np.float32)
    if m.name == "audeering":
        # Wav2Vec2 feature extractor: zero-mean, unit-variance per window
        x = (x - x.mean(axis=1, keepdims=True)) / np.sqrt(x.var(axis=1, keepdims=True) + 1e-7)
        out = m.model(torch.from_numpy(x).to(m.device))[1]
    else:
        x = (x - m.mean) / (m.std + 1e-6)
        mask = torch.ones(x.shape, device=m.device)
        out = m.model(torch.from_numpy(x).to(m.device), mask)
        out = out.logits if hasattr(out, "logits") else out
    out = out.float().cpu().numpy()
    return out[:, m.order]
