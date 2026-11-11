"""Stage 5: loaders for the two pretrained dimensional speech-emotion (SER) models.

Both were trained by others on MSP-Podcast (natural podcast speech rated by listeners for arousal, dominance and
valence). We do not train or fine-tune anything: the models are used as fixed measuring instruments.

  audeering  audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim   (wav2vec2-large, 12 layers; CC-BY-NC-SA 4.0)
  odyssey    3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes        (WavLM-large; Odyssey 2024 challenge baseline)

predict(model, windows) -> array (n, 3) with columns ARousal, DOMinance, VALence, each roughly in [0, 1].

The odyssey repo ships its model class as custom code (needs trust_remote_code=True). We do NOT run it: the
architecture is re-implemented here from the repo's pipeline_utils.py and only model.safetensors is loaded.
Download once into outputs/audio/cache/hf_odyssey_raw/ (huggingface_hub.snapshot_download; the .bin is not needed).
"""
from __future__ import annotations

from pathlib import Path

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


ODYSSEY_DIR = Path(__file__).resolve().parents[2] / "outputs" / "audio" / "cache" / "hf_odyssey_raw"


class _AttentiveStatsPool(nn.Module):
    """Attentive statistics pooling, re-implemented from the repo's pipeline_utils.py (read, never executed).
    Windows are equal-length and unpadded, so the repo's per-item mask trimming is a no-op and we batch it."""
    def __init__(self, d):
        super().__init__()
        self.sap_linear = nn.Linear(d, d)
        self.attention = nn.Parameter(torch.zeros(d, 1))

    def forward(self, x):                                   # (b, T, d)
        w = torch.softmax(torch.tanh(self.sap_linear(x)) @ self.attention, dim=1)   # (b, T, 1)
        mu = (x * w).sum(1)
        sd = torch.sqrt(((x ** 2) * w).sum(1).sub(mu ** 2).clamp(min=1e-5))
        return torch.cat([mu, sd], dim=1)


class _OdysseyModel(nn.Module):
    """Same layer names as the repo's SERModel, so the safetensors weights load with strict=True.
    The WavLM encoder is transformers' own WavLMModel; no code from the model repo is run."""
    def __init__(self, cfg):
        super().__init__()
        from transformers import WavLMConfig, WavLMModel
        self.ssl_model = WavLMModel(WavLMConfig.from_pretrained(cfg["ssl_type"]))
        d = cfg["hidden_size"]
        self.pool_model = _AttentiveStatsPool(d)
        self.ser_model = nn.Module()
        self.ser_model.fc = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.ReLU(), nn.Dropout(0.5))])
        self.ser_model.out = nn.Sequential(nn.Linear(d, cfg["num_classes"]))

    def forward(self, x):
        h = self.ssl_model(x, attention_mask=torch.ones_like(x, dtype=torch.long)).last_hidden_state
        h = self.pool_model(h)
        for fc in self.ser_model.fc:
            h = fc(h)
        return self.ser_model.out(h)


def _load_odyssey(device):
    import json
    from safetensors.torch import load_file
    cfg = json.loads((ODYSSEY_DIR / "config.json").read_text())
    assert cfg["classifier_hidden_layers"] == 1 and cfg["ssl_type"] == "microsoft/wavlm-large"
    model = _OdysseyModel(cfg)
    sd = load_file(str(ODYSSEY_DIR / "model.safetensors"))
    # old-style weight_norm names -> torch parametrizations (if this transformers version uses them)
    target = model.state_dict().keys()
    for old, new in (("weight_g", "parametrizations.weight.original0"), ("weight_v", "parametrizations.weight.original1")):
        for k in [k for k in sd if k.endswith(old)]:
            nk = k[: -len(old)] + new
            if nk in target and k not in target:
                sd[nk] = sd.pop(k)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), cfg


class Loaded:
    def __init__(self, name, model, device, order, mean=None, std=None):
        self.name, self.model, self.device, self.order, self.mean, self.std = name, model, device, order, mean, std


def load(name: str, device: str = "cuda") -> Loaded:
    if name == "audeering":
        model = _audeering_classes().from_pretrained(MODEL_IDS[name]).to(device).eval()
        return Loaded(name, model, device, order=[0, 1, 2])  # card: arousal, dominance, valence
    if name == "odyssey":
        model, cfg = _load_odyssey(device)
        id2label = {int(k): v.lower() for k, v in cfg["id2label"].items()}
        order = [next(i for i, l in id2label.items() if l == a) for a in ATTRIBUTES]
        return Loaded(name, model, device, order, mean=float(cfg["mean"]), std=float(cfg["std"]))
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
        x = (x - m.mean) / (m.std + 1e-6)   # repo card: global mean/std from config, not per-window
        out = m.model(torch.from_numpy(x).to(m.device))
    out = out.float().cpu().numpy()
    return out[:, m.order]
