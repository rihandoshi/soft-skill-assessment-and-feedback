"""Vox-Profile WavLM-large models (github.com/tiantiaf0627/vox-profile-release), loaded from their weights only.

The repo's WavLMWrapper classes (src/model/fluency/wavlm_fluency.py, src/model/voice_quality/wavlm_voice_quality.py)
were read, never run. Both are: WavLM-large with LoRA (rank 16) on the feed-forward layers of the upper half
(layers 13-23), a softmax-weighted sum of all 25 hidden states, three 1x1 convolutions (1024 -> 256), mean pooling
over time, then one or more small heads. In eval mode a loralib Linear equals a plain Linear with
W + (B @ A) * lora_alpha / r (lora_alpha = 1, loralib's default), so the LoRA updates are folded into transformers' own
WavLMModel. Inputs: 16-kHz mono, normalised per window (the wavlm-large feature extractor has do_normalize = true).
At load time every weight in the file must be used and none may be missing.

  speech_flow     tiantiaf/wavlm-large-speech-flow (rev aea5a480): heads fluency_layer (2, softmax: fluent /
                  disfluent) and dysfluency_layer (5, sigmoid)
  voice_quality   tiantiaf/wavlm-large-voice-quality (rev 5c0df771, percept "complete"): head output_layer
                  (25, sigmoid), the tags in VOICE_QUALITY_TAGS
Both: OpenRAIL licence, no commercial use.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

VOICE_QUALITY_TAGS = ['shrill', 'nasal', 'deep',                                          # pitch
                      'silky', 'husky', 'raspy', 'guttural', 'vocal-fry',                 # texture
                      'booming', 'authoritative', 'loud', 'hushed', 'soft',               # volume
                      'crisp', 'slurred', 'lisp', 'stammering',                           # clarity
                      'singsong', 'pitchy', 'flowing', 'monotone', 'staccato', 'punctuated', 'enunciated',
                      'hesitant']                                                         # rhythm
SPEECH_FLOW_TYPES = ["Block", "Prolongation", "Sound Repetition", "Word Repetition", "Interjection"]

SPECS = {
    "speech_flow": {"dir": P.HF_SPEECH_FLOW, "heads": {"fluency_layer": 2, "dysfluency_layer": 5},
                    "config": {"finetune_method": "lora", "freeze_params": True, "hidden_dim": 256, "lora_rank": 16,
                               "pretrain_model": "wavlm_large", "use_conv_output": True}},
    "voice_quality": {"dir": P.HF_VOICE_QUALITY, "heads": {"output_layer": 25},
                      "config": {"finetune_method": "lora", "freeze_params": True, "hidden_dim": 256,
                                 "lora_rank": 16, "percept": "complete", "pretrain_model": "wavlm_large",
                                 "use_conv_output": True}},
}


def load(name: str, device: str):
    """The model in eval mode; forward(x: (b, samples) float32 normalised) returns a tuple of head logits."""
    import torch
    from safetensors.torch import load_file
    from torch import nn
    from transformers import WavLMConfig, WavLMModel

    spec = SPECS[name]
    cfg = json.loads((spec["dir"] / "config.json").read_text())
    assert cfg == spec["config"], cfg
    r, alpha, hd = cfg["lora_rank"], 1.0, cfg["hidden_dim"]

    class VoxProfileWavLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_model = WavLMModel(WavLMConfig.from_pretrained("microsoft/wavlm-large"))
            n = self.backbone_model.config.num_hidden_layers + 1
            d = self.backbone_model.config.hidden_size
            self.weights = nn.Parameter(torch.ones(n) / n)
            self.model_seq = nn.Sequential(nn.Conv1d(d, hd, 1), nn.ReLU(), nn.Dropout(0.1), nn.Conv1d(hd, hd, 1),
                                           nn.ReLU(), nn.Dropout(0.1), nn.Conv1d(hd, hd, 1))
            self.head_names = list(spec["heads"])
            for h, k in spec["heads"].items():
                setattr(self, h, nn.Sequential(nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, k)))

        def forward(self, x):
            hs = self.backbone_model(x, attention_mask=torch.ones_like(x, dtype=torch.long),
                                     output_hidden_states=True).hidden_states
            w = torch.softmax(self.weights, dim=-1)
            f = (torch.stack(hs, 0) * w[:, None, None, None]).sum(0)          # (b, T, d)
            f = self.model_seq(f.transpose(1, 2)).transpose(1, 2).mean(1)    # (b, hd)
            return tuple(getattr(self, h)(f) for h in self.head_names)

    model = VoxProfileWavLM()
    sd = load_file(str(spec["dir"] / "model.safetensors"))
    for k in [k for k in sd if k.endswith(".lora_A")]:          # fold LoRA (exact in eval mode)
        base = k[: -len(".lora_A")]
        sd[base + ".weight"] = sd[base + ".weight"] + (sd.pop(base + ".lora_B") @ sd.pop(k)) * (alpha / r)
    target = model.state_dict().keys()                          # old weight_norm names -> parametrizations
    for old, new in (("weight_g", "parametrizations.weight.original0"), ("weight_v", "parametrizations.weight.original1")):
        for k in [k for k in sd if k.endswith(old)]:
            nk = k[: -len(old)] + new
            if nk in target and k not in target:
                sd[nk] = sd.pop(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "masked_spec_embed" not in k]  # training-only parameter
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.to(device).eval()


def normalise(chunks: np.ndarray) -> np.ndarray:
    """Per-window zero mean, unit variance (Wav2Vec2FeatureExtractor with do_normalize=True)."""
    chunks = chunks.astype(np.float32)
    return (chunks - chunks.mean(1, keepdims=True)) / np.sqrt(chunks.var(1, keepdims=True) + 1e-7)
