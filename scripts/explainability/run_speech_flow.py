"""Phase 1, fluency second opinion: Vox-Profile speech-flow classifier per 3-second window.

Model: tiantiaf/wavlm-large-speech-flow (Vox-Profile, arXiv 2505.14648; OpenRAIL, no commercial use), revision
aea5a480. WavLM-large fine-tuned with LoRA (rank 16) on the feed-forward layers of the upper half (layers 13-23),
a softmax-weighted sum of all 25 hidden states, three 1x1 convolutions (1024 -> 256), mean pooling, and two heads:
  fluency      softmax over (fluent, disfluent)
  disfluency   sigmoid over (Block, Prolongation, Sound Repetition, Word Repetition, Interjection), multi-label

Loading: the repo's WavLMWrapper (src/model/fluency/wavlm_fluency.py in github.com/tiantiaf0627/vox-profile-release)
was read, not run. In eval mode a loralib Linear equals a plain Linear with W + (B @ A) * lora_alpha / r
(lora_alpha = 1, loralib's default), so the LoRA updates are folded into transformers' own WavLMModel and no code
from the model repo is executed. Inputs are normalised per window (the wavlm-large feature extractor has
do_normalize = true). The check at load time: every weight in the file is used, nothing is missing.

Windows: the same as the disfluency classifier (run_disfluency_classifier.py): 3 s, 1.5-s hop, kept when >= 50% VAD
speech, so the two models are compared on identical audio. The model card uses a 1-s hop; the window length (3 s)
is what matters for the classifier.

Per answer (speech_flow_answers.csv):
  n_windows           speech windows scored
  sf_p_disfluent      mean "disfluent" probability over the windows
  sf_share_disfluent  share of windows with p(disfluent) > 0.5
  sf_<type>           mean sigmoid probability of each disfluency type

Output  stage8_detectors/tables/speech_flow_windows.csv, speech_flow_answers.csv (RecruitView + MIT)
Cache   outputs/audio/cache/stage8_detectors/speech_flow/{id}.npz
Run:    python scripts/explainability/run_speech_flow.py --limit 10      (smoke test: memory + speed, no tables)
        python scripts/explainability/run_speech_flow.py                 (all answers; resumes from the cache)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import run_disfluency_classifier as D  # noqa: E402  (clip list, VAD regions, windows)
from run_ser_windows import cap_gpu_memory, report_gpu_peak  # noqa: E402

SR = D.SR
MODEL_DIR = P.ROOT / "outputs" / "audio" / "cache" / "hf_speech_flow"
CACHE = P.CACHE_S8 / "speech_flow"
TYPES = ["Block", "Prolongation", "Sound Repetition", "Word Repetition", "Interjection"]
OUT_WINDOWS = P.S8_TABLES / "speech_flow_windows.csv"
OUT_ANSWERS = P.S8_TABLES / "speech_flow_answers.csv"


def load_model(device: str):
    import torch
    from safetensors.torch import load_file
    from torch import nn
    from transformers import WavLMConfig, WavLMModel

    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    assert cfg == {"finetune_method": "lora", "freeze_params": True, "hidden_dim": 256, "lora_rank": 16,
                   "pretrain_model": "wavlm_large", "use_conv_output": True}, cfg
    r, alpha, hd = cfg["lora_rank"], 1.0, cfg["hidden_dim"]

    class SpeechFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_model = WavLMModel(WavLMConfig.from_pretrained("microsoft/wavlm-large"))
            n = self.backbone_model.config.num_hidden_layers + 1
            d = self.backbone_model.config.hidden_size
            self.weights = nn.Parameter(torch.ones(n) / n)
            self.model_seq = nn.Sequential(nn.Conv1d(d, hd, 1), nn.ReLU(), nn.Dropout(0.1), nn.Conv1d(hd, hd, 1),
                                           nn.ReLU(), nn.Dropout(0.1), nn.Conv1d(hd, hd, 1))
            self.fluency_layer = nn.Sequential(nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, 2))
            self.dysfluency_layer = nn.Sequential(nn.Linear(hd, hd), nn.ReLU(), nn.Linear(hd, 5))

        def forward(self, x):
            hs = self.backbone_model(x, attention_mask=torch.ones_like(x, dtype=torch.long),
                                     output_hidden_states=True).hidden_states
            w = torch.softmax(self.weights, dim=-1)
            f = (torch.stack(hs, 0) * w[:, None, None, None]).sum(0)          # (b, T, d)
            f = self.model_seq(f.transpose(1, 2)).transpose(1, 2).mean(1)    # (b, hd)
            return self.fluency_layer(f), self.dysfluency_layer(f)

    model = SpeechFlow()
    sd = load_file(str(MODEL_DIR / "model.safetensors"))
    # fold LoRA into the plain feed-forward weights (exact in eval mode)
    for k in [k for k in sd if k.endswith(".lora_A")]:
        base = k[: -len(".lora_A")]
        sd[base + ".weight"] = sd[base + ".weight"] + (sd.pop(base + ".lora_B") @ sd.pop(k)) * (alpha / r)
    # old-style weight_norm names -> torch parametrizations (the positional conv embedding)
    target = model.state_dict().keys()
    for old, new in (("weight_g", "parametrizations.weight.original0"), ("weight_v", "parametrizations.weight.original1")):
        for k in [k for k in sd if k.endswith(old)]:
            nk = k[: -len(old)] + new
            if nk in target and k not in target:
                sd[nk] = sd.pop(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "masked_spec_embed" not in k]  # training-only parameter
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.to(device).eval()


def main() -> None:
    import soundfile as sf
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-gpu-frac", type=float, default=0.8)
    args = ap.parse_args()
    cap_gpu_memory(args.max_gpu_frac)
    clips = D.all_clips()
    if args.limit:
        clips = pd.concat([g.sample(max(1, args.limit // 2), random_state=0) for _, g in clips.groupby("dataset")])
    regions = D.speech_regions(clips)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None

    t0, done, rows = time.time(), 0, []
    for k, c in enumerate(clips.itertuples(), 1):
        cache = CACHE / f"{c.id}.npz"
        if cache.exists() and not args.limit:
            z = np.load(cache)
            spans, probs = z["spans"], z["probs"]
        else:
            if model is None:
                model = load_model(device)
            x, sr = sf.read(c.audio_path, dtype="float32")
            x = x.mean(axis=1) if x.ndim > 1 else x
            assert sr == SR, f"{c.id}: {sr} Hz"
            w = D.windows(len(x), regions.get(c.id, np.empty((0, 2))))
            spans = np.asarray(w, float).reshape(-1, 3)
            n = int(D.WIN_S * SR)
            chunks = np.stack([np.pad(x[int(s * SR):int(e * SR)], (0, max(0, n - (int(e * SR) - int(s * SR)))))[:n]
                               for s, e, _ in w]) if w else np.empty((0, n), np.float32)
            chunks = (chunks - chunks.mean(1, keepdims=True)) / np.sqrt(chunks.var(1, keepdims=True) + 1e-7)
            out = []
            for b in range(0, len(chunks), args.batch):
                with torch.no_grad():
                    fl, ty = model(torch.from_numpy(chunks[b:b + args.batch]).to(device))
                out.append(np.concatenate([torch.softmax(fl, -1).cpu().numpy(), torch.sigmoid(ty).cpu().numpy()], 1))
            probs = np.concatenate(out).astype(np.float32) if out else np.empty((0, 7), np.float32)
            if not args.limit:
                np.savez(cache, spans=spans, probs=probs)
            done += 1
        for (s, e, sp), pr in zip(spans, probs):
            rows.append({"id": c.id, "dataset": c.dataset, "start": s, "end": e, "speech_frac": sp,
                         "p_fluent": float(pr[0]), "p_disfluent": float(pr[1]),
                         **{f"p_{t.replace(' ', '')}": float(v) for t, v in zip(TYPES, pr[2:])}})
        if k % 100 == 0 or k == len(clips) or args.limit:
            print(f"  {k}/{len(clips)} clips ({done} computed, {(time.time() - t0) / 60:.1f} min)", flush=True)

    report_gpu_peak()
    win = pd.DataFrame(rows)
    tcols = [f"p_{t.replace(' ', '')}" for t in TYPES]
    g = win.groupby("id")
    ans = pd.DataFrame({"n_windows": g.size(), "sf_p_disfluent": g["p_disfluent"].mean(),
                        "sf_share_disfluent": g["p_disfluent"].apply(lambda v: float((v > 0.5).mean())),
                        **{f"sf_{c[2:]}": g[c].mean() for c in tcols}}).reset_index()
    ans = clips[["id", "dataset"]].merge(ans, on="id", how="left")
    ans["n_windows"] = ans["n_windows"].fillna(0).astype(int)
    if args.limit:
        print(ans.describe().T.round(3).to_string())
        return
    win.to_csv(OUT_WINDOWS, index=False)
    ans.to_csv(OUT_ANSWERS, index=False)
    print(f"wrote {OUT_ANSWERS}: {len(ans)} answers, {len(win)} windows")
    print(ans.groupby("dataset")[["sf_p_disfluent", "sf_share_disfluent"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
