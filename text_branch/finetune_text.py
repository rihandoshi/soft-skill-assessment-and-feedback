"""Fine-tune a transformer cross-encoder on "question [SEP] answer" to predict all
interview targets jointly (one regression output per target).

Best-epoch selection uses a participant-grouped 15% slice of TRAIN (all epochs run; the
best one is kept), so the validation split stays clean for model comparison against the
Ridge baselines. Several seeds are trained and their predictions averaged.

Options from the design review:
  --loss mse_ccc     MSE + CCC loss (SmoothL1 on z-scores pulls predictions to the centre)
  --length-input     feed the standardised log word count of the full answer into the head,
                     so the question becomes "does the text add anything beyond length?"
  --full-train       after best-epoch selection, retrain every seed on ALL of train for the
                     best epoch of the seed-averaged selection curve (uses the held-out 15%)
  --oof              5-fold out-of-fold train predictions for fusion, made the same way as the
                     val/test predictions (all seeds averaged, same training recipe)
  --tag              suffix for output names, so runs do not overwrite each other

Usage:
  python text_branch/finetune_text.py                     # val metrics, 3 seeds
  python text_branch/finetune_text.py --oof               # also out-of-fold train preds for fusion
  python text_branch/finetune_text.py --eval-test         # once, after model selection
  python text_branch/finetune_text.py --loss mse_ccc --length-input --full-train --oof --tag v2

Writes:
  outputs/text/reports/finetune_val.csv (and _test.csv with --eval-test)
  outputs/text/predictions/finetune_<model>.parquet
  outputs/text/models/finetune_<model>/seed<k>/  (weights + target scaling)
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from text_common import (OUT_DIR, SEED, TARGETS, _WORD, ccc, cluster_bootstrap_ci, load_splits,
                         regression_metrics, write_json)

SELECT_TARGETS = TARGETS   # best-epoch criterion: mean CCC over all predicted targets (not
                           # answer_score alone: the four labels are rated jointly and correlate 0.7-0.85)


def log_len(answers) -> np.ndarray:
    """log(1 + word count) of the full answer, same tokenisation as the linguistic features."""
    return np.array([np.log1p(len(_WORD.findall((a or "").lower()))) for a in answers], np.float32)


def ccc_loss(p, y):
    """1 - CCC per output, averaged. Noisy on small batches, so it is added to MSE with a weight."""
    pm, ym = p.mean(0), y.mean(0)
    cov = ((p - pm) * (y - ym)).mean(0)
    c = 2 * cov / (p.var(0, unbiased=False) + y.var(0, unbiased=False) + (pm - ym) ** 2 + 1e-8)
    return (1 - c).mean()


class CrossEncoderRegressor(nn.Module):
    def __init__(self, name: str, n_out: int, length_input: bool = False):
        super().__init__()
        # transformers>=5 keeps the checkpoint dtype (fp16 for DeBERTa-v3), which makes
        # pure-fp16 training diverge to NaN; train fp32 weights under bf16 autocast instead.
        self.encoder = AutoModel.from_pretrained(name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.length_input = length_input
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(hidden + int(length_input), n_out)

    def forward(self, input_ids, attention_mask, length=None):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = self.drop((h * m).sum(1) / m.sum(1).clamp(min=1))   # mean pooling
        if self.length_input:   # mean pooling hides length, so it is given explicitly
            pooled = torch.cat([pooled, length.to(pooled.dtype).unsqueeze(-1)], -1)
        return self.head(pooled)


def make_loader(tok, q, a, y, bs, shuffle, max_len, length, seed=0):
    enc = tok(list(q), list(a), truncation="only_second", max_length=max_len)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
              "labels": y[i], "length": length[i]} for i in range(len(q))]

    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        ids = torch.tensor([b["input_ids"] + [pad] * (L - len(b["input_ids"])) for b in batch])
        att = torch.tensor([b["attention_mask"] + [0] * (L - len(b["attention_mask"])) for b in batch])
        return {"input_ids": ids, "attention_mask": att,
                "labels": torch.tensor(np.stack([b["labels"] for b in batch]), dtype=torch.float32),
                "length": torch.tensor([b["length"] for b in batch], dtype=torch.float32)}

    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=bs, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(model(b["input_ids"].to(device), b["attention_mask"].to(device),
                             b["length"].to(device)).float().cpu())
    return torch.cat(out).numpy()


def train_one(args, tok, tr_df, es_df, pred_dfs, seed, device, save_dir=None, epochs=None):
    """Train on tr_df and return de-standardised predictions for pred_dfs.
    es_df given: keep the epoch with the best mean CCC on es_df. es_df None: train exactly
    `epochs` epochs and keep the last state."""
    epochs = epochs or args.epochs
    torch.manual_seed(seed)
    np.random.seed(seed)
    Y = tr_df[TARGETS].to_numpy(float)
    lo, hi = np.percentile(Y, 1, axis=0), np.percentile(Y, 99, axis=0)   # winsorise train labels
    Yc = np.clip(Y, lo, hi)
    mu, sd = Yc.mean(0), Yc.std(0)
    z = lambda d: ((np.clip(d[TARGETS].to_numpy(float), lo, hi) - mu) / sd).astype(np.float32)

    L_tr = log_len(tr_df["answer_text"])
    l_mu, l_sd = float(L_tr.mean()), float(L_tr.std())
    zl = lambda d: (log_len(d["answer_text"]) - l_mu) / l_sd
    loader = lambda d, y, bs, shuf, sd_=0: make_loader(tok, d["question"], d["answer_text"], y, bs, shuf,
                                                       args.max_len, zl(d), sd_)
    tr_dl = loader(tr_df, z(tr_df), args.batch_size, True, seed)
    es_dl = loader(es_df, z(es_df), 32, False) if es_df is not None else None
    model = CrossEncoderRegressor(args.model, len(TARGETS), args.length_input).to(device)
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},
        {"params": head_params, "lr": args.lr * 20},
    ], weight_decay=0.01)
    steps = epochs * math.ceil(len(tr_dl) / args.grad_accum)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    if args.loss == "smoothl1":
        loss_fn = nn.SmoothL1Loss(beta=1.0)
    elif args.loss == "mse":
        loss_fn = nn.functional.mse_loss
    else:
        loss_fn = lambda p, y: nn.functional.mse_loss(p, y) + args.ccc_weight * ccc_loss(p, y)
    sel_idx = [TARGETS.index(t) for t in SELECT_TARGETS]
    y_es = es_df[TARGETS].to_numpy(float) if es_df is not None else None

    best, best_state, best_ep, curve = -np.inf, None, -1, []
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for i, b in enumerate(tr_dl):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["length"].to(device))
            loss = loss_fn(out.float(), b["labels"].to(device)) / args.grad_accum
            loss.backward()
            tot += loss.item() * args.grad_accum
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(tr_dl):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        if es_dl is None:   # fixed-epoch run: keep the final state
            print(f"  seed {seed} ep {ep + 1}/{epochs} loss {tot / len(tr_dl):.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if ep + 1 == epochs:
                best, best_ep = np.nan, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue
        p_es = predict(model, es_dl, device) * sd + mu
        score = float(np.mean([ccc(y_es[:, j], p_es[:, j]) for j in sel_idx]))
        print(f"  seed {seed} ep {ep + 1}/{epochs} loss {tot / len(tr_dl):.4f} "
              f"es_ccc {score:.3f} ({time.time() - t0:.0f}s)", flush=True)
        curve.append(score)
        if score > best:
            best, best_ep = score, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, save_dir / "model.pt")
        write_json(save_dir / "scaling.json", {"targets": TARGETS, "mean": mu.tolist(), "std": sd.tolist(),
                   "clip_lo": lo.tolist(), "clip_hi": hi.tolist(), "best_epoch": best_ep,
                   "es_mean_ccc": best, "base_model": args.model, "max_len": args.max_len,
                   "loss": args.loss, "length_input": args.length_input,
                   "log_len_mean": l_mu, "log_len_std": l_sd})
    outs = []
    for d in pred_dfs:
        outs.append(predict(model, loader(d, np.zeros((len(d), len(TARGETS)), np.float32), 32, False),
                            device) * sd + mu)
    del model, opt
    torch.cuda.empty_cache()
    return outs, best_ep, best, curve


def inner_split(df, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    a, b = next(gss.split(df, groups=df["user_no"]))
    return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--loss", default="smoothl1", choices=["smoothl1", "mse", "mse_ccc"])
    ap.add_argument("--ccc-weight", type=float, default=0.5)
    ap.add_argument("--length-input", action="store_true")
    ap.add_argument("--full-train", action="store_true")
    ap.add_argument("--oof", action="store_true", help="5-fold out-of-fold train predictions, all seeds")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. v2")
    ap.add_argument("--eval-test", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.model.split("/")[-1] + (f"_{args.tag}" if args.tag else "")
    print(f"device={device} model={args.model}")

    dfs = load_splits()
    train, val, test = (dfs[s].reset_index(drop=True) for s in ("train", "val", "test"))
    tok = AutoTokenizer.from_pretrained(args.model)
    n_tok = [len(x) for x in tok(list(train["question"]), list(train["answer_text"]))["input_ids"]]
    print(f"train token lengths: median {int(np.median(n_tok))}, max {max(n_tok)}, "
          f"truncated at {args.max_len}: {sum(n > args.max_len for n in n_tok)}")

    tr_in, es_in = inner_split(train, SEED)
    val_preds, test_preds, info = [], [], []
    for s in args.seeds:
        sdir = OUT_DIR / "models" / f"finetune_{tag}" / f"seed{s}"
        (pv, pt), ep, sc, curve = train_one(args, tok, tr_in, es_in, [val, test], s, device,
                                     None if args.full_train else sdir)
        info.append({"seed": s, "best_epoch": ep, "es_mean_ccc": sc, "es_curve": curve})
        m = regression_metrics(val["answer_score"], pv[:, 0])
        print(f"seed {s}: best epoch {ep}, val answer_score spearman {m['spearman']:.3f} ccc {m['ccc']:.3f}")
        if not args.full_train:
            np.savez(sdir / "preds.npz", val=pv, test=pt)   # keep finished seeds if a later one fails
            val_preds.append(pv); test_preds.append(pt)

    # --full-train: retrain each seed on all of train for that epoch count (no selection).
    # The epoch is taken from the seed-averaged selection curve: the selection slice is only
    # about 35 participants, so one seed's best epoch is noisy.
    mean_curve = np.mean([i["es_curve"] for i in info], 0)
    final_epochs = int(np.argmax(mean_curve)) + 1 if args.full_train else None
    print("seed-averaged es_ccc by epoch: " + " ".join(f"{c:.3f}" for c in mean_curve), flush=True)
    if args.full_train:
        print(f"full-train: {final_epochs} epochs on all {len(train)} train responses")
        for s in args.seeds:
            sdir = OUT_DIR / "models" / f"finetune_{tag}" / f"seed{s}"
            (pv, pt), _, _, _ = train_one(args, tok, train, None, [val, test], s, device, sdir, final_epochs)
            np.savez(sdir / "preds.npz", val=pv, test=pt)
            val_preds.append(pv); test_preds.append(pt)
    pv, pt = np.mean(val_preds, 0), np.mean(test_preds, 0)

    # Out-of-fold train predictions for fusion, made like the val/test ones (all seeds averaged,
    # same recipe), so a fusion layer fitted on them sees the same prediction distribution.
    p_train = np.full((len(train), len(TARGETS)), np.nan)
    if args.oof:
        for k, (a, b) in enumerate(GroupKFold(5, shuffle=True, random_state=SEED)
                                   .split(train, groups=train["user_no"])):
            fold = train.iloc[a].reset_index(drop=True)
            if args.full_train:
                f_tr, f_es = fold, None
            else:
                f_tr, f_es = inner_split(fold, SEED + k)
            pbs = [train_one(args, tok, f_tr, f_es, [train.iloc[b]], s, device, epochs=final_epochs)[0][0]
                   for s in args.seeds]
            p_train[b] = np.mean(pbs, 0)
            print(f"oof fold {k + 1}/5 done", flush=True)

    pdir = OUT_DIR / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    parts = []
    for d, p, s in ((train, p_train, "train"), (val, pv, "val"), (test, pt, "test")):
        out = d[["id", "user_no", "question_id"]].copy()
        out["split"] = s
        out[[f"pred_{t}" for t in TARGETS]] = p
        parts.append(out)
    pd.concat(parts).to_parquet(pdir / f"finetune_{tag}.parquet", index=False)

    rows = []
    for split, d, p in [("val", val, pv)] + ([("test", test, pt)] if args.eval_test else []):
        for j, t in enumerate(TARGETS):
            y = d[t].to_numpy(float)
            r = {"model": f"finetune_{tag}", "target": t, "split": split, **regression_metrics(y, p[:, j])}
            r["spearman_ci95"] = cluster_bootstrap_ci(y, p[:, j], d["user_no"], "spearman")
            r["ccc_ci95"] = cluster_bootstrap_ci(y, p[:, j], d["user_no"], "ccc")
            rows.append(r)
    res = pd.DataFrame(rows)
    rdir = OUT_DIR / "reports"
    for split in res["split"].unique():
        res[res["split"] == split].to_csv(rdir / f"finetune{'_' + args.tag if args.tag else ''}_{split}.csv",
                                          index=False)
    write_json(rdir / f"finetune_config{'_' + args.tag if args.tag else ''}.json",
               {**vars(args), "seeds_info": info, "final_epochs": final_epochs,
                "epoch_selection": "participant-grouped 15% of train; criterion mean CCC over "
                                   + ", ".join(SELECT_TARGETS)})
    print(res[["split", "target", "spearman", "ccc", "r2", "rmse", "mae"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
