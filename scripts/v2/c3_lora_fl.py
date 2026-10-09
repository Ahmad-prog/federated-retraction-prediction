"""C3: federated LoRA fine-tuning of text encoders/decoders on leakage-clean text (R2.5–R2.9).

Regimes (one per invocation, so the GPU queue can interleave and resume):
  central       pooled training data (reference)
  local         one LoRA model per silo
  fedavg        FedAvg over LoRA + head parameters; then per-silo fine-tuning (personalised)
  fedper        FedAvg over LoRA parameters only; every silo keeps its own head (FedPer)
  ditto         FedAvg global model + per-silo personal models pulled towards it (Ditto)
  fedavg_dp     client-level DP-FedAvg (clipped, size-weighted, Gaussian noise)
  dpsgd_central record-level DP-SGD on pooled data (per-example clipping, Poisson sampling)
  dpsgd_fedavg  FedAvg where every silo trains with record-level DP-SGD (per-silo epsilon)
Every regime: 70/10/20 split stratified by silo x label (same split as the tabular
track for a given seed), model selection on (federated) validation loss, test scores
saved for the main test set and the prospective set (retracted after 2026-07-19).
Global models also score a fixed sample of training articles (membership-inference audit).

Sources:
  fulltext      title + abstract + sections, each truncated to its own token budget
  abstracts     121K corpus, title + abstract, B2 split; prospective = new retractions
                vs test controls of the same publishers
  ft_abstracts  full-text corpus, title + abstract only (same articles as fulltext)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.evaluate import evaluate  # noqa: E402
from src.v2 import DATA2, RES2  # noqa: E402

BUDGET = {"title": 48, "abstract": 336, "introduction": 256, "methods": 512,
          "results": 512, "discussion": 288, "conclusion": 96}  # sums to 2048
SECTIONS = ["introduction", "methods", "results", "discussion", "conclusion"]
MIA_N = 5000


# ------------------------------------------------------------------ data
def load_abstracts(seed: int):
    """121K leakage-clean corpus (title + abstract) with EXACTLY the B2 split and the
    10 publisher silos, so text models line up with the tabular benchmark. Prospective
    positives are appended as part 'prospective' (their controls come from the test set)."""
    from scripts.v2.b2_main_benchmark import partition, split
    df = pd.read_parquet(DATA2 / "tabular_v2.parquet").reset_index(drop=True)
    df["silo"] = partition(df, "publisher")
    df["part"] = split(df, seed)
    pf = DATA2 / "prospective_abstracts.parquet"
    if pf.exists():
        p = pd.read_parquet(pf).assign(silo="prospective", part="prospective")
        p = p[~p["doi"].isin(set(df["doi"]))]
        df = pd.concat([df, p[[c for c in p.columns if c in df.columns or c in ("silo", "part")]]],
                       ignore_index=True)
    df["pmcid"] = df["doi"]  # row key used by the tokenizer cache
    for c in SECTIONS + ["body"]:
        df[c] = ""
    return df


def load_dataset(k: int, seed: int):
    df = pd.read_parquet(DATA2 / "ft_dataset.parquet")
    df["publisher"] = df["publisher"].fillna("Unknown")
    top = df.loc[df.split == "main", "publisher"].value_counts().index[: k - 1]
    df["silo"] = df["publisher"].where(df["publisher"].isin(top), "Other")
    rng = np.random.default_rng(seed)
    df["part"] = "prospective"
    main = df[df.split == "main"]
    for _, idx in main.groupby(["silo", "retracted"]).groups.items():
        idx = rng.permutation(np.array(idx))
        n_te, n_va = int(round(0.2 * len(idx))), int(round(0.1 * len(idx)))
        df.loc[idx[:n_te], "part"] = "test"
        df.loc[idx[n_te:n_te + n_va], "part"] = "val"
        df.loc[idx[n_te + n_va:], "part"] = "train"
    return df.reset_index(drop=True)


def load_ft_abstracts(k: int, seed: int):
    """Full-text corpus and split, but only title + abstract (isolates the value of full text)."""
    df = load_dataset(k, seed)
    for c in SECTIONS + ["body"]:
        df[c] = ""
    return df


def tok_family(model: str) -> str:
    return "modernbert" if "modernbert" in model.lower() else model.split("/")[-1].lower()


def tokenize(df: pd.DataFrame, tok, max_len: int, source: str, family: str) -> list[list[int]]:
    cache = DATA2 / "tok" / f"{family}_{source}_{max_len}.json"
    store = json.loads(cache.read_text()) if cache.exists() else {}
    if all(p in store for p in df["pmcid"]):
        return [store[p] for p in df["pmcid"]]
    scale = max_len / 2048
    if source in ("abstracts", "ft_abstracts"):
        budget = {"title": 64, "abstract": max_len - 66}
        scale = 1.0
    else:
        budget = BUDGET
    out = []
    cls =[tok.cls_token_id] if tok.cls_token_id is not None else []
    sep = [tok.sep_token_id] if tok.sep_token_id is not None else \
        tok("\n\n", add_special_tokens=False)["input_ids"]
    for r in df.to_dict("records"):
        if r["pmcid"] in store:
            out.append(store[r["pmcid"]])
            continue
        ids = list(cls)
        for sec, b in budget.items():
            txt = r.get(sec) or ""
            if not txt:
                continue
            txt = txt[: int(b * scale) * 8]  # ~8 chars/token cap before tokenizing
            ids += tok(f"{sec}: {txt}", add_special_tokens=False, truncation=True,
                       max_length=int(b * scale))["input_ids"] + sep
        if len(ids) < 64 and r.get("body"):  # no recognised sections: use the body
            ids = cls + tok(r["body"][: max_len * 8], add_special_tokens=False, truncation=True,
                            max_length=max_len - 2)["input_ids"] + sep
        if not ids:
            ids = sep
        ids = ids[:max_len]
        out.append(ids)
        store[r["pmcid"]] = ids
    cache.parent.mkdir(parents=True, exist_ok=True)  # merged, atomic (two GPU queues share it)
    tmp = cache.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
    tmp.write_text(json.dumps(store))
    tmp.replace(cache)
    return out


def batches(idx: np.ndarray, lens: np.ndarray, bs: int, rng, shuffle=True):
    """Length-bucketed batches: shuffle, then sort within chunks of 50 batches."""
    idx = rng.permutation(idx) if shuffle else np.array(idx)
    chunk = bs * 50
    out = []
    for i in range(0, len(idx), chunk):
        c = idx[i:i + chunk]
        c = c[np.argsort(-lens[c])]
        out += [c[j:j + bs] for j in range(0, len(c), bs)]
    if shuffle:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def collate(ids_list, pad_id, device):
    L = max(len(x) for x in ids_list)
    inp = torch.full((len(ids_list), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        inp[i, :len(x)] = torch.tensor(x)
        att[i, :len(x)] = 1
    return inp.to(device), att.to(device)


# ------------------------------------------------------------------ model
def build_model(name: str, r: int, pad_id: int):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=1, dtype=torch.bfloat16, attn_implementation="sdpa")
    if "modernbert" in name.lower():
        targets, heads = ["Wqkv", "Wo", "Wi"], ["head", "classifier"]
    else:  # decoder LLM (Qwen3): last-token classification head "score"
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        heads = ["score"]
        m.config.pad_token_id = pad_id
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05,
                     target_modules=targets, modules_to_save=heads)
    m = get_peft_model(m, cfg)
    for n, p in m.named_parameters():  # trainable params in fp32 for stable updates
        if p.requires_grad:
            p.data = p.data.float()
    return m


def trainable(m) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}


def load_trainable(m, state):
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n in state:
                p.copy_(state[n])


def is_head(name: str) -> bool:
    return "modules_to_save" in name


class Runner:
    def __init__(self, a, df, ids):
        self.a, self.df, self.ids = a, df, ids
        self.lens = np.array([len(x) for x in ids])
        self.y = df["retracted"].to_numpy(np.float32)
        self.dev = "cuda"
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(a.model)
        self.pad = t.pad_token_id if t.pad_token_id is not None else t.eos_token_id
        self.model = build_model(a.model, a.lora_r, self.pad).to(self.dev)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self.init_state = trainable(self.model)
        self.tokens_seen = 0

    def _logits(self, idx):
        inp, att = collate([self.ids[i] for i in idx], self.pad, self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = self.model(input_ids=inp, attention_mask=att).logits.squeeze(-1).float()
        self.tokens_seen += int(att.sum())
        return lo

    def run_epochs(self, idx, epochs, lr, seed, pos_weight, prox=None):
        """AdamW, linear warm-up/decay. prox = (lambda, reference state) adds the Ditto
        proximal term lambda/2 * ||theta - reference||^2."""
        m, ga = self.model, self.a.grad_accum
        m.train()
        opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.01)
        rng = np.random.default_rng(seed)
        bl = [b for _ in range(epochs) for b in batches(idx, self.lens, self.a.bs, rng)]
        total = max(1, math.ceil(len(bl) / ga))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * total))) *
            max(0.0, (total - s) / total))
        lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=self.dev))
        if prox is not None:
            lam, ref = prox
            ref = [ref[n].to(self.dev) for n in self.names]
        for i, b in enumerate(bl):
            loss = lf(self._logits(b), torch.tensor(self.y[b], device=self.dev))
            if prox is not None:
                loss = loss + lam / 2 * sum(((p - r) ** 2).sum() for p, r in zip(self.params, ref))
            (loss / ga).backward()
            if (i + 1) % ga == 0 or i == len(bl) - 1:
                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
        return len(bl)

    def run_dpsgd(self, idx, epochs, lr, seed, pos_weight, sigma, clip, lot):
        """Record-level DP-SGD: Poisson sampling with rate q = lot / n, per-example
        gradients clipped to `clip`, Gaussian noise sigma * clip on the sum, divided by the
        expected lot size. Per-example gradients come from micro-batches of `micro` rows
        whose rows are processed one at a time (exact per-example clipping)."""
        m = self.model
        m.train()
        opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0)
        rng = np.random.default_rng(seed)
        gen = torch.Generator(device=self.dev).manual_seed(seed)
        n = len(idx)
        q = lot / n
        steps = int(round(epochs / q))
        lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=self.dev))
        for _ in range(steps):
            b = idx[rng.random(n) < q]
            acc = [torch.zeros_like(p) for p in self.params]
            for i in b:
                loss = lf(self._logits([i]), torch.tensor(self.y[[i]], device=self.dev))
                g = torch.autograd.grad(loss, self.params, allow_unused=True)
                g = [torch.zeros_like(p) if x is None else x for x, p in zip(g, self.params)]
                norm = torch.sqrt(sum((x.float() ** 2).sum() for x in g))
                f = torch.clamp(clip / (norm + 1e-12), max=1.0)
                for s, x in zip(acc, g):
                    s.add_(x * f)
            for p, s in zip(self.params, acc):
                noise = torch.randn(s.shape, generator=gen, device=self.dev) * sigma * clip
                p.grad = (s + noise) / lot
            opt.step()
            opt.zero_grad(set_to_none=True)
        return steps

    @torch.no_grad()
    def predict(self, idx):
        m = self.model
        m.eval()
        out = np.zeros(len(idx), dtype=np.float32)
        pos = {j: k for k, j in enumerate(idx)}
        for b in batches(np.array(idx), self.lens, self.a.bs * 4, None, shuffle=False):
            for j, v in zip(b, self._logits(b).cpu().numpy()):
                out[pos[j]] = v
        return out

    def val_loss(self, idx, pos_weight):
        lo = torch.tensor(self.predict(idx))
        return float(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))(
            lo, torch.tensor(self.y[idx])))


def pw(y):
    p = float(np.mean(y))
    return (1 - p) / max(p, 1e-6)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dp_sigma(target_eps: float, delta: float, q: float, epochs: float) -> float:
    from opacus.accountants.utils import get_noise_multiplier
    return float(get_noise_multiplier(target_epsilon=target_eps, target_delta=delta,
                                      sample_rate=q, steps=int(round(epochs / q)),
                                      accountant="rdp"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True,
                    choices=["central", "local", "fedavg", "fedper", "ditto", "fedavg_dp",
                             "dpsgd_central", "dpsgd_fedavg"])
    ap.add_argument("--source", choices=["fulltext", "abstracts", "ft_abstracts"], default="fulltext")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)       # central / local / dpsgd
    ap.add_argument("--rounds", type=int, default=8)       # federated
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--ditto-lambda", type=float, default=0.1)
    ap.add_argument("--dp-sigma", type=float, default=0.5)  # client-level (fedavg_dp)
    ap.add_argument("--dp-clip", type=float, default=1.0)
    ap.add_argument("--target-eps", type=float, default=8.0)  # record-level (dpsgd_*)
    ap.add_argument("--dp-lot", type=int, default=256)
    ap.add_argument("--max-docs", type=int, default=0, help="subsample (smoke/benchmark)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    t_start = time.time()

    from transformers import AutoTokenizer
    loader = {"abstracts": lambda: load_abstracts(a.seed), "fulltext": lambda: load_dataset(a.k, a.seed),
              "ft_abstracts": lambda: load_ft_abstracts(a.k, a.seed)}[a.source]
    df = loader()
    if a.max_docs:  # proportional subsample of every part (smoke tests / benchmarks)
        keep = []
        for _, g in df.groupby("part"):
            n = min(len(g), max(8, a.max_docs * len(g) // len(df)))
            keep.append(g.sample(n=n, random_state=a.seed))
        df = pd.concat(keep).reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tokenize(df, tok, a.max_len, a.source, tok_family(a.model))
    R = Runner(a, df, ids)
    P = {p: np.where(df["part"].to_numpy() == p)[0] for p in ["train", "val", "test", "prospective"]}
    default_len = 512 if a.source != "fulltext" else 2048
    out_dir = RES2 / ("c3_lora" + {"fulltext": "", "abstracts": "_abstracts", "ft_abstracts": "_ftabs"}[a.source]
                      + ("" if a.model == "answerdotai/ModernBERT-base" else "_" + a.model.split("/")[-1].lower())
                      + ("" if a.max_len == default_len else f"_{a.max_len}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    silos = sorted(set(df.loc[df["part"] != "prospective", "silo"]))
    S = {s: {p: np.intersect1d(P[p], np.where(df["silo"].to_numpy() == s)[0])
             for p in ["train", "val", "test"]} for s in silos}
    silos = [s for s in silos if len(S[s]["train"]) and 0 < R.y[S[s]["train"]].mean() < 1]
    tag = f"{a.regime}_s{a.seed}" + (f"_sig{a.dp_sigma}" if a.regime == "fedavg_dp" else "") + \
          (f"_eps{a.target_eps:g}" if a.regime.startswith("dpsgd") else "") + \
          ("_smoke" if a.max_docs else "")
    # prospective test: for abstracts the positives are new retractions and the negatives
    # are test controls of the same publishers; for full text both classes are in the part
    yt, st = R.y[P["test"]], df["silo"].to_numpy()[P["test"]]
    pros_ctl = None
    if a.source == "abstracts" and len(P["prospective"]):
        pubs = {p for p in df["publisher"].to_numpy()[P["prospective"]] if isinstance(p, str)}
        pros_ctl = (yt == 0) & np.isin(df["publisher"].to_numpy()[P["test"]].astype(object), list(pubs))
    print(f"[{tag}] docs {len(df)}; train {len(P['train'])} val {len(P['val'])} "
          f"test {len(P['test'])} prospective {len(P['prospective'])}; silos {silos}", flush=True)

    results, saved, log = [], {}, []
    mia_idx = np.sort(np.random.default_rng(a.seed).choice(P["train"], min(MIA_N, len(P["train"])),
                                                         replace=False))

    def add(name, view, silo, y, s):
        if len(y) and 0 < np.mean(y) < 1:
            results.append({"regime": a.regime, "seed": a.seed, "model": name, "view": view,
                             "silo": silo, **evaluate(y, s)})

    def pros_view(name, sc_test, sc_pros, model_silo):
        if sc_pros is None or not len(P["prospective"]):
            return
        if pros_ctl is not None:
            y = np.r_[R.y[P["prospective"]], yt[pros_ctl]]
            s = np.r_[sc_pros, sc_test[pros_ctl]]
        else:
            y, s = R.y[P["prospective"]], sc_pros
        add(name, "prospective", model_silo, y, s)

    def evaluate_scores(name, sc_test, sc_pros, model_silo="ALL"):
        add(name, "pooled", model_silo, yt, sc_test)
        for s in silos:
            if model_silo in ("ALL", s):
                add(name, "own_silo", s, yt[st == s], sc_test[st == s])
        if model_silo != "ALL":
            add(name, "out_silo", model_silo, yt[st != model_silo], sc_test[st != model_silo])
        pros_view(name, sc_test, sc_pros, model_silo)
        saved[f"{name}__{model_silo}__test"] = sc_test
        if sc_pros is not None:
            saved[f"{name}__{model_silo}__prospective"] = sc_pros

    def personal_pooled(name, per_silo_scores):
        """Each test article scored by its own publisher's personalised model."""
        s = np.zeros(len(yt), dtype=np.float32)
        for silo, sc in per_silo_scores.items():
            s[st == silo] = sc[st == silo]
        add(name, "pooled_personal", "ALL", yt[np.isin(st, list(per_silo_scores))],
            s[np.isin(st, list(per_silo_scores))])
        saved[f"{name}__personal__test"] = s

    def global_extras(name, state):
        """Membership-inference sample + adapter checkpoint for a global model."""
        saved[f"{name}__mia_train"] = sigmoid(R.predict(mia_idx))
        torch.save({k: v.cpu() for k, v in state.items()}, out_dir / f"adapter_{tag}_{name}.pt")

    def fit_select(train_idx, val_idx, epochs, seed):
        """Train epoch by epoch; keep the epoch with the lowest validation loss."""
        best = (np.inf, None, 0)
        for e in range(epochs):
            R.run_epochs(train_idx, 1, a.lr, seed + e, pw(R.y[train_idx]))
            vl = R.val_loss(val_idx, pw(R.y[train_idx]))
            log.append({"epoch": e + 1, "val_loss": vl})
            if vl < best[0]:
                best = (vl, trainable(R.model), e + 1)
        load_trainable(R.model, best[1])
        return best[2], best[1]

    def fed_val(state_for):
        vals = []
        for s in silos:
            load_trainable(R.model, state_for(s))
            vals.append(R.val_loss(S[s]["val"], pw(R.y[S[s]["train"]])))
        return float(np.average(vals, weights=[len(S[s]["train"]) for s in silos])), vals

    sizes = np.array([len(S[s]["train"]) for s in silos], dtype=float)
    w = sizes / sizes.sum()

    if a.regime == "central":
        ep, st_best = fit_select(P["train"], P["val"], a.epochs, a.seed * 1000)
        evaluate_scores("central_lora", sigmoid(R.predict(P["test"])), sigmoid(R.predict(P["prospective"])))
        global_extras("central_lora", st_best)
        for r in results:
            r["best_epoch"] = ep

    elif a.regime == "dpsgd_central":
        n = len(P["train"])
        sig = dp_sigma(a.target_eps, 1e-5, a.dp_lot / n, a.epochs)
        print(f"  DP-SGD central: n={n} q={a.dp_lot / n:.5f} sigma={sig:.3f} eps={a.target_eps}", flush=True)
        best = (np.inf, None, 0)
        for e in range(a.epochs):
            R.run_dpsgd(P["train"], 1, a.lr, a.seed * 1000 + e, pw(R.y[P["train"]]), sig, a.dp_clip, a.dp_lot)
            vl = R.val_loss(P["val"], pw(R.y[P["train"]]))
            log.append({"epoch": e + 1, "val_loss": vl, "min": round((time.time() - t_start) / 60, 1)})
            print(f"  epoch {e + 1}: val loss {vl:.4f} ({(time.time() - t_start) / 60:.0f} min)", flush=True)
            if vl < best[0]:
                best = (vl, trainable(R.model), e + 1)
        load_trainable(R.model, best[1])
        evaluate_scores("dpsgd_central_lora", sigmoid(R.predict(P["test"])), sigmoid(R.predict(P["prospective"])))
        global_extras("dpsgd_central_lora", best[1])
        for r in results:
            r.update(best_epoch=best[2], sigma=sig, epsilon=a.target_eps, delta=1e-5)

    elif a.regime == "local":
        per = {}
        for s in silos:
            load_trainable(R.model, R.init_state)
            fit_select(S[s]["train"], S[s]["val"], a.epochs, a.seed * 1000 + zlib.crc32(s.encode()) % 997)
            per[s] = sigmoid(R.predict(P["test"]))
            evaluate_scores("local_lora", per[s], sigmoid(R.predict(P["prospective"])), model_silo=s)
            print(f"  local {s} done ({(time.time() - t_start) / 60:.0f} min)", flush=True)
        personal_pooled("local_lora", per)

    else:  # federated regimes
        g = trainable(R.model)
        shared = [n for n in g if not (a.regime == "fedper" and is_head(n))]
        heads = {s: {n: t.clone() for n, t in g.items() if is_head(n)} for s in silos}
        personal = {s: {n: t.clone() for n, t in g.items()} for s in silos}
        best = (np.inf, g, 0)
        best_heads = {s: heads[s] for s in silos}
        best_personal = {s: (np.inf, personal[s]) for s in silos}
        gen = torch.Generator().manual_seed(a.seed)
        sig_k = {}
        if a.regime == "dpsgd_fedavg":
            for s in silos:
                n = len(S[s]["train"])
                sig_k[s] = dp_sigma(a.target_eps, 1e-5, min(1.0, a.dp_lot / n), a.rounds * a.local_epochs)
            print(f"  DP-SGD per silo sigma: { {s: round(v, 3) for s, v in sig_k.items()} }", flush=True)
        for r in range(1, a.rounds + 1):
            agg = {n: torch.zeros_like(g[n]) for n in shared}
            for k, s in enumerate(silos):
                load_trainable(R.model, g)
                if a.regime == "fedper":
                    load_trainable(R.model, heads[s])
                seed_k = a.seed * 100003 + r * 101 + k
                if a.regime == "dpsgd_fedavg":
                    lot = min(a.dp_lot, len(S[s]["train"]))
                    R.run_dpsgd(S[s]["train"], a.local_epochs, a.lr, seed_k, pw(R.y[S[s]["train"]]),
                                sig_k[s], a.dp_clip, lot)
                else:
                    R.run_epochs(S[s]["train"], a.local_epochs, a.lr, seed_k, pw(R.y[S[s]["train"]]))
                loc = trainable(R.model)
                if a.regime == "fedper":
                    heads[s] = {n: loc[n].clone() for n in heads[s]}
                delta = {n: loc[n] - g[n] for n in shared}
                if a.regime == "fedavg_dp":
                    norm = math.sqrt(sum(float((d.float() ** 2).sum()) for d in delta.values()))
                    f = min(1.0, a.dp_clip / (norm + 1e-12))
                    delta = {n: d * f for n, d in delta.items()}
                for n in agg:
                    agg[n] += float(w[k]) * delta[n]
                if a.regime == "ditto":  # personal model: one local epoch pulled towards g
                    load_trainable(R.model, personal[s])
                    R.run_epochs(S[s]["train"], a.local_epochs, a.lr, seed_k + 7, pw(R.y[S[s]["train"]]),
                                 prox=(a.ditto_lambda, g))
                    personal[s] = trainable(R.model)
                    vl_s = R.val_loss(S[s]["val"], pw(R.y[S[s]["train"]]))
                    if vl_s < best_personal[s][0]:
                        best_personal[s] = (vl_s, {n: t.clone() for n, t in personal[s].items()})
            if a.regime == "fedavg_dp":
                sens = float(w.max()) * a.dp_clip
                for n in agg:
                    agg[n] += (torch.randn(agg[n].shape, generator=gen) * a.dp_sigma * sens
                               ).to(agg[n].device, agg[n].dtype)
            g = {**g, **{n: g[n] + agg[n] for n in shared}}
            if a.regime == "fedper":
                vl, _ = fed_val(lambda s: {**g, **heads[s]})
            else:
                vl, _ = fed_val(lambda s: g)
            log.append({"round": r, "val_loss": vl, "min": round((time.time() - t_start) / 60, 1)})
            print(f"  round {r}: fed val loss {vl:.4f} ({(time.time() - t_start) / 60:.0f} min)", flush=True)
            if vl < best[0]:
                best = (vl, {n: t.clone() for n, t in g.items()}, r)
                best_heads = {s: {n: t.clone() for n, t in heads[s].items()} for s in silos}
        name = {"fedavg": "fl_fedavg_lora", "fedavg_dp": "fl_dp_lora", "fedper": "fl_fedper_lora",
                "ditto": "fl_fedavg_lora", "dpsgd_fedavg": "dpsgd_fedavg_lora"}[a.regime]
        if a.regime == "fedper":
            per = {}
            for s in silos:
                load_trainable(R.model, {**best[1], **best_heads[s]})
                per[s] = sigmoid(R.predict(P["test"]))
                evaluate_scores(name, per[s], None, model_silo=s)
            personal_pooled(name, per)
        else:
            load_trainable(R.model, best[1])
            evaluate_scores(name, sigmoid(R.predict(P["test"])), sigmoid(R.predict(P["prospective"])))
            global_extras(name, best[1])
        for r in results:
            r["best_round"] = best[2]
            if a.regime == "dpsgd_fedavg":
                r.update(epsilon=a.target_eps, delta=1e-5)
        if a.regime == "fedavg":
            per = {}
            for s in silos:  # personalised: one local epoch from the global model
                load_trainable(R.model, best[1])
                R.run_epochs(S[s]["train"], 1, a.lr / 3, a.seed + 7, pw(R.y[S[s]["train"]]))
                per[s] = sigmoid(R.predict(P["test"]))
                evaluate_scores("fl_fedavg_lora_ft", per[s], None, model_silo=s)
            personal_pooled("fl_fedavg_lora_ft", per)
        if a.regime == "ditto":
            per = {}
            for s in silos:
                load_trainable(R.model, best_personal[s][1])
                per[s] = sigmoid(R.predict(P["test"]))
                evaluate_scores("fl_ditto_lora", per[s], None, model_silo=s)
            personal_pooled("fl_ditto_lora", per)

    secs = time.time() - t_start
    for r in results:
        r.update(gpu_hours=round(secs / 3600, 2), tokens_seen=R.tokens_seen)
    pd.DataFrame(results).to_csv(out_dir / f"results_{tag}.csv", index=False)
    np.savez_compressed(out_dir / f"scores_{tag}.npz", y_test=yt, silo_test=st,
                        test_idx=P["test"], y_pros=R.y[P["prospective"]],
                        pros_ctl=pros_ctl if pros_ctl is not None else np.zeros(0, bool),
                        mia_idx=mia_idx, y_mia=R.y[mia_idx], **saved)
    json.dump(log, open(out_dir / f"log_{tag}.json", "w"))
    print(f"[{tag}] done in {secs / 3600:.2f} h; {R.tokens_seen / secs:,.0f} tok/s")
    print(pd.DataFrame(results).query("view in ['pooled', 'pooled_personal', 'prospective']")
          [["model", "view", "silo", "auprc", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
