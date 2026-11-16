# NOTE: The model checkpoint (model.pt) could not be included due to
# Kaggle GPU quota limits and a corrupted download. The script is fully
# functional and will work once the checkpoint is placed in
# artifacts/transformer/model.pt.

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TwoHead
from text_norm import normalize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ART = Path(__file__).resolve().parent.parent / "artifacts"
REC, BAND, MAX_LEN = ["recommended", "not_recommended", "no_idea"], ["low", "medium", "high"], 128


def load_models():
    tdir = ART / "transformer"
    config = AutoConfig.from_pretrained("HooshvareLab/bert-fa-base-uncased")
    model = TwoHead(config)
    missing, unexpected = model.load_state_dict(
        torch.load(tdir / "model.pt", map_location="cpu"), strict=False)
    bad = list(missing) + [k for k in unexpected if not k.endswith("position_ids")]
    if bad:
        raise RuntimeError(f"checkpoint mismatch: {bad[:5]}")
    model.eval()
    vec = joblib.load(ART / "tfidf" / "tfidf_vectorizer.joblib")
    return {"vec": vec, "terms": vec.get_feature_names_out(),
            "clf": {"recommendation": joblib.load(ART / "tfidf" / "logreg_recommendation.joblib"),
                    "rating_band": joblib.load(ART / "tfidf" / "logreg_rating_band.joblib")},
            "tok": AutoTokenizer.from_pretrained(tdir / "tokenizer"), "model": model}


@torch.no_grad()
def tr_probs(M, texts, bs=32):
    pr, pb = [], []
    for i in range(0, len(texts), bs):
        enc = M["tok"](texts[i:i + bs], truncation=True, max_length=MAX_LEN,
                       padding=True, return_tensors="pt")
        a, b = M["model"](enc["input_ids"], enc["attention_mask"])
        pr.append(F.softmax(a, -1)); pb.append(F.softmax(b, -1))
    return torch.cat(pr).numpy(), torch.cat(pb).numpy()


def explain_tfidf(M, x, head, label, k=5):
    clf = M["clf"][head]
    row, ci = x.tocoo(), list(clf.classes_).index(label)
    contrib = row.data * clf.coef_[ci][row.col]
    order = np.argsort(-contrib)
    f = lambda idx: [(str(M["terms"][row.col[i]]), round(float(contrib[i]), 3)) for i in idx]
    return {"supports": f([i for i in order[:k] if contrib[i] > 0]),
            "opposes": f([i for i in order[::-1][:k] if contrib[i] < 0])}


def explain_occlusion(M, text, head, ci, base_p, max_words=60, k=5):
    words = text.split()
    if not words or len(words) > max_words:
        return {"note": f"skipped (more than {max_words} words)"}
    variants = [" ".join(words[:i] + words[i + 1:]) or "." for i in range(len(words))]
    P = tr_probs(M, variants)[0 if head == "recommendation" else 1]
    drop = base_p - P[:, ci]
    order = np.argsort(-drop)
    f = lambda idx: [(words[i], round(float(drop[i]), 3)) for i in idx]
    return {"supports": f([i for i in order[:k] if drop[i] > 0]),
            "opposes": f([i for i in order[::-1][:k] if drop[i] < 0])}


def analyze(M, text, explain=True):
    norm = normalize(text)
    if not norm:
        return {"original": text, "error": "empty text after normalization"}
    ids = M["tok"](norm, truncation=False)["input_ids"]
    res = {"original": text, "normalized": norm,
           "tokens": {"count": len(ids), "max_len": MAX_LEN, "truncated": len(ids) > MAX_LEN,
                      "first_tokens": M["tok"].convert_ids_to_tokens(ids[:12])}}

    t0 = time.perf_counter(); x = M["vec"].transform([norm])
    tf = {h: dict(zip(M["clf"][h].classes_, M["clf"][h].predict_proba(x)[0]))
          for h in ("recommendation", "rating_band")}
    lat_tf = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter(); pr, pb = tr_probs(M, [norm])
    lat_tr = (time.perf_counter() - t0) * 1000
    tr = {"recommendation": dict(zip(REC, pr[0])), "rating_band": dict(zip(BAND, pb[0]))}

    for name, probs, lat in (("tfidf", tf, lat_tf), ("transformer", tr, lat_tr)):
        out = {"latency_ms": round(lat, 1)}
        for head, classes in (("recommendation", REC), ("rating_band", BAND)):
            p = {c: round(float(probs[head][c]), 4) for c in classes}
            label = max(p, key=p.get)
            out[head] = {"label": label, "probs": p}
            if explain:
                out[head]["explanation"] = (
                    explain_tfidf(M, x, head, label) if name == "tfidf"
                    else explain_occlusion(M, norm, head, classes.index(label), p[label]))
        res[name] = out
    return res


def show(r):
    print("=" * 72)
    print("original  :", r["original"])
    if "error" in r:
        print("error     :", r["error"]); return
    print("normalized:", r["normalized"])
    t = r["tokens"]
    print(f"tokens    : {t['count']} (max {t['max_len']}, truncated={t['truncated']}) {t['first_tokens']}")
    for name in ("tfidf", "transformer"):
        print(f"\n[{name}]  {r[name]['latency_ms']} ms")
        for head in ("recommendation", "rating_band"):
            h = r[name][head]
            print(f"  {head:15s} → {h['label']:16s} " + "  ".join(f"{k}={v:.2f}" for k, v in h["probs"].items()))
            ex = h.get("explanation") or {}
            for key in ("supports", "opposes"):
                if ex.get(key):
                    print(f"      {key:8s}: " + ", ".join(f"{w}({s:+.2f})" for w, s in ex[key]))
            if ex.get("note"):
                print("      note    :", ex["note"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--file", help="a text file, one review per line")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-explain", action="store_true")
    a = ap.parse_args()
    texts = list(a.texts)
    if a.file:
        texts += [l.strip() for l in Path(a.file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not texts:
        ap.error("give a review text or --file")
    M = load_models()
    results = [analyze(M, t, explain=not a.no_explain) for t in texts]
    print(json.dumps(results, ensure_ascii=False, indent=2)) if a.json else [show(r) for r in results]