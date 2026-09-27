#!/usr/bin/env python3
"""Inference for the joint interaction + direction model, CPU by default.

The model answers two questions about a candidate triple (taxon1, relation, taxon2)
and the passage it came from:

  interacts   -- do these two taxa interact, per this passage?  (symmetric)
  direction   -- which taxon is the subject of the relation?    (antisymmetric)

Direction is decoded outside the model. The encoder sees the two taxa sorted
alphabetically, marked @first@ and #second# in the passage, so swapping the input
order gives a byte-identical input: `interacts` is order-invariant by construction,
and the direction answer flips exactly as it should.

Usage
  python predict_joint.py --model models/dirhead/joint_a05_s1 --in pairs.csv --out scored.csv
  python predict_joint.py --model models/dirhead/joint_a05_s1 \
      --s1 "Haemophilus influenzae" --rel "pathogen of" --s2 human --text "..."

Input CSV needs: species1, relation, species2, sentence.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "direction"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import xenc_format as X                                   # noqa: E402
from train_direction import Student                       # noqa: E402
from transformers import AutoTokenizer                    # noqa: E402

# Operating points. Both are pre-specified, not fitted on any reporting set.
#   INTERACT_THR -- 0.5. Precision has priority over recall by project policy;
#                   raise it to trade recall for precision.
#   DIR_ABSTAIN  -- report a direction only when the head is at least this confident.
#                   At 0.71 the head was right 9/10 on the 17-item gold (see the model
#                   card). That estimate rests on 17 items; treat it as indicative.
INTERACT_THR = 0.5
DIR_ABSTAIN = 0.60


def load(md, device="cpu", threads=8):
    torch.set_num_threads(threads)
    md = Path(md)
    cfg = json.loads((md / "student_config.json").read_text())
    m = Student(enc=str(md), detach_dir=cfg.get("detach_dir", False))
    m.dir.load_state_dict(torch.load(md / "direction_head.pt", map_location="cpu"))
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    return m.to(device).eval(), tok, cfg


# Load the lexicon once, at import, and fail loudly. A per-call try/except would swallow a
# missing lexicon and give every relation the same default polarity, quietly degrading the
# direction output to near-chance with exit code 0.
try:
    import polarity as _polarity_mod
except Exception as _e:                                    # pragma: no cover
    raise SystemExit(
        f"cannot load the relation-polarity lexicon: {type(_e).__name__}: {_e}\n"
        f"experiments/direction/polarity.py + robi_maps.json + "
        f"data/taxonomies/robiext_v2025.json are required for direction output.") from _e


def polarity_of(rel, ro_id=None):
    """Relation polarity feeds the direction head. Returns 1 (agent-side) or 0."""
    p, _ = _polarity_mod.polarity(str(rel), ro_id)
    return 1 if (p is not None and p > 0) else 0


@torch.no_grad()
def predict(m, tok, s1, s2, rel, text, device="cpu", bs=16, max_len=256):
    s1, s2, rel = list(map(str, s1)), list(map(str, s2)), list(map(str, rel))
    text = [str(t) for t in text]
    a, b = X.build_many("mark_canon", s1, rel, s2, text)
    pol = np.array([polarity_of(r) for r in rel], dtype="int64")
    P, D, OK = [], [], []
    for i in range(0, len(a), bs):
        e = tok(a[i:i + bs], b[i:i + bs], truncation="only_second",
                max_length=max_len, padding=True, return_tensors="pt").to(device)
        pt = torch.tensor(pol[i:i + bs], dtype=torch.long, device=device)
        logits, s, ok = m(e["input_ids"], e["attention_mask"], e.get("token_type_ids"), pol=pt)
        P.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        D.extend(s.float().cpu().numpy())
        OK.extend(ok.float().cpu().numpy())
    P, D, OK = np.array(P), np.array(D), np.array(OK)

    # the @ taxon is the alphabetically first one; decode back to stored order
    at_is_s1 = np.array([x.lower() <= y.lower() for x, y in zip(s1, s2)])
    p_at_subject = 1 / (1 + np.exp(-D))
    p_s1_subject = np.where(at_is_s1, p_at_subject, 1 - p_at_subject)
    conf = np.abs(p_s1_subject - 0.5) * 2

    direction = np.where(conf < DIR_ABSTAIN, "UNCERTAIN",
                         np.where(p_s1_subject >= 0.5, "FORWARD", "REVERSE"))
    direction = np.where(OK > 0, direction, "UNCERTAIN")   # a taxon was not found in the passage
    return pd.DataFrame({
        "species1": s1, "relation": rel, "species2": s2,
        "p_interact": P, "interacts": (P >= INTERACT_THR).astype(int),
        "direction": direction, "p_species1_is_subject": p_s1_subject,
        "direction_confidence": conf, "both_taxa_located": (OK > 0).astype(int),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out")
    ap.add_argument("--s1"); ap.add_argument("--rel"); ap.add_argument("--s2"); ap.add_argument("--text")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    m, tok, cfg = load(a.model, a.device, a.threads)
    if a.inp:
        d = pd.read_csv(a.inp)
        miss = {"species1", "relation", "species2", "sentence"} - set(d.columns)
        if miss:
            sys.exit(f"input is missing columns: {sorted(miss)}")
        r = predict(m, tok, d.species1, d.species2, d.relation, d.sentence,
                    a.device, a.batch_size)
        for c in d.columns:
            if c not in r.columns:
                r[c] = d[c].values
        if a.out:
            r.to_csv(a.out, index=False); print(f"wrote {a.out}  ({len(r)} rows)")
        else:
            print(r.to_string(index=False))
    elif a.s1:
        r = predict(m, tok, [a.s1], [a.s2], [a.rel], [a.text], a.device, 1)
        print(json.dumps(r.iloc[0].to_dict(), indent=2, default=float))
    else:
        sys.exit("give --in CSV or --s1/--rel/--s2/--text")


if __name__ == "__main__":
    main()
