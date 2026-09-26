#!/usr/bin/env python3
"""Score candidate biotic-interaction triples: does the pair interact, and in which direction.

  python predict.py --in candidates.csv --out scored.csv
  python predict.py --s1 "Haemophilus influenzae" --rel "pathogen of" --s2 human \
                    --text "H. influenzae is a major pathogen of humans."

Input CSV needs four columns: species1, relation, species2, sentence.
Any other columns you supply are carried through to the output untouched.

Runs on CPU. ~34 candidate/s on 8 threads.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xenc_format as X                                  # noqa: E402
from joint_model import Student                                # noqa: E402
from transformers import AutoTokenizer                   # noqa: E402

# ---------------------------------------------------------------- operating points
# INTERACT_THR 0.50 -- the shipped default. On the 437-row evaluation set this gives
#                      precision 0.852 / recall 0.959. Raise it to trade recall for
#                      precision; the table in README.md gives the exact trade.
# DIR_ABSTAIN  0.71 -- report a direction only when the head is at least this confident.
#                      Below it the output is UNCERTAIN rather than a guess.
INTERACT_THR = 0.50
DIR_ABSTAIN = 0.71


def load(model_dir, device="cpu", threads=8):
    torch.set_num_threads(threads)
    md = Path(model_dir)
    cfg = json.loads((md / "student_config.json").read_text())
    m = Student(enc=str(md), detach_dir=cfg.get("detach_dir", False))
    m.dir.load_state_dict(torch.load(md / "direction_head.pt", map_location="cpu"))
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    return m.to(device).eval(), tok, cfg


# Load the polarity lexicon ONCE, at import, and fail loudly if it is missing.
# A per-call try/except here would swallow a missing lexicon and hand every relation the
# same default polarity -- direction would quietly degrade to near-chance with exit code 0.
try:
    import polarity as _polarity_mod
except Exception as _e:                                    # pragma: no cover
    raise SystemExit(
        f"cannot load the relation-polarity lexicon: {type(_e).__name__}: {_e}\n"
        f"polarity.py, robi_maps.json and data/robiext_v2025.json must sit beside this "
        f"script. Direction output is not meaningful without them.") from _e


def _polarity(rel):
    """Relation polarity feeds the direction head: 1 = agent-side subject, 0 = patient-side.

    An unknown relation legitimately falls back to agent-side; `unknown_polarity` in the
    output counts how often that happened so you can see it.
    """
    p, _src = _polarity_mod.polarity(str(rel))
    if p is None or p == 0:
        return 1, True                                      # fallback used
    return (1 if p > 0 else 0), False


@torch.no_grad()
def predict(m, tok, s1, s2, rel, text, device="cpu", bs=16, max_len=256):
    s1, s2, rel = [list(map(str, v)) for v in (s1, s2, rel)]
    text = [str(t) for t in text]
    a, b = X.build_many("mark_canon", s1, rel, s2, text)
    _pl = [_polarity(r) for r in rel]
    pol = np.array([x[0] for x in _pl], dtype="int64")
    unknown = np.array([x[1] for x in _pl])
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

    # the @ taxon is the alphabetically first one; decode back to the order you gave us
    at_is_s1 = np.array([x.lower() <= y.lower() for x, y in zip(s1, s2)])
    p_at_subject = 1 / (1 + np.exp(-D))
    p_s1_subject = np.where(at_is_s1, p_at_subject, 1 - p_at_subject)
    conf = np.abs(p_s1_subject - 0.5) * 2
    direction = np.where(conf < DIR_ABSTAIN, "UNCERTAIN",
                         np.where(p_s1_subject >= 0.5, "FORWARD", "REVERSE"))
    direction = np.where(OK > 0, direction, "UNCERTAIN")   # a taxon was not found in the passage

    return pd.DataFrame({
        "species1": s1, "relation": rel, "species2": s2,
        "interacts": (P >= INTERACT_THR).astype(int),
        "p_interact": P.round(4),
        "direction": direction,
        "p_species1_is_subject": p_s1_subject.round(4),
        "direction_confidence": conf.round(4),
        "both_taxa_located": (OK > 0).astype(int),
        "unknown_polarity": unknown.astype(int),
    })


def main():
    global INTERACT_THR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(HERE / "model"))
    ap.add_argument("--in", dest="inp", help="CSV with species1, relation, species2, sentence")
    ap.add_argument("--out", help="where to write the scored CSV")
    ap.add_argument("--s1"); ap.add_argument("--rel"); ap.add_argument("--s2"); ap.add_argument("--text")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"override the interaction threshold (default {INTERACT_THR})")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    if a.threshold is not None:
        INTERACT_THR = a.threshold

    if not Path(a.model).exists():
        sys.exit(f"model directory not found: {a.model}\n"
                 f"Pass --model /path/to/joint_a05_s1, or put the weights in {HERE / 'model'}.")
    m, tok, cfg = load(a.model, threads=a.threads)

    if a.inp:
        d = pd.read_csv(a.inp)
        need = {"species1", "relation", "species2", "sentence"}
        missing = need - set(d.columns)
        if missing:
            sys.exit(f"input is missing required columns: {sorted(missing)}\n"
                     f"found: {list(d.columns)}")
        r = predict(m, tok, d.species1, d.species2, d.relation, d.sentence,
                    bs=a.batch_size)
        for c in d.columns:                       # carry the caller's own columns through
            if c not in r.columns:
                r[c] = d[c].values
        if a.out:
            r.to_csv(a.out, index=False)
            n = int(r.interacts.sum())
            print(f"wrote {a.out}: {len(r)} candidates, {n} accepted ({n/max(len(r),1):.1%}), "
                  f"{int((r.direction != 'UNCERTAIN').sum())} with a direction")
        else:
            print(r.to_string(index=False))
    elif a.s1:
        r = predict(m, tok, [a.s1], [a.s2], [a.rel], [a.text])
        print(json.dumps(r.iloc[0].to_dict(), indent=2, default=float))
    else:
        ap.error("give --in CSV, or --s1/--rel/--s2/--text for a single candidate")


if __name__ == "__main__":
    main()
