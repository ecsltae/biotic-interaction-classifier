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
# INTERACT_THR 0.50 -- the shipped default, fixed before evaluation (not tuned). Raise it to
#                      trade recall for precision; README.md gives the exact trade.
# DIR_ABSTAIN  0.60 -- report FORWARD/REVERSE only when the direction head is at least this
#                      confident; below it the answer is UNCERTAIN rather than a guess.
INTERACT_THR = 0.50
DIR_ABSTAIN = 0.60

# The four direction categories. BIDIRECTIONAL is a property of the relation, not a guess:
# the relation lexicon calls it symmetric (mutualism, interacts with, co-occurs with ...), so
# neither taxon is the subject. UNCERTAIN means the model could not decide, or a taxon was not
# found in the passage, or the pair does not interact at all.
DIRECTIONS = ("FORWARD", "REVERSE", "BIDIRECTIONAL", "UNCERTAIN")


def load(model_dir, device="cpu", threads=8):
    torch.set_num_threads(threads)
    md = Path(model_dir)
    cfg = json.loads((md / "student_config.json").read_text())
    if cfg.get("input_format", "mark_canon") != "mark_canon":
        raise SystemExit(f"{md} was trained with input_format={cfg.get('input_format')!r}; "
                         f"this scorer builds mark_canon inputs and would silently mis-score it.")
    dir_state = torch.load(md / "direction_head.pt", map_location="cpu")
    m = Student(enc=str(md), detach_dir=cfg.get("detach_dir", False), dir_state=dir_state)
    tok = AutoTokenizer.from_pretrained(md, local_files_only=True)
    return m.to(device).eval(), tok, cfg


# Load the polarity lexicon ONCE, at import, and fail loudly if it is missing.
# A per-call try/except here would swallow a missing lexicon and hand every relation the
# same default polarity -- direction would quietly degrade to near-chance with exit code 0.
import candidate_rules as _rules                          # noqa: E402

try:
    import polarity as _polarity_mod
except Exception as _e:                                    # pragma: no cover
    raise SystemExit(
        f"cannot load the relation-polarity lexicon: {type(_e).__name__}: {_e}\n"
        f"polarity.py, robi_maps.json and data/robiext_v2025.json must sit beside this "
        f"script. Direction output is not meaningful without them.") from _e


def _polarity(rel, n_pol):
    """(head index, lexicon polarity, unknown?) for one relation string.

    The index follows polarity.polarity_to_index, which is the mapping training used. An earlier
    version of this function sent symmetric and unknown relations to the agent index, which the
    head never saw for them in training.
    """
    p, _src = _polarity_mod.polarity(str(rel))
    return _polarity_mod.polarity_to_index(p, n_pol), p, p is None


@torch.no_grad()
def predict(m, tok, s1, s2, rel, text, device="cpu", bs=16, max_len=256, rules=True):
    s1, s2, rel = [list(map(str, v)) for v in (s1, s2, rel)]
    text = [str(t) for t in text]
    a, b = X.build_many("mark_canon", s1, rel, s2, text)
    n_pol = m.dir.n_pol
    _pl = [_polarity(r, n_pol) for r in rel]
    pol = np.array([x[0] for x in _pl], dtype="int64")
    lex = [x[1] for x in _pl]
    unknown = np.array([x[2] for x in _pl])
    # how much of each passage survives the window: segment B is cut first ("only_second")
    full = [len(tok(x, y, add_special_tokens=True)["input_ids"]) for x, y in zip(a, b)]
    truncated = np.array([n > max_len for n in full])
    P, D, OK = [], [], []
    for i in range(0, len(a), bs):
        e = tok(a[i:i + bs], b[i:i + bs], truncation="only_second",
                max_length=max_len, padding=True, return_tensors="pt").to(device)
        pt = torch.tensor(pol[i:i + bs], dtype=torch.long, device=device)
        logits, s, ok, _u = m(e["input_ids"], e["attention_mask"], e.get("token_type_ids"), pol=pt)
        P.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        D.extend(s.float().cpu().numpy())
        OK.extend(ok.float().cpu().numpy())
    P, D, OK = np.array(P), np.array(D), np.array(OK)

    # candidate rules: deterministic rejections of candidates that cannot be an interaction
    # between two distinct organisms (candidate_rules.py; validated on training data)
    if rules:
        why = [_rules.reject_reason(t, x, r, y) or "" for t, x, r, y in zip(text, s1, rel, s2)]
    else:
        why = [""] * len(text)
    interacts = ((P >= INTERACT_THR) & np.array([w == "" for w in why])).astype(int)

    # the @ taxon is the alphabetically first one; decode back to the order you gave us
    at_is_s1 = np.array([x.lower() <= y.lower() for x, y in zip(s1, s2)])
    p_at_subject = 1 / (1 + np.exp(-D))
    p_s1_subject = np.where(at_is_s1, p_at_subject, 1 - p_at_subject)
    conf = np.abs(p_s1_subject - 0.5) * 2
    symmetric = np.array([_polarity_mod.is_symmetric(p) for p in lex])
    direction = np.where(conf < DIR_ABSTAIN, "UNCERTAIN",
                         np.where(p_s1_subject >= 0.5, "FORWARD", "REVERSE"))
    direction = np.where(symmetric, "BIDIRECTIONAL", direction)
    direction = np.where(OK > 0, direction, "UNCERTAIN")   # a taxon was not found in the passage
    direction = np.where(interacts == 1, direction, "UNCERTAIN")  # no interaction, no direction

    return pd.DataFrame({
        "species1": s1, "relation": rel, "species2": s2,
        "interacts": interacts,
        "p_interact": P.round(4),
        "rejected_by_rule": why,
        "direction": direction,
        "p_species1_is_subject": np.where(symmetric, np.nan, p_s1_subject).round(4),
        "direction_confidence": np.where(symmetric, np.nan, conf).round(4),
        "symmetric_relation": symmetric.astype(int),
        "both_taxa_located": (OK > 0).astype(int),
        "unknown_polarity": unknown.astype(int),
        "truncated": truncated.astype(int),
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
    ap.add_argument("--max-len", type=int, default=None,
                    help="wordpiece window; default is the length the model was trained with")
    ap.add_argument("--no-rules", action="store_true",
                    help="skip the deterministic candidate rules (candidate_rules.py)")
    a = ap.parse_args()
    if a.threshold is not None:
        INTERACT_THR = a.threshold

    if not Path(a.model).exists():
        sys.exit(f"model directory not found: {a.model}\n"
                 f"Pass --model /path/to/joint_a05_s1, or put the weights in {HERE / 'model'}.")
    m, tok, cfg = load(a.model, threads=a.threads)
    max_len = a.max_len or int(cfg.get("max_len", 256))

    if a.inp:
        d = pd.read_csv(a.inp)
        need = {"species1", "relation", "species2", "sentence"}
        missing = need - set(d.columns)
        if missing:
            sys.exit(f"input is missing required columns: {sorted(missing)}\n"
                     f"found: {list(d.columns)}")
        need_text = ["species1", "relation", "species2", "sentence"]
        blank = d[need_text].isna().any(axis=1) | (
            d[need_text].astype(str).apply(lambda col: col.str.strip() == "").any(axis=1))
        if blank.any():
            rows = ", ".join(str(i) for i in d.index[blank][:10])
            sys.exit(f"{int(blank.sum())} row(s) have an empty or missing value in one of "
                     f"{need_text}: rows {rows}{' ...' if blank.sum() > 10 else ''}\n"
                     f"These would be scored as the literal text 'nan'. Fix or drop them first.")

        r = predict(m, tok, d.species1, d.species2, d.relation, d.sentence,
                    bs=a.batch_size, max_len=max_len, rules=not a.no_rules)
        clashes = [c for c in d.columns if c in r.columns and c not in need_text]
        if clashes:
            print(f"note: your input has column(s) {clashes} whose names collide with this "
                  f"tool's output; yours are kept as {[c + '_input' for c in clashes]}",
                  file=sys.stderr)
        for c in d.columns:                       # carry the caller's own columns through
            if c in need_text:
                continue
            r[c if c not in r.columns else f"{c}_input"] = d[c].values
        if a.out:
            r.to_csv(a.out, index=False)
            n = int(r.interacts.sum())
            dirs = r.direction[r.interacts == 1].value_counts().to_dict()
            print(f"wrote {a.out}: {len(r)} candidates, {n} accepted ({n/max(len(r),1):.1%}), "
                  f"{int((r.rejected_by_rule != '').sum())} rejected by a rule; directions of the "
                  f"accepted: {dirs}")
            nt = int(r.truncated.sum())
            if nt:
                print(f"note: {nt} passage(s) were longer than {max_len} wordpieces and were "
                      f"truncated; see the `truncated` column", file=sys.stderr)
        else:
            print(r.to_string(index=False))
    elif a.s1:
        if not all(str(x).strip() for x in (a.s1, a.s2, a.rel, a.text)):
            ap.error("--s1, --rel, --s2 and --text must all be non-empty")
        r = predict(m, tok, [a.s1], [a.s2], [a.rel], [a.text], max_len=max_len,
                    rules=not a.no_rules)
        print(json.dumps(r.iloc[0].to_dict(), indent=2, default=float))
    else:
        ap.error("give --in CSV, or --s1/--rel/--s2/--text for a single candidate")


if __name__ == "__main__":
    main()
