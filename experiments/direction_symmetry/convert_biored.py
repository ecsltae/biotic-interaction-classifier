#!/usr/bin/env python3
"""Convert BioRED into the input format of our direction model.

Their processed TSV carries the pair, the relation and the direction label but its text is
pre-tokenised with their own Src/Tgt markers. PubTator carries clean title+abstract and entity
mention offsets. Join the two: labels from the TSV, strings from PubTator.

BioRED is document-level, so the passage is a title+abstract of ~1500 characters against our
256-token budget. We take the INFIX window -- the smallest span of sentences containing both
entities -- which is what BioREDirect's own chunking does for the same reason.
"""
import csv, re, sys
from collections import defaultdict
import pandas as pd


def read_pubtator(path):
    docs = {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            cur = None; continue
        if "|t|" in line[:24]:
            pmid, _, txt = line.split("|", 2)
            docs.setdefault(pmid, {"title": txt, "abs": "", "ents": defaultdict(list)})
            cur = pmid
        elif "|a|" in line[:24]:
            pmid, _, txt = line.split("|", 2)
            docs.setdefault(pmid, {"title": "", "abs": "", "ents": defaultdict(list)})
            docs[pmid]["abs"] = txt
            cur = pmid
        else:
            f = line.split("\t")
            if len(f) >= 6 and f[1].isdigit():          # an entity annotation
                pmid, s, e, mention, typ, ids = f[0], int(f[1]), int(f[2]), f[3], f[4], f[5]
                for i in ids.split(";"):
                    docs[pmid]["ents"][i].append((s, e, mention))
    return docs


SENT = re.compile(r"(?<=[.!?])\s+")


def infix(text, spans):
    """Smallest window of sentences covering every given character span."""
    if not spans:
        return text[:1200]
    lo, hi = min(s for s, _ in spans), max(e for _, e in spans)
    bounds, pos = [], 0
    for part in SENT.split(text):
        bounds.append((pos, pos + len(part)))
        pos += len(part) + 1
    keep = [b for b in bounds if not (b[1] < lo or b[0] > hi)]
    if not keep:
        return text[max(0, lo - 200):hi + 200]
    return text[keep[0][0]:keep[-1][1]]


def main(split):
    docs = read_pubtator(f"bioredirect/bioredirect_{split}.pubtator")
    rows, skipped = [], 0
    for r in csv.reader(open(f"bioredirect/processed/{'bc8_test' if split=='bc8_test' else split}.tsv"),
                        delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(r) < 9:
            continue
        pmid, e1id, e2id, rel, subj = r[0], r[3], r[4], r[6], r[8]
        if rel in ("None", ""):
            continue                                     # direction is only defined on real relations
        d = docs.get(pmid)
        if not d or e1id not in d["ents"] or e2id not in d["ents"]:
            skipped += 1; continue
        text = (d["title"] + " " + d["abs"]).strip()
        a = max(d["ents"][e1id], key=lambda x: x[1] - x[0])
        b = max(d["ents"][e2id], key=lambda x: x[1] - x[0])
        if a[2].lower() == b[2].lower():
            skipped += 1; continue                       # same surface string: no pair to orient
        rows.append(dict(
            pmid=pmid, s1=a[2], s2=b[2], rel=rel,
            passage=infix(text, [(a[0], a[1]), (b[0], b[1])]),
            subject_is=1 if subj == e1id else (2 if subj == e2id else -1),
            directed=int(subj in (e1id, e2id))))
    df = pd.DataFrame(rows)
    out = f"biored_{split}_ours.csv"
    df.to_csv(out, index=False)
    print(f"{split:9} -> {out}: {len(df):5,} rows  "
          f"(directed {int(df.directed.sum()):,}, undirected {int((df.directed==0).sum()):,}, "
          f"skipped {skipped})")
    print(f"            subject_is: {df.subject_is.value_counts().to_dict()}  "
          f"passage len median {int(df.passage.str.len().median())}")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["train", "dev", "test"]):
        main(s)
