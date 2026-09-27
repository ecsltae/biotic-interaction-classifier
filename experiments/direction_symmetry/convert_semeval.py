#!/usr/bin/env python3
"""SemEval-2010 Task 8 -> our direction format.

The canonical directional relation-extraction benchmark, and general-domain rather than
biomedical, so it tests whether the symmetry result is a property of relation extraction or an
artefact of one field. Direction is carried in the label: Cause-Effect(e1,e2) against
Cause-Effect(e2,e1). 'Other' means no relation holds, which we use as the undirected class.
"""
import re
import pandas as pd
from datasets import load_dataset

E1 = re.compile(r"<e1>(.*?)</e1>", re.S)
E2 = re.compile(r"<e2>(.*?)</e2>", re.S)
STRIP = re.compile(r"</?e[12]>")


def convert(split_name, ds):
    names = ds.features["relation"].names
    rows = []
    for ex in ds:
        s = ex["sentence"]
        m1, m2 = E1.search(s), E2.search(s)
        if not m1 or not m2:
            continue
        lab = names[ex["relation"]]
        if lab == "Other":
            rel, subj, directed = "Other", -1, 0
        else:
            rel = lab.split("(")[0]
            subj = 1 if lab.endswith("(e1,e2)") else 2
            directed = 1
        e1, e2 = m1.group(1).strip(), m2.group(1).strip()
        if e1.lower() == e2.lower():
            continue
        rows.append(dict(s1=e1, s2=e2, rel=rel, passage=STRIP.sub("", s).strip(),
                         subject_is=subj, directed=directed))
    df = pd.DataFrame(rows)
    out = f"semeval_{split_name}_ours.csv"
    df.to_csv(out, index=False)
    d = df[df.directed == 1]
    print(f"{split_name:6} -> {out}: {len(df):5,} rows  "
          f"(directed {len(d):,}, Other {int((df.directed==0).sum()):,})")
    print(f"        subject_is among directed: {d.subject_is.value_counts().to_dict()}  "
          f"relation types: {df.rel.nunique()}")
    return df


if __name__ == "__main__":
    ds = load_dataset("sem_eval_2010_task_8")
    tr = convert("train", ds["train"])
    te = convert("test", ds["test"])
    print(f"\nrelation types: {sorted(tr.rel.unique())}")
