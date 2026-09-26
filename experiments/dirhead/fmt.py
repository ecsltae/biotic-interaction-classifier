"""mark_canon encoding that also returns the character spans of the two marked mentions."""
import sys, re
from pathlib import Path
sys.path.insert(0, "/home/egaillac/MetaP/classifier/scripts")
import xenc_format as X


def encode(s1, s2, text, order="canon", names_only=False, mask_relation=None, anon=False):
    """-> (segment_a, segment_b, spanA, spanB, taxonA, taxonB, ok)

    order='canon'  : the two taxa sorted alphabetically (deployment behaviour)
    order='swap'   : reverse-alphabetical (training-time marker-swap augmentation)
    names_only     : replace the passage by "A and B ." -- the passage-blind control
    mask_relation  : string to blank out of the passage before marking (ablation)
    """
    s1, s2, text = str(s1), str(s2), str(text)
    a, b = sorted([s1, s2], key=str.lower)
    if order == "swap":
        a, b = b, a
    if mask_relation:
        for part in str(mask_relation).split("|"):
            part = part.strip()
            if part:
                text = re.sub(re.escape(part), "[MASK]", text, flags=re.I)
    if names_only:
        text = f"{a} and {b} ."
    sa = "taxon [SEP] taxon" if anon else f"{a} [SEP] {b}"
    pa, pb = X._locate(a, text), X._locate(b, text)
    if pa and pb and not (pb[1] <= pa[0] or pb[0] >= pa[1]):
        pb = None                                   # overlapping mentions: drop the second
    spans = []
    if pa:
        spans.append((pa[0], pa[1], "A"))
    if pb:
        spans.append((pb[0], pb[1], "B"))
    out, prev, pos = [], 0, {}
    for st, en, tag in sorted(spans):
        out.append(text[prev:st])
        m = "@" if tag == "A" else "#"
        start = sum(len(x) for x in out)
        body = "taxon" if anon else text[st:en]
        out.append(f"{m} {body} {m}")
        pos[tag] = (start, start + len(f"{m} {body} {m}"))
        prev = en
    out.append(text[prev:])
    sb = "".join(out)
    return sa, sb, pos.get("A"), pos.get("B"), a, b, ("A" in pos and "B" in pos)
