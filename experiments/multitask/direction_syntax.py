"""Syntactic argument-role labeller for direction supervision.

Given a passage, two taxon surface forms and a relation surface form, decide which
taxon fills the SUBJECT-side argument slot of the relation predicate, using a real
dependency parse (spaCy en_core_web_sm).

Why this source and not the obvious ones:
  * stored order is a coin flip (measured 9/17 on human gold),
  * text order is a deterministic function of the pipeline's own char-offset sort,
  * a GloBI pair-join labels 0.39% of the pool and is a taxon-name lookup,
  * the "rule" label used previously is exactly text_order x voice(relation_form),
    i.e. a deterministic function of two features the cross-encoder already sees,
    so it carries zero new bits.
This labeller reads the SENTENCE. Its output is not recoverable from the taxon
names or from mention order, so a head trained on it must read the passage.

NOTE on spaCy: Token objects are not cached, so `tok_a is tok_b` is always False
even for the same token. Every comparison here is on `.i`.
"""
import re, spacy

_NLP = None
def nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP

SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
OBJ_DEPS = {"dobj", "obj", "dative", "oprd", "attr", "pobj"}
# prepositions that mark the OBJECT-side argument of a nominal/verbal relation
OBJ_PREPS = {"of", "on", "by", "with", "to", "for", "in", "from", "upon", "against", "into"}


def locate(sp, txt):
    """Character spans of taxon/relation surface form `sp` in `txt`."""
    sp = str(sp).strip()
    if not sp:
        return []
    cands = [sp]
    toks = sp.split()
    if len(toks) > 1:
        cands += [toks[0], toks[-1]]
    cands += [sp + "s", sp.rstrip("s")]
    for c in cands:
        if not c or len(c) < 3:
            continue
        hits = [m.span() for m in re.finditer(r"\b" + re.escape(c) + r"\b", txt, re.I)]
        if hits:
            return hits
    return []


def head_token(doc, span):
    s = doc.char_span(span[0], span[1], alignment_mode="expand")
    return None if (s is None or len(s) == 0) else s.root


def rel_tokens(doc, rel):
    """Head tokens of every mention of the relation surface form."""
    outs, seen = [], set()
    for alt in str(rel).lower().split("|"):
        alt = alt.strip()
        if not alt:
            continue
        keys = [alt]
        stem = re.sub(r"(ions|ion|ed|ing|s)$", "", alt)
        if len(stem) >= 4:
            keys.append(stem)
        for k in keys:
            for sp in locate(k, doc.text):
                t = head_token(doc, sp)
                if t is not None and t.i not in seen:
                    seen.add(t.i); outs.append(t)
            if outs:
                break
    return outs


def _chain(tok, maxup=6):
    """[tok, parent, grandparent, ...] nearest first; stops at ROOT."""
    out, t, n = [tok], tok, 0
    while t.head.i != t.i and n < maxup:
        t = t.head; out.append(t); n += 1
    return out


def role_of(tax_tok, rel_tok):
    """'SUBJ' / 'OBJ' / None -- which argument slot of rel_tok does tax_tok fill?"""
    if tax_tok is None or rel_tok is None:
        return None, ""
    ch = _chain(tax_tok)
    for i, t in enumerate(ch):
        if t.i == rel_tok.i:
            if i == 0:
                return None, "self"
            prev = ch[i - 1]
            if prev.dep_ in SUBJ_DEPS:
                return "SUBJ", f"d:{prev.dep_}"
            if prev.dep_ == "agent":
                return "OBJ", "d:agent"
            if prev.dep_ == "prep":
                p = prev.text.lower()
                return ("OBJ", f"p:{p}") if p in OBJ_PREPS else (None, f"p?:{p}")
            if prev.dep_ in OBJ_DEPS:
                return "OBJ", f"d:{prev.dep_}"
            # "the pathogen Y" / "its prey Bosmina" -- Y IS the relation nominal
            if prev.dep_ in ("appos", "compound", "nmod", "conj"):
                return "SUBJ", f"d:{prev.dep_}"
            if prev.dep_ == "poss":
                return "SUBJ", "d:poss"      # "its prey" -> possessor is the predator
            return None, f"d?:{prev.dep_}"
        # the taxon (or an ancestor of it) GOVERNS the relation
        if rel_tok.head.i == t.i:
            d = rel_tok.dep_
            if d in ("acl", "advcl", "relcl", "partmod", "amod"):
                return "SUBJ", f"g:{d}"      # "Meloidogyne infecting Ixora"
            if d == "appos":
                return "SUBJ", "g:appos"     # "Y, a pathogen of X"
            if d in ("attr", "acomp", "oprd", "dobj", "obj"):
                # copular / object predicate: the SUBJ is the copula's nsubj
                for c in t.children:
                    if c.dep_ in SUBJ_DEPS:
                        sub = {x.i for x in c.subtree}
                        if tax_tok.i in sub:
                            return "SUBJ", f"cop:{d}"
                return None, f"g?:{d}"
            if d == "pobj":
                return None, "g?:pobj"
    # copular predicate nominal reached from the other side:
    # "X is a pathogen of Y" with tax=X, rel=pathogen
    if rel_tok.dep_ in ("attr", "acomp", "oprd"):
        for c in rel_tok.head.children:
            if c.dep_ in SUBJ_DEPS and tax_tok.i in {x.i for x in c.subtree}:
                return "SUBJ", "cop:nsubj"
    return None, ""


def syn_subject_side(passage, t1, t2, rel, doc=None):
    """-> (idx, evidence): idx 0 = t1 is the subject-side argument, 1 = t2, None = undecided.

    'Subject-side' is SURFACE SYNTAX of the relation phrase as written. Compose it
    with direction_lexicon.polarity() to get the annotation's SUBJECT.
    """
    doc = doc if doc is not None else nlp()(str(passage))
    rts = rel_tokens(doc, rel)
    if not rts:
        return None, "no-rel"
    s1, s2 = locate(t1, doc.text), locate(t2, doc.text)
    if not s1:
        return None, "no-taxon1"
    if not s2:
        return None, "no-taxon2"
    partial = None
    for rt in rts:
        h1 = min((head_token(doc, s) for s in s1 if head_token(doc, s)),
                 key=lambda t: abs(t.i - rt.i), default=None)
        h2 = min((head_token(doc, s) for s in s2 if head_token(doc, s)),
                 key=lambda t: abs(t.i - rt.i), default=None)
        if h1 is None or h2 is None or h1.i == h2.i:
            continue
        r1, e1 = role_of(h1, rt)
        r2, e2 = role_of(h2, rt)
        if r1 == "SUBJ" and r2 == "OBJ":
            return 0, f"BOTH:{e1}/{e2}"
        if r2 == "SUBJ" and r1 == "OBJ":
            return 1, f"BOTH:{e2}/{e1}"
        if partial is None:
            if r1 == "SUBJ" and r2 is None:   partial = (0, f"ONE:{e1}")
            elif r2 == "SUBJ" and r1 is None: partial = (1, f"ONE:{e2}")
            elif r1 == "OBJ" and r2 is None:  partial = (1, f"ONEi:{e1}")
            elif r2 == "OBJ" and r1 is None:  partial = (0, f"ONEi:{e2}")
    return partial if partial else (None, "no-role")
