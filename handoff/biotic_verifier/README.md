# Biotic interaction verifier: handoff

Scores candidate `(taxon, relation, taxon)` triples against the passage they were extracted
from, and answers two questions:

* **does this pair interact**, according to this passage?
* **which taxon is the subject** of the relation?

It runs on CPU. It replaces the sentence-level interaction classifier in the filtering step.

---

## 1. Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the torch/transformers pair the weights were written with. Nothing else
is needed: the package is self-contained and makes **no network calls** at inference time.

Check it works:

```bash
python predict.py --in example_input.csv --out /tmp/check.csv
```

You should see `3 candidates, 3 accepted`.

## 2. Run it

**A file of candidates** (the normal case):

```bash
python predict.py --in candidates.csv --out scored.csv
```

`candidates.csv` needs exactly these four columns (any others you add are copied through to the
output untouched, so you can keep your own ids):

| column | meaning |
|---|---|
| `species1` | first taxon, **as it appears in the passage** (surface form, not the canonical name) |
| `relation` | interaction surface form |
| `species2` | second taxon, same rule |
| `sentence` | the passage the candidate was extracted from |

**One candidate**, for spot checks:

```bash
python predict.py --s1 "Haemophilus influenzae" --rel "pathogen of" --s2 human \
                  --text "Haemophilus influenzae is a major pathogen of humans."
```

Useful flags: `--threshold 0.95` (see §4), `--threads 16`, `--batch-size 32`.

## 3. Read the output

| column | meaning |
|---|---|
| `interacts` | **0/1: the decision.** This is the column that replaces the old classifier's verdict |
| `p_interact` | the model's score, 0–1, before the candidate rules. Use this if you want your own cutoff |
| `rejected_by_rule` | empty, or the name of the candidate rule that rejected the row (§5); a rejected row has `interacts = 0` whatever its score |
| `direction` | one of four: `FORWARD` (species1 acts on species2), `REVERSE` (species2 acts on species1), `BIDIRECTIONAL` (the relation is mutual, such as mutualism, symbiosis, *interacts with* or *co-occurs with*, so neither taxon is the subject), `UNCERTAIN` (the model is not confident, a taxon was not found, or the pair does not interact) |
| `p_species1_is_subject` | 0–1; `direction` is this thresholded with an abstention band. Empty for `BIDIRECTIONAL` rows |
| `direction_confidence` | `abs(p − 0.5) × 2`. Below 0.60 a directed relation is reported as `UNCERTAIN`. Empty for `BIDIRECTIONAL` rows |
| `symmetric_relation` | 1 if the polarity lexicon calls the relation mutual; these rows are `BIDIRECTIONAL` |
| `both_taxa_located` | 0 if a taxon string was not found in the passage (**see §5**) |
| `unknown_polarity` | 1 if the relation is not in the polarity lexicon (§5) |
| `truncated` | 1 if the passage was longer than the model's 256-wordpiece window and its end was cut |

`FORWARD`/`REVERSE` are judged against the **canonical** relation, not the surface string. So
`Leptodora --prey--> Bosmina` comes back `REVERSE`, because the canonical relation is
*preyed upon by* and Leptodora is the predator. That is the convention the gold annotation uses.

**Order does not matter.** Give the same pair the other way round and `p_interact` is bit-for-bit
identical while `FORWARD` and `REVERSE` swap (`BIDIRECTIONAL` and `UNCERTAIN` stay put). That is a
property of the architecture, not of training, so it cannot drift.

## 4. Which threshold to use

The default is **0.50**, fixed before evaluation rather than tuned. Every benchmark number in
this README uses the revised benchmark labels (251 of 437 positive; revision of 2026-10-06).
Measured on a 437-item expert-graded evaluation set, with the candidate rules on (the default):

| `--threshold` | precision | recall | false pos | false neg | |
|---|---|---|---|---|---|
| 0.50 | 0.888 | 0.948 | 30 | 13 | **default**, catches nearly everything |
| 0.70 | 0.890 | 0.932 | 29 | 17 | |
| 0.90 | 0.907 | 0.892 | 23 | 27 | |
| 0.95 | 0.920 | 0.869 | 19 | 33 | best balance if curation time is scarce |
| 0.99 | 0.969 | 0.745 | 6 | 64 | only when a false positive is expensive |
| *the filter this replaces* | *0.863* | *0.781* | *31* | *55* | |

At the default the model is **more precise than the old filter (0.888 against 0.863) with 17
points more recall**. Raise the threshold to buy precision; the table is the exact trade. Without
the rules (`--no-rules`) the default gives precision 0.873, recall 0.960.

### Optional: a second opinion from a local LLM on the uncertain candidates

If you have a GPU, `escalate.py` keeps the verifier's decision on every candidate except the
fraction whose score lies nearest the threshold, which a local LLM re-decides by answering one
question: does the passage assert an interaction between *these two* taxa? Rule rejections stay
final. Nothing leaves the machine (Ollama).

```bash
python predict.py  --in candidates.csv --out scored.csv
python escalate.py --in candidates.csv --scored scored.csv --out final.csv --band 0.3
```

Same 437-item evaluation, default threshold, rules on, LLM `qwen3.8:27b` (needs Ollama >= 0.35):

| `--band` | LLM calls | precision | recall | false pos |
|---|---|---|---|---|
| 0 (verifier alone) | 0% | 0.888 | 0.948 | 30 |
| 0.1 | 10% | 0.910 | 0.928 | 23 |
| 0.2 | 20% | 0.931 | 0.908 | 17 |
| **0.3** | 30% | **0.961** | 0.888 | 9 |
| 0.5 | 50% | 0.976 | 0.821 | 5 |

For the same precision the verifier alone needs `--threshold 0.99` and keeps recall 0.745, so the
LLM buys about 14 points of recall at that precision; the column to read is `interacts_final`.
On an older Ollama, `--model qwen3:32b` gives 0.943 / 0.916 at band 0.3. Cost: one LLM call per
escalated candidate, about 1--2 per second on one A100 for a 27B model.

## 5. When to distrust it

* **`both_taxa_located = 0`.** The model marks the two taxa in the passage so it knows which pair
  it is being asked about. If a taxon string is not found, that marking failed and the answer is
  much weaker. Treat those rows as unreviewed. It fires on 4 of the 437 evaluation rows.

  An earlier copy of this package matched taxon strings with no word boundaries, so a genus could
  match inside an unrelated one (`Aedes` marked the fragment `Aede` in *Aedeomyia*) and the flag
  still reported success; alternative forms joined with `|` (`African buffalo|Syncerus caffer`)
  were never matched in full, so one word of one alternative was marked instead (`@African@
  elephant`). Both are fixed. If you scored data with an earlier copy, rows whose taxa are
  substrings of other words, or that carry `|`-joined forms, are worth re-scoring.
* **`truncated = 1`.** Only the first 256 wordpieces of a passage are read; a taxon or the
  interaction stated after the cut is invisible. One of the 437 evaluation rows is affected.
  Whole abstracts often exceed the window: split them into sentences first.
* **Candidate rules** (`candidate_rules.py`) reject, before the model's verdict, candidates that
  cannot be an interaction between two distinct organisms: the same organism named twice
  (*sulla (Hedysarum coronarium)*), a taxon and its own clade, a taxonomic author parsed as a
  taxon (*Pterostichus melanarius (Illiger)*), a relation term that is not biotic (*relative to*,
  *hybridization*), an explicit negation of the pair (*found only on non-wheat hosts*), an organ
  or syndrome parsed as a taxon, and an adjective inside a pathogen's name (*equine* influenza
  virus). Each rule was checked on 42,000 training rows before use: what it rejects is a teacher
  negative 88–100% of the time. On the evaluation set they remove 5 false positives and 3 true
  ones. `rejected_by_rule` says which rule fired; `--no-rules` turns them off.
* **Co-occurrence in a shared host.** The clearest residual error class. When two organisms are
  both mentioned in relation to a *third* one, the model can read that as an interaction between
  them. `example_input.csv` row 3 is a real case: *Acanthamoeba* and *Pseudomonas* both infect
  the horse, not each other, and the model accepts it at p=0.88. It is a confirmed false positive.
* **Entity errors are not fixed here.** Of the errors the old filter made, about half are
  pair-binding (this model addresses them) and about half are wrong taxon resolution upstream
  (it does not). If the rule layer hands it the wrong species, it will happily verify the wrong
  species.
* **Direction is measured on the 97 human-graded direction items** (84 with a direction, 12
  marked undecidable, 1 bidirectional): on the 84, it answers 74% and is right on **0.887** of
  those; of the 12 the curator could not decide, it also abstains on 9; the bidirectional one comes
  back `BIDIRECTIONAL`. A control that hides the passage and shows only the two taxon names scores
  0.464 (chance), so the head is reading the sentence, not recalling which organisms usually
  parasitise which. `BIDIRECTIONAL` comes from the relation lexicon, not from the model: it is only
  as complete as the lexicon's list of mutual relations.
  It is weakest on bare relational nouns (*host*, *pathogen*, *infection*: 0.725) and strongest
  where the relation word itself carries direction (*pathogen of*, *ectoparasite of*: 0.909).
* **Relations outside the polarity lexicon** get the patient-side polarity the head was trained
  with for them, and `unknown_polarity = 1`. The polarity input turned out to matter little to the
  trained head: flipping it on every held-out row changes 5 predictions in about 4,700, so an
  unknown relation weakens the direction answer less than one might fear. A mutual relation the
  lexicon does not list is not reported `BIDIRECTIONAL`; competition and co-infection are listed.

## 6. Speed

~**32 candidates/s** on 8 CPU threads on real passages (≈115,000/hour; measured through
`predict.py` on the 437 evaluation rows, rules on). Short sentences run
faster (the example file benches near 95/s), so plan with the lower figure. `--threads` scales
roughly linearly to the core count. Memory: about 1.5 GB resident.

The direction output is **free**: the same model at the same speed. The head is 594,465
parameters, +0.54% on top of the encoder.

## 7. What is in this folder

```
predict.py            what you run
escalate.py           optional: a local LLM re-decides the least confident candidates (section 4)
candidate_rules.py    deterministic rules that reject impossible candidates before scoring (section 5)
joint_model.py        the architecture (encoder + antisymmetric direction head)
xenc_format.py        builds the encoder input; do not edit, training used this exact code
polarity.py           relation-polarity lexicon feeding the direction head
robi_maps.json        }
data/robiext_v2025.json } the ROBI ontology those lexicons are built from
model/                the weights, 421 MB
example_input.csv     three candidates to check the install
MODEL_CARD.md         full evaluation, statistics and caveats (section 8)
requirements.txt      pinned torch/transformers
```

Total 422 MB. `model/` holds the weights: **do not commit it to git.**

## 8. More detail

`MODEL_CARD.md` in this folder: the full evaluation, statistics and caveats behind the
numbers above.
