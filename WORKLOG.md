# Classifier worklog

**Append-only.** Add new entries at the **bottom**. Never edit or delete an existing entry —
if something turns out to be wrong, say so in a new entry. Append-only keeps concurrent
sessions (multiple Claude windows, multiple machines) from clobbering each other.

Entry format: `## YYYY-MM-DD — <short title>` then what changed, what's in flight, what's blocked.

Keep it short. Durable conclusions belong in
`~/.claude/projects/-home-egaillac-MetaP/memory/`; this file is the running record of *what
happened when*, so a session starting cold can see what another one is doing.

---

## 2026-09-14 — BiotXplorer recall evaluation (window A)

### Context
Precision side was already done: `data/evaluation/biotx_retrieval_eval_100.csv`, Emilie's
curation of 100 kept triples (50 GloBI + 50 random). That gives precision only. This work
builds the recall counterpart.

### Located the negatives
No "rejected" collection exists in MongoDB. Rejects are a **set difference** in `sibils_eval`
on `sibils-mongodb.lan.text-analytics.ch:27017`:

| | collection | passages |
|---|---|---|
| ROBI baseline | `med25_r1_v_ep_robirules_corr_passages` | 171,596 |
| classifier kept | `med25_r1_v_ep_classifier_2026_testfull2_passages` | 63,633 |
| **rejected** | difference | **107,963** |

Reproduces the slide (`biotxplorer_slide.html`) exactly: −62.9% passages, −57.6% triplets.

**Trap:** `triplet_key` is `taxon1;taxon2;interaction` and taxon order is NOT stable between
runs. Diffing raw keys gives 137,068 bogus rejects + 29,105 phantom "kept but not in baseline".
Sort the two taxon IDs first — then kept ⊂ baseline exactly.

### Built (all in `data/evaluation/biotx_recall/`)
- `rejected_passages_full.csv` — all 107,963 negatives
- `recall_curation_100.csv` — 100 uniform-random rejects, 5-column triple-level rubric
- `sentence_level_blind_200.csv` (+ `_KEY`) — 100 kept + 100 rejected, shuffled, triple hidden,
  one binary column
- `score_recall.py`, `score_sentence_level.py` — both smoke-tested
- `README.md` — rubric, worked examples, conventions
- `convention_questions_emilie.md` — borderline cases + French email draft

### Key methodological finding
Two evaluations exist and **must not be mixed**:
- *sentence-level* ("does this sentence describe an interaction?") — measures the classifier,
  which only ever sees the sentence
- *triple-level* ("is this extracted triple correct?") — measures BiotXplorer end-to-end

Pairing triple-level precision (0.48) with a sentence-level reject rate (0.55) gives a bogus
recall of 0.34. Consistent pairs land near 0.45–0.50.

### Opus 5 zero-shot baseline (provisional)
Run at effort max on the 100 rejects, bare sentence, no examples: 55/100 YES, 24 uncertain.
But it answered the sentence-level question. Re-scored at triple level only ~28/100 hold
(16 wrong pair, 6 non-taxon slot, 5 wrong type). Errors are systematically **pair-specificity**
failures — same root cause as `project_ep_error_analysis`. Re-running with entity-prepended
prompts should close most of the gap.

### Upstream (non-classifier) failure modes found
Frequencies are a result about BiotXplorer's entity layer, not the model:

| mode | example |
|---|---|
| coordination artifact | R007, R010, R017, R019 — co-listed taxa read as a pair. **4 of first 15 — dominant.** |
| homonym | R004 — "cerium" the element matched genus *Cerium* |
| vernacular gap | R010 — "palm trees" absent from Open Tree |
| abbreviation gap | R008 — "Cm"/"Ss" defined outside passage window |
| compound-adjective fragment | R009 — "human" taken from "human-made bridges" |

### In flight / blocked
- **Annotation in progress.** 15 of 100 rows scored, 2 true positives (R006 ticks×rabbits,
  R022 wasps×spiders). Both are *nominalised* interactions ("infestation", "prey capture")
  rather than verbal — possible classifier blind spot, too early to tell.
- **BLOCKED on Emilie:** convention for "human pathogens" as a passing descriptor (R003, R011).
  Current working call is `ok_species=0` (correct rejection). See
  `convention_questions_emilie.md`. Also open: is synanthropy an interaction (R014)?
- **User observation (2026-09-14):** reports most of their false negatives come from
  "human pathogens" phrasing, and wants the model trained to recognise those.
  ⚠ **Sign is unresolved:** under the current convention those rows score 0 = *correct*
  rejections, not false negatives. If Emilie rules the other way they flip to FN and become a
  training target; if not, training on them would teach the model to make errors. Do not act on
  this until the convention is settled.

---

## 2026-09-14 — convention resolved: descriptive "human pathogens" IS an interaction (window A)

**Corrects the previous entry.** Emilie ruled on R003 (*Salmonella* / "as in the case of other
human pathogens ... surveillance of Salmonella infections"): there **is** an interaction. Scores
`1;1;1;1;1`, not `1;1;1;0;0`.

Consequences:
- The assertion rule is revised. A **descriptive or category mention counts** — the interaction
  need not be the sentence's topic, nor phrased as an event. What still fails is a relation
  asserted about a *different* pair (her *M. marinum* → fish, not → *M. tuberculosis*).
- R011 (Viruses / "contain a number of important human pathogens") follows → `1`. Worth a
  one-line confirmation from Emilie since it is the weaker case (subordinate clause in a
  quasicrystal-modelling paper).
- `biotx_recall/README.md` assertion-rule section rewritten accordingly.

### Sign resolved on the user's observation
The previous entry flagged the "human pathogens" cluster as possibly *correct rejections*. With
Emilie's ruling they are **genuine false negatives** — the user was right. This is now a valid
training target.

Cluster size in the 100-row reject sample: **5 rows** with explicit "human pathogen(s)" phrasing
(R003, R011, R016, R033, R074); **17 rows** mention "pathogen" at all.

### Stronger signal: nominalised interactions
All three true positives found so far express the interaction as a **noun, not a finite verb**:

| row | phrasing | pair |
|---|---|---|
| R006 | "rabbits **infested** with **ticks**" / "during **infestation**" | Ixodida × *Oryctolagus* |
| R022 | "utilized for **prey capture**" | Vespidae × Araneae |
| R030 | "pack-hunting **predator**", "its **prey**" | *Speothos* × Dasypodidae |

Hypothesis: the classifier is tuned to verbal interaction phrasing (*preys on*, *feeds on*,
*parasitizes*) and under-fires on nominalisations (*predator*, *prey capture*, *infestation*).
3/3 is suggestive, not evidence — recheck once more of the 100 are scored. If it holds it is a
cleaner and more actionable training target than the pathogen cluster, because it is a
*linguistic* pattern that can be augmented synthetically.

### Scored so far (18/100), 3 true positives
R003 and R011 rescored per the ruling above.

---

## 2026-09-17 — recall annotation complete (n=50), first real numbers (window A)

Sheet returned as `classifier/recall_curation_100.xlsx`, sheet **`curation`** (the first sheet is
the unsplit import — ignore it). Rows R001–R050 annotated; R051–R100 deliberately not done.

### Result (triple-level rubric, rejects only)

| | p_rejected | recall | 95% CI |
|---|---|---|---|
| full-triple (`triples_ok_full`) | 0.22 | **0.56** | 0.45–0.69 |
| species-level (`triples_ok_species`) | 0.26 | **0.60** | 0.50–0.71 |

With precision 0.48 (random stratum of Emilie's kept sheet), F1 ≈ 0.52 at triple level.
Recall is relative to the ROBI candidate set — an upper bound.

### METHODOLOGICAL WARNING — do not repeat this mistake
During the row-by-row walkthrough the running estimate sat at 0.34. That was **biased upward**:
the 32 rows discussed were chosen by the user as interesting/ambiguous, not at random.

| | n | full-triple TP |
|---|---|---|
| rows walked through together | 32 | 10 |
| rows never discussed | 18 | **0** |

Never quote a rate computed over a conveniently-selected subset as if it estimated the population.

### Conventions settled
- **Cas 3 — "effect of X on Y was studied" → NEGATIVE.** R031, R048 scored `ok_species=0`.
  An aims/methods framing does not assert the relation. (Claude had leaned positive; overruled.)
- **Cas 4 — co-infection in a shared host → POSITIVE at species level.** R046 = `1;1;1;1;0`.

### Data issues outstanding in the sheet
- **R022** = `1;1;1;0;1` — nesting violation (`ok_full=1` with `ok_species=0`). Likely `ok_species`
  should be 1 (wasps capture spiders, explicit).
- **R044** — `interaction_eval`, `ok_species`, `ok_full` blank.
Numbers above assume R022 `ok_species=1` and R044 = `1;1;1;1;1`.

### Still not done
`sentence_level_blind_200.csv` — the blinded sentence-level sheet (100 kept + 100 rejected) has
not been annotated. That is the one that measures the **classifier** on its own task; the numbers
above measure BiotXplorer end-to-end.

---

## 2026-09-17 — CORRECTION: recall is 0.62, not 0.56 (unit mismatch) (window A)

The previous entry computed recall at **passage** level (kept 63,633 / rejected 107,963). Wrong
unit: Emilie's random 50 were sampled as **triplets** (`biotx-random-triplets` in her sheet's
`file name` column), so precision 0.48 is a per-triplet rate. Mixing a per-triplet precision with
a per-passage FN rate is invalid.

Correct, unit-matched:

```
kept triplets      43,708   x 0.48 precision  =  TP 20,980
rejected triplets  59,414   x 0.22 FN rate    =  FN 13,071
recall = 20,980 / (20,980 + 13,071) = 0.616
```

| | value | 95% CI |
|---|---|---|
| recall | **0.62** | 0.50–0.73 |
| precision | 0.48 | — |
| F1 | **0.54** | — |

Sampling check: all 50 annotated rejects are **distinct triplet_keys**, so the passage-drawn
sample yields an unbiased triplet-level rate — no reweighting needed. Had triplets repeated, the
passage sample would over-weight high-frequency ones.

User corrections applied: R044 = all 1; R022 left as submitted (`1;1;1;0;1`).

**Use 0.62.** Still BiotXplorer end-to-end, still relative to the ROBI candidate set (upper bound).
