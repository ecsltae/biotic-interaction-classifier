# Is the two-head design novel? — assessment 2026-09-27

Three independent literature surveys (relation extraction and argument order; antisymmetry as an
architectural constraint; multi-task biomedical IE), then a skeptic and an advocate ruling on the
result. Both verdicts landed on **novel combination, not novel contribution**.

## What I verified myself

| claim | status |
|---|---|
| Lai, Wei, Tian, Leaman & Lu (2025), *Enhancing Biomedical Relation Extraction with Directionality*, arXiv:2501.14079, Bioinformatics 41(Suppl_1) ISMB/ECCB | **verified** — title, authors, venue, and the 10,864 directionality annotations all confirmed |
| Keck, Broadbent & Altermatt (2025), GPT-4o over 83,910 articles extracting pairwise species interactions, *Ecology and Evolution* | **verified** — bioRxiv 10.1101/2025.01.24.634685 |
| Lai et al.'s head is a 4-way softmax and their input is order-*sensitive* | **not verified** — the README and abstract do not state it; relayed from the survey. The remaining delta depends on this, so read the paper before relying on it. |

## What is not new — cite it, do not claim it

**The two-head factorization, in this exact domain, one year ago.** Lai et al. 2025 put separate
relation / novelty / directionality heads on one shared PubMedBERT encoder, with 10,864 human
directionality annotations. All three surveys independently flagged this as desk-reject-grade if
"two heads, one classifying and one asserting direction" is the headline. It belongs in the
**introduction**, not the related-work tail.

**Factoring type from direction is sixteen years old.** Rink & Harabagiu (SemEval-2010, the winning
Task 8 system) used one classifier for relation type and a second, per-type, for direction.

**Antisymmetry-by-construction is textbook.** It is the antisymmetrisation operator for the sign
representation of S₂. In ML: RankNet (ICML 2005) already gives P(a≻b) = 1 − P(b≻a) identically;
SortNet / CmpNN (Rigutini et al., ICANN 2008; IEEE TNN 2011) enforces N(x,y) = −N(y,x) exactly by
dual-neuron weight sharing **and proves the constrained network is still a universal
approximator** — a stronger result than our difference form.

**"First to do direction in species interactions" is false.** Keck et al. 2025 extracted 144,402
directional pairwise interactions from 83,910 articles with GPT-4o.

**The `@`/`#` markers are Zhou & Chen's (2021)**, with the semantics inverted: theirs denote
subject/object, ours denote alphabetical position. Every RE reviewer will read them the standard
way. Flag the inversion in the first sentence of the method or the invariance claim gets skimmed.

**The invariance/equivariance vocabulary is borrowed.** LOGIN (Woo et al., TNNLS 2022) already
writes "equivariant under the subject-object ordering while being invariant to the permutation".

## What does appear to survive

1. **An order-invariant-by-construction encoder input for relation extraction.** No survey found an
   RE paper that sorts the arguments canonically so the two orders give a byte-identical input. The
   field does the opposite: direction goes in the label space, or into role markers, or (LOGIN)
   into a constrained permutation set. This is the strongest element and currently the most
   undersold.
2. **The relation admitted only as a 2-valued agent/patient polarity embedding at the head.** This
   is what makes the invariance airtight — the encoder cannot depend on the relation, so direction
   cannot be smuggled back in through the relation string — and what lets one 594k-parameter head
   serve every relation in the ontology rather than a per-relation head set.
3. **The conjunction of the two guarantees.** Existence is provably order-invariant *and* direction
   is exactly antisymmetric, so the two heads are structurally incapable of contradicting each
   other. Lai et al. have neither guarantee.

## How to frame it

**Defensible:** a symmetry-structured formulation of triple verification, in which argument-order
invariance of the existence decision and exact antisymmetry of the direction decision are
architectural guarantees rather than learned behaviours — demonstrated at deployment cost (110M,
CPU) on species-interaction extraction, with direction evaluated separately against human
annotation.

**Not defensible:** "we introduce a two-head architecture that jointly classifies interaction and
direction". That is Lai et al. 2025, and Rink & Harabagiu 2010 before them.

## What the claim needs

* Read Lai et al. properly and **run BioREDirect as a baseline**, or explain why it is not
  comparable. It is simultaneously the biggest threat and the best available baseline.
* An ablation isolating the invariance: the same model with role markers and no canonical sort,
  against the canonical-sort version. Without it the guarantees are asserted, not shown to matter.
* A swap-consistency metric on a standard RE benchmark, not only on 84 in-house items — the
  measurable consequence of the guarantee is that a swapped input cannot yield a contradictory
  answer, and an unconstrained head can.
* The direction gold is 84 items. It is enough to show the head reads the passage (names-only
  control drops to chance) and not enough to separate it from a good dependency rule.
