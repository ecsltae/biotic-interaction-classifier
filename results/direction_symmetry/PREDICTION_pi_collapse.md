# Prediction registered before evaluating the pi = 0.80 / 0.90 / 1.00 arms

Written 2026-09-27 18:56, with pi = 0.50, 0.60, 0.70 evaluated (0.989, 0.981, 0.963) and the
remaining arms still training.

## The refinement

pi alone does not fix the level of swap consistency. SemEval at pi = 0.70 keeps 0.963, whereas
the biodiversity corpus at pi = 0.708 collapses to 0.473. The difference is what the shortcut
is worth *relative to the best content solution available in that corpus*:

| corpus | shortcut value (max(pi, 1-pi)) | content ceiling (best dir acc) | gap | observed consistency |
|---|---|---|---|---|
| BioRED, pi=0.486        | 0.514 | 0.911 | 0.397 | 0.969 |
| SemEval, pi=0.544       | 0.544 | 0.966 | 0.422 | 0.938 |
| SemEval forced, pi=0.70 | 0.700 | 0.966 | 0.266 | 0.963 |
| Biodiversity, pi=0.708  | 0.708 | 0.869 | 0.161 | 0.473 |

In the biodiversity data the relation surface form adds almost nothing over position: a lookup
table from the relation form predicts the subject at 0.7152 against 0.7075 for answering
"first-listed" unconditionally. The shortcut is therefore close to competitive with everything
else on offer, and it wins. In SemEval the content solution is worth 0.966 against a 0.70
shortcut, so it does not.

## What this predicts for the arms now training

Consistency should stay high through pi = 0.80, where the shortcut is worth 0.80 against a
content ceiling of about 0.966, and then fall away sharply between 0.90 and 1.00 as the
shortcut approaches and then exceeds what reading the passage can buy. Concretely:

- pi = 0.80: consistency still above 0.90.
- pi = 0.90: a clear drop, somewhere in 0.5-0.9, and the first point where accuracy also
  starts to move.
- pi = 1.00: collapse. The shortcut is worth 1.000 on the training distribution and nothing
  else can beat it, so consistency should approach 0 and test accuracy should fall towards
  the test set's own positional prior rather than the 0.96 ceiling.

If instead consistency degrades linearly in pi, the "relative value of the shortcut" account
is wrong and the simpler "pi alone" story should be told.

---

# Outcome, recorded 2026-09-27 19:25 after evaluating all arms

| pi | predicted | observed swap | observed acc |
|---|---|---|---|
| 0.80 | consistency above 0.90 | **0.9449** | 0.9502 |
| 0.90 | clear drop into 0.5-0.9 | **0.8779** | 0.9272 |
| 1.00 | collapse; acc falls to the test prior | **0.0000** | **0.5845** |

**Held.** All three range predictions came in. The pi = 1.00 endpoint was sharper than
predicted: consistency is exactly 0.0000, not merely near it, and accuracy is 0.5845 against a
test-set positional prior of 0.5845 -- equal to four decimals, which is what a model answering
"first-listed" on every item must score. "Falls towards" understated it.

**Wrong in one detail.** The prediction said pi = 0.90 would be "the first point where accuracy
also starts to move". It is not: accuracy had already dropped at pi = 0.80, from 0.9582 to
0.9502. Accuracy begins eroding one step earlier than claimed. The qualitative point -- that
consistency degrades far faster than accuracy -- survives, and is if anything the more useful
form: between pi = 0.5 and pi = 0.9, consistency loses 0.111 while accuracy loses 0.034.

**The alternative hypothesis is rejected.** Consistency does not degrade linearly in pi. The
observed sequence is 0.989, 0.981, 0.963, 0.945, 0.878, 0.000: gentle while the shortcut is
worth less than the content ceiling of about 0.966, then total collapse once it is worth more.
That is the shape the "relative value of the shortcut" account predicts and not the shape the
"pi alone" account predicts.

**Control.** The canonical + symmetric arm trained on the same pi = 1.00 data is unaffected:
0.9650 accuracy and 1.0000 consistency, against 0.9646 and 1.0000 at pi = 0.50. The
manipulation that destroys the unconstrained arm does not touch it.

**Replication.** BioRED under the same manipulation: 0.9842, 0.9732, 0.8789 at pi = 0.50, 0.70,
0.90, tracking SemEval's 0.9889, 0.9628, 0.8779.
