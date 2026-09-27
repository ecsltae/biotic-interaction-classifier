# Reproducing the argument-order symmetry results

Every number in the paper comes from this directory. One 80 GB A100; each training arm is
about two minutes, so the full set is roughly an hour.

    source /home/egaillac/MetaP/MPvenv/bin/activate

## 1. Data

    python3 convert_biored.py      # -> biored_{train,dev,test}_ours.csv
    python3 convert_semeval.py     # -> semeval_{train,test}_ours.csv

Both emit `s1, s2, rel, passage, subject_is, directed`. `subject_is` is 1 when the
first-listed argument is the subject, 2 when the second is, and -1 when the pair is
undirected. Pairs whose two mentions have identical surface strings are dropped, because the
swap is then a no-op and the item cannot test anything.

Corpus sizes after conversion: BioRED 4,164 train / 1,163 test (1,045 directed, 118
undirected); SemEval 7,991 train / 2,713 test (2,260 directed, 453 *Other*).

## 2. The four design cells

`train_direction_general.py` takes `--input canonical|text` and `--head symmetric|uncon`,
which is the 2x2. Everything else -- encoder, optimiser, schedule, epochs, data -- is shared,
so the only difference between two arms is the axis under test.

    for inp in canonical text; do for hd in symmetric uncon; do for s in 1 2 3; do
      python3 train_direction_general.py --train semeval_train_ours.csv \
        --out semeval_${inp}_${hd}_s$s --input $inp --head $hd --epochs 4 --seed $s
    done; done; done

    python3 semeval_2x2.py          # -> SEMEVAL_2X2.json, the ablation table
    python3 all_seeds_table.py      # -> per_seed_semeval.csv, the appendix table

## 3. BioRED arms

    for s in 1 2 3; do
      python3 train_ours_on_biored.py   --out ours_on_biored_s$s   --seed $s   # canonical + sym
      python3 train_biored_ordersens.py --out biored_ordersens_s$s --seed $s   # text + uncon
    done
    python3 eval_ours_on_biored.py ours_on_biored_s1
    python3 eval_ordersens.py biored_ordersens_s1

`biored_ordersens_*` is the matched baseline: same encoder, budget, data and seeds as our arm,
differing only in the input format and the head. It is the comparison the accuracy claim rests
on -- **not** the released BioREDirect weights, which perform a different, joint task.

## 4. Significance

    python3 mcnemar_arms.py semeval_test_ours.csv \
        semeval_canonical_symmetric_s semeval_text_uncon_s
    python3 mcnemar_biored.py

Each arm is reduced to the majority vote of its three seeds before the test, so the comparison
is between arms rather than between two lucky initialisations. McNemar with continuity
correction.

## 5. The pi intervention

`make_pi.py` rewrites a training set to a target pi by changing only the order in which the two
arguments are presented. The passage, the pair, the relation and the gold subject are
untouched, and the script asserts that the pair multiset, the passages and the directedness
flags are unchanged before writing. The test set is never modified.

    for pi in 0.50 0.60 0.70 0.80 0.90 1.00; do
      python3 make_pi.py --pi $pi --out semeval_train_pi${pi/./}.csv
    done
    bash /tmp/pi_sweep.sh
    python3 pi_sweep_eval.py        # -> PI_SWEEP.json

This is what turns the pi claim from a correlation over three corpora into a controlled
result: the same corpus at several priors, everything else held fixed.

## 6. Gotchas that cost time here

- **Decoding must match the arm.** A canonical arm decodes through the alphabetical order of
  the pair; an order-sensitive arm does not. Applying canonical decoding to an order-sensitive
  model reads 0.468 accuracy and 0.019 consistency instead of 0.912 and 0.969. If a baseline
  looks impossibly bad, check the decode before believing it.
- **The swap must exchange every slot that encodes order**, not just the role markers. Marking
  differently without reordering the question changes the gold answer rather than the input,
  and produced a spurious 0.175 where the true value is 0.700.
- **Undirectedness AUC has large seed variance** (sigma ~ 0.02-0.03). Seed 1 alone overstates
  the head effect by about a factor of two. Use three seeds.
- **`pgrep -f` / `pkill -f` match the waiting shell itself.** A waiter loop built that way
  waited on its own process for seven hours. Use explicit PIDs.

## 7. Result files

| file | contents |
|---|---|
| `RESULTS_CONSOLIDATED.json` | every arm, mean and sd over seeds |
| `SEMEVAL_2X2.json` | the design ablation |
| `per_seed_semeval.csv` | per-seed values behind the appendix table |
| `PI_SWEEP.json` | the controlled pi dose-response |
