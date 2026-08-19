All findings verified. Writing up the adversarial verification.

## Dimension: sibling-transfer

*Verification pass: every cited file was opened; every quoted number re-derived independently; the champion checkpoint and the KG builder were executed read-only to test mechanism claims. Nothing was modified.*

---

### CONFIRMED

#### 1. The best taxon resolver in MetaP sits unused inside the classifier, while the deployed pipeline calls one that cannot return an identifier

**Evidence (all re-derived).** `classifier/src/data/ott_resolver.py:149` `class OTTResolver`; cascade `resolve` at `:183-192` → `_exact:201` → `_abbrev:213` → `_fuzzy:231`; kingdom parent-climb `_kingdom_of:270-284`; thread-local read-only connections `:164-179`. Index cardinalities re-counted directly against `classifier/data/processed/ott_index.sqlite` (1,071,099,904 B): **4,529,129 concepts / 9,316,799 names / 7,579,845 distinct `name_norm` / 4,787,671 synonym rows** — all four match the finder exactly.

Sole importer in the entire tree is `species_qa_service/extractor.py:35` (confirmed; `geonames_resolver.py:4` only names it in a docstring).

The deployed pipeline imports the other one at `classifier/api/fastapi_pipeline.py:62` and calls it at `:144`. `classifier/src/data/ott_lookup.py:104-134` returns:

```python
"ott_id": None,    # not yet resolved (needs full OTT TSV join)   # :132
"rank": None,      # not yet resolved                             # :133
```

So `ott_id` and `rank` in the port-8002 response are null by construction on every request. **Correction:** the finder wrote "ott_lookup.py:132-133" for the whole `lookup()`; the function spans `:104-134` and the two literals are at `:132` and `:133`.

**Additional:** the same defect is mirrored in `biotic-interaction-classifier-anon/api/fastapi_pipeline.py:62,134` — fixing only the classifier leaves the published mirror wrong.

**Change / integration point:** swap the import at `fastapi_pipeline.py:62` to `from data.ott_resolver import OTTResolver`, instantiate once at module level, replace `ott = ott_lookup(name)` at `:144` with `OTTResolver.resolve(name)`, populating `ott_id/canonical/rank/kingdom`. Retire `ott_lookup.py` and the 165 MB `species_dict.csv`.
**Impact:** high. **Effort:** hours.

---

#### 2. The entity-prepend experiment was fed a string absent from the sentence in 35% of rows

**Evidence.** `scripts/test_entity_prefix.py:71-72` reads `species1_term`/`species2_term`. `scripts/build_v16_dataset.py:32-38` prepends `source_species`/`target_species`. Re-derived on the 100-row EP-relax TSV:

| | both `*_term` in sentence | both `*_form` in sentence |
|---|---|---|
| all rows | **65/100** | **87/100** |
| gold=1 (n=48) | 31 | 38 |
| gold=0 (n=52) | 34 | 49 |

`interaction_form` present 96/100. All six figures match the finder exactly. Result file `results/error_analysis/ep_relax/entity_prefix_test.json` confirms baseline F1 0.8679 (TP46/FP12/FN2) vs bracket 0.8257 (FP16), qa_format 0.8182 (FP17), natural 0.7068 (FP38) — FP rises in all three.

**Correction the finder missed — the stored "best threshold" is fake.** `test_entity_prefix.py:91-96` computes `best_thr`, prints it at `:97`, then the JSON writer at `:122-124` hardcodes `{"threshold": 0.0, **res_best}`. The `0.0` in all three "best" blocks is a literal, not a measurement (`THRESHOLD = 0.13` at `:25`, sweep starts at 0.05). Anyone reading that artifact will conclude the optimal threshold is zero.

**Caveat on the recommendation:** switching to `*_form` adds alignment asymmetrically — +15 rows on gold=0 versus +7 on gold=1. The re-test is still the right call, but it is not label-neutral and must be scored, not assumed.

**Change:** re-run with `species1_form`/`species2_form`, plus a `Entity1: {canonical} ({form})` variant from `OTTResolver.resolve(term).canonical`; mirror the choice in `build_v16_dataset.py:32-38`; fix the JSON writer to emit the real `best_thr`.
**Impact:** high. **Effort:** hours (no training).

---

#### 3. qa_bert is a reusable QA harness; the eval files already carry 84% free span supervision; the checkpoint is Spanish

**Evidence — all citations exact.** `prepare_data.py:51-58` `_squad_qa`; `:108-147` `teacher_batch` with `start = text.find(span)` at `:143`; `:229-245` document-level split carrying the verbatim comment *"A per-example split would put the same document's text in both train and dev and make any dev metric optimistic."* `train_qa.py:33-47` `_cap_example`; `:55-71` `_downsample_negatives` with *"Long BOE docs make ~93% of windows no-answer"*; `:74+` `_prep_train`. `infer.py:46-80` `_best`, with the CLS seeded at `-inf` and the documented rationale at `:63-65`. `evaluate_qa.py:77-105` `_score_at`; `:141-145` threshold sweep; `:173-180` writes `best_tau`.

Free supervision re-derived: all three `*_form` strings co-present in **84/100** rows of *both* `globi-relax...EP.tsv` and `biotx_retrieval_eval_100.csv`; `interaction_form` alone present 96/100 (EP) and 98/100 (retrieval). Exact match to the finder.

Checkpoint confirmed Spanish: `train_qa.py:103` default `mrm8488/roberta-base-bne-finetuned-sqac`; `:5` references `PlanTL-GOB-ES/roberta-base-bne-sqac`.

**Correction — the port is three pieces, not two.** The finder said "write only two new pieces (question template + row→example builder)". Two further parts are Spanish-locked beyond the checkpoint: the teacher prompt at `prepare_data.py:115-121` is written in Spanish (*"Para cada PREGUNTA, extrae del DOCUMENTO el fragmento textual EXACTO…"*, with a Qwen-specific `/no_think`), and `evaluate_qa.py:9` documents "Spanish-aware normalization" in the EM/F1 scorers. Budget a prompt rewrite and a normalizer check.

**Change:** seed `classifier/src/models/qa_span/` from the cited ranges; swap `--model` to a biomedical SQuAD checkpoint; translate the teacher prompt; keep the document-level split, negative downsampling and `best_tau` contract unchanged.
**Impact:** high. **Effort:** days.

---

#### 4. Siblings calibrate the decision threshold to disk; the classifier hardcodes three different ones in source

**Evidence.** Writer: `qa_bert/evaluate_qa.py:173-180` emits `{"best_tau": …}`. Reader: `species_qa_service/app.py:113-125` `_calibrated_min_score()` loads `qa_bert/results/qa_eval.json` with `SPECIES_QA_MIN_SCORE` override. `species_qa_service/README.md:87-89` states the contract: *"If you retrain, re-run evaluate_qa.py against the new model and point … at it — the threshold and the model must match."*

Classifier: `api/fastapi_multitask.py:34` `THRESHOLD = 0.360   # optimised on the 500-sentence test set`; `api/fastapi_pipeline.py:99` `ML_THRESHOLD = 0.5`; `results/error_analysis/ep_relax/error_analysis.json` `"threshold": 0.13`. Three operating points for one task family, none written by an evaluation script.

**Change:** have evaluation write `{threshold, model_dir, eval_set}` beside the checkpoint; load it at startup with an env override, refusing to start on a model_dir mismatch. Copy `app.py:113-125` verbatim as the loader.
**Impact:** medium. **Effort:** hours.

---

#### 5. The classifier is the only component with no packaging, no config-as-data, and no run provenance

**Evidence — counts re-run.** `ls classifier/scripts/*.py | wc -l` = **80**. `sys.path.insert|append` across `src api scripts tools validator experiments` = **81** (finder said 80; off by one). No `pyproject.toml` / `setup.py` / `setup.cfg`. Only `src/agents/__init__.py` exists, so `src/` is not a package.

Against that: `biotx_community_check/pyproject.toml` with a `dev` extra; `api/models.py:1-82` typed request/response models. `ampliseq/pipeline/__init__.py:8-10` fixes `ROOT` once; `db.py` declares 4 tables (`runs:22`, `species_results:39`, `sample_results:50`, `validation_runs:61`); `manifest.py:17` `_sha256`, `:28` `_git_sha`, `:49` `build_manifest`; `config/pipeline.yaml`; `runner.py` as `python -m`. `biotx_community_check/tests/test_check.py` = 56 lines, 3 test functions (confirmed "3 mocked smoke tests").

**Change:** (1) `classifier/pyproject.toml` + `src/__init__.py`, then delete `sys.path` blocks as imports resolve; (2) move `MODEL_CONFIG` (`fastapi_pipeline.py:81-90`), `GENERATIVE_MODEL_PATH:94-97`, `ML_THRESHOLD:99`, `MAX_LENGTH:100` into `config/classifier.yaml`; (3) adopt `manifest.py` in `scripts/train_cv_regularized.py` so every number in `results/` carries input SHA-256s and a git SHA.
**Impact:** medium. **Effort:** weeks.

---

#### 6. `kingdom_mapper.py` is dead code that is wrong by its own test

**Evidence — exact.** `src/data/kingdom_mapper.py:359-364`:

```python
    # 3. Check if it looks like a binomial name (Genus species)
    # Most binomials in ecology papers are animals
    if re.match(r'^[A-Z][a-z]+\s+[a-z]+$', taxon_name.strip()):
        # Could be animal, plant, or fungi - default to Animalia
        return 'Animalia'
```

`:457` `("Escherichia coli", "Bacteria"),  # Will default to Animalia (binomial)`. Sole importer tree-wide: `scripts/archive/build_hybrid_dataset.py:33`.

**Change:** delete the module; route kingdom lookups through `ott_resolver._kingdom_of` — **but only after the vocabulary fix in "missed" item 2 below**, otherwise the replacement is a different silent failure.
**Impact:** low. **Effort:** hours.

---

#### 7. `biotx_retrieval_eval_100.csv` shows the deployed classifier is a near pass-through, and is not a ranked evaluation

**Evidence — every number reproduced.** Naive `read_csv` raises `ParserError: Expected 2 fields in line 6, saw 3`; `sep=';', encoding='utf-8-sig'` yields (100, 29). `rank == 1` for **all 100** rows; 100 distinct `doc_id`, 100 distinct triple `id`. `classifier` vs `triples_ok_full`: accepts 91, **TP 65 / FP 26 / FN 0 / TN 9** → precision **0.7143**, recall **1.000**, reject-recall **9/35 = 0.2571**. By arm: globi 48/50 accepted (41 correct), random 43/50 accepted (24 correct). Of the 9 rejects, **7** still have `interaction_eval == 1`.

**Two corrections.** (a) The finder wrote "21 triples had 2-38 passages available" — that conflates two columns: `passages_count > 1` is **26 rows** (max 38); `docs_count > 1` is **21 rows** (max 36). The upstream ranked data existed either way. (b) The `classifier` column is `int64` with values `{0,1}` only — **no probability is stored**. The proposed rank-separation / permutation test therefore requires re-scoring all 100 sentences with the champion; it cannot be run off this file as-is.

**Change:** stop describing this as a ranked eval; re-export retrieval keeping `rank > 1`; then point `evaluate_reranking_extrinsic.py:133-165` (rank separation) and `validate_reranking_globi.py:148-226` (`reciprocal_rank`, `ndcg@k`, `permutation_test`, `bootstrap_ci`) at it with truth = `triples_ok_full` and score = a freshly computed champion probability.
**Impact:** high. **Effort:** days.

---

#### 8. A full human-in-the-loop curation system exists and no serving path feeds it

**Evidence.** `tools/curation_db.py:24` `CREATE TABLE curation_queue`, four indices at `:43-46`, `PRAGMA journal_mode=WAL` at `:59`, `submit_decision` at `:194-228`. `grep -rn 'curation|curate' classifier/api/*.py` returns **0** — re-run and confirmed zero across `ensemble_api.py`, `fastapi_distilled.py`, `fastapi_ensemble.py`, `fastapi_multitask.py`, `fastapi_pipeline.py`, `trust_service.py`.

**Change:** fire-and-forget enqueue in the predict handler when `abs(prob - threshold) < delta`. WAL mode makes the concurrent FastAPI writer safe. Choose `delta` from the observed score distribution.
**Impact:** medium. **Effort:** days.

---

#### 9. The siblings ship a local-LLM `/explain`; the classifier's only explanation path calls the Anthropic API

**Evidence.** `biotx_community_check/api/main.py:13-14` `_OLLAMA_URL = "http://localhost:11434/api/generate"`, `_EXPLAIN_MODEL = "qwen3:32b"`; `:87` `async def explain(req: ExplainRequest)`; `:91` *"Runs locally via Ollama (qwen3:32b) — no external API calls."* Meanwhile `classifier/validator/interaction_validator.py:111` `client = anthropic.Anthropic()` and `:115` `model="claude-3-haiku-20240307"` — the one explanation path inside the classifier violates the project's own "never use the Anthropic API key for background work" rule.

**Change:** port the `/explain` handler shape into the classifier's serving app, prompting from structured output (species + OTT ids + kingdoms + lexicon hits + outcome code); point it at local Ollama; retire the Anthropic client.
**Impact:** medium. **Effort:** days.

---

#### 10. Transfer runs both ways: the classifier's statistical rigour is missing from the siblings

**Evidence.** `tests/test_training_data.py:34-39` sets numeric gates (`GATE1_MAX_INVALID_RATE = 0.01`, `GATE2_MAX_VIOLATION_RATE = 0.005`, `GATE4_MIN_HARD_NEGATIVE_RATIO = 0.50`, `GATE5_MIN/MAX_POS_RATIO = 0.20/0.50`) and asserts on them (`:102`). `scripts/bootstrap_ci.py:114` `bootstrap_ci`, `:143` `mcnemar`, `:207-208` pairwise comparisons. Against that, `biotx_community_check/tests/test_check.py` is 56 lines / 3 mocked smoke tests measuring no accuracy, and its orphan / absent-partner flags have never been scored.

**Change:** apply `bootstrap_ci.py` to the biotx anomaly flags on the `honey_rbcl` and `fungal_its` runs so the demo carries an interval; add a numeric-gate module to `ampliseq/pipeline/tests`.
**Impact:** low. **Effort:** days.

---

### PARTIALLY CONFIRMED — corrected

#### 11. The `ORGANISM` hardcode is real, but it is **not** the cause of the zero-triple KG, and the proposed fix does not work

**Confirmed.** `classifier/api/fastapi_pipeline.py:604-608` is verbatim as quoted. `NER_PAIR_TABLE` (`kg_builder.py:99-109`, not 100-108) contains no `ORGANISM` key. `robi_prefilter.is_pair_plausible` short-circuits on unknown kingdoms (`:135-136`), so `kingdom=None` makes the gate at `kg_builder.py:361` a no-op.

**Refuted — "no edge is ever created."** `infer_relation` has three fallbacks after the type-pair lookup: robiext term lookup (`:236-247`), symmetric set (`:249-257`), and an unconditional Step 4 (`:259-264`). I executed the builder with the exact dict the pipeline emits:

```
add_from_api_response(entities=[ORGANISM/None, ORGANISM/None], interaction_type='pathogenOf')
  → KGEdge(predicate='pathogenOf', directed=False, ro_id=None, flag='unvalidated')
interaction_type=''  → KGEdge(predicate='interactsWith', directed=False, flag='ambiguous')
n_edges 2
```

Edges *are* produced — untyped, undirected, unvalidated, but produced.

**Refuted — the zero-triple location.** The 0 triples come from a different file: `experiments/knowledge_graph/tier2_triple_extractor.py:181-208`, where **both** emission branches require `hosts` to be non-empty. Hard evidence on disk: `experiments/knowledge_graph/results/tier2/triples.jsonl` is **0 bytes / 0 lines**, and `kg_stats.json` records `"n_high_confidence_triples": 0, "n_low_confidence_triples": 0` while its 8,859 edges all carry Tier-1 `source_breakdown` provenance (`v7_llm_cleaned`, `epmc_direct`, …).

**Refuted — "the model already emits it."** I ran the champion (`models/multitask/full_typed_a05_ner2`, CPU, inference only) over six textbook host-pathogen sentences. Across all six, the NER head emitted **15 `B-SPECIES`, 33 `I-SPECIES`, 2 `B-INT`, 2 `I-INT` — and zero `HOST`, zero `PATHOGEN`.** Example: *"Plasmodium falciparum is transmitted to humans by Anopheles gambiae mosquitoes"* → every span tagged `SPECIES`. The labels exist in `multitask_config.json` but never fire, exactly as the audit premise states. Feeding predicted labels into `fastapi_pipeline.py:606` would set every entity to `SPECIES`, which is also absent from `NER_PAIR_TABLE` — the KG stays untyped.

**Corrected change.** Two independent fixes, neither of which is the finder's:
- `fastapi_pipeline.py:604-608` → populate `kingdom` from `OTTResolver` (after the vocabulary fix). This does not create edges; it converts today's `flag='unvalidated'` edges into ROBI-gated ones and makes `_orient_by_ner` functional. Impact: medium. Effort: hours.
- Tier 2's zero output is a **label-supervision** problem at `tier2_triple_extractor.py:181-208`, not a glue problem. The cheap interim fix is to relax the emission rule to use `SPECIES` spans plus the `INT` span for direction; the real fix is out of this pass's scope. Impact: high. Effort: days.

#### 12. Grounding is a normalizer, not a filter — conclusion holds, the specific F1 does not reproduce

**Reproduced exactly.** Baseline from `results/error_analysis/ep_relax/error_analysis.json` and recomputed from `all_predictions.csv`: threshold 0.13, P=0.7931, R=0.9583, F1=0.8679, TP=46 FP=12 FN=2 TN=40. Literal string matching: **31/46 TPs and 6/12 FPs survive** — both figures match the finder to the row.

**Did not reproduce exactly.** My independent Direction A (resolve each term, expand to all indexed names for that concept plus genus and abbreviated-genus, word-boundary match) gives TP 43/46, FP 11/12 → P=0.796, R=0.896, **F1=0.843**, against the finder's 45/46, 11/12, F1=0.865. The gap is variant-expansion permissiveness. **The conclusion is unchanged and slightly strengthened:** under both implementations grounding lands at or below the 0.868 baseline, precision moves by ≤0.011, and FPs pass the gate at essentially the TP rate (11/12 vs 43-45/46). Do not ship grounding as a hard gate.

**Correction.** The one row where a gate is genuinely right is `error_analysis.json` `false_positives[2]`, whose `species2_term` is literally `"ÃAustralia"` — mojibake, not the `"×Australia"` the finder rendered. That the eval file carries a mis-decoded non-taxon in a species column is itself worth an input-validation check.

**Change:** use the resolver for normalization into the model and for `match_type`/`canonical`/matched-surface-form as response fields at `fastapi_pipeline.py:144`. Impact: medium. Effort: hours.

#### 13. The classifier **does** have a systemd unit — it already has the restart policy the finder recommends adding

**Refuted.** `systemctl list-units` at system scope shows no classifier unit, which is what the finder ran. At **user** scope the unit exists and is running:

```
# /home/egaillac/.config/systemd/user/biotic-classifier.service
ExecStart=/home/egaillac/MetaP/MPvenv/bin/python -u classifier/api/fastapi_multitask.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/home/egaillac/MetaP/classifier/logs/api_distilled.log
```

`systemctl --user show`: `NRestarts=48947`, `ActiveState=activating`, still looping during this session. So `Restart=on-failure` + `RestartSec=10` — the finder's headline recommendation — are **already present** and are not what is broken.

**Confirmed and sharpened.** The crash is real and live: `fastapi_multitask.py:41` raises `OSError: Incorrect path_or_model_id: …/models/multitask/mt_distill_warm_ner0`, because `:33` points at a directory that does not exist (`ls models/multitask/` returns only `full_typed_a05_ner2`). The log is **532,264,572 bytes (507.6 MiB)** and grew during this session; I counted **137,428** `Loading MultiTask BiomedBERT` restart banners in it.

**Two defects the finder did not name.** (a) The log is `api_distilled.log` but carries the *multitask* API's output, and `start_distilled_api.sh:25` launches the same binary to the same file — two launchers competing for one port and one log, which is why nobody noticed. (b) `start_api.sh`'s nohup line is `:16`, not `:17`; `:13` prints `Model: SciBERT (F1=0.774, Precision=0.783)`, and there are five `start_*.sh` scripts, not four.

**Corrected change:** the fix is three lines in an existing unit, not a new file — add `StartLimitIntervalSec`/`StartLimitBurst` so the loop halts, switch `StandardOutput=append:` to `journal` (this alone prevents the 507 MB file), replace the `:33` literal with `Environment=MODEL_DIR=` plus a startup existence assertion, and delete the competing `start_distilled_api.sh` launcher. Impact: high. Effort: hours.

#### 14. ROBI is encoded three times — but the proposed plant×plant gate does not gate

**Confirmed.** `kg_builder.py:49-56` loads the sibling by absolute path via `importlib`, with a silent `return True, []` fallback. `robi_validator.validate_interaction(source, target, interaction_type, …)` at `:151-158` requires an interaction type and is reachable only from `tests/test_training_data.py:27`, `scripts/validate_training_data.py:27`, `kg_builder.py:47` — never from a serving path. `robi_prefilter.is_pair_plausible(kingdom_a, kingdom_b)` at `:123-144` needs none. Third copy confirmed at `src/data/quality_filter.py:306-314` (`validate_domain_rules`, same signature shape, same JSON).

**Refuted — the honey_rbcl fix.** `is_pair_plausible('Plantae','Plantae')` returns `(True, ['parasite','stem_parasite','root_parasite','epiphyte','symbiosis','mutualism','host'])`. Plant×plant is an **allowed** key (`:51-54`), not a blocked one. To suppress a crop co-mention you must test a *specific interaction type* against that list — which requires an interaction type, the very thing the finder correctly argues the binary classifier never emits. The reranker fix is therefore not 6 lines and is not reachable through this gate alone.

**Missed defect.** `_POSSIBLE` has **16** distinct keys, not 17: `frozenset` keys collide silently. `:115` (`virus`,`plant`) overwrites `:69` (`plant`,`virus`), and `:114` (`virus`,`animal`) overwrites `:94-96` (`animal`,`virus`). Verified at runtime: `_POSSIBLE[frozenset({'animal','virus'})] == ['pathogen','infects','transmits']` — `"vector"` and `"host"` are gone. Any consumer using the returned type list for vector-borne disease gets a wrong answer.

**Change:** `pip install -e ../biotx_community_check` and replace the importlib hack with a real import; declare `robi_prefilter` authoritative at inference and `robi_validator` for typed gating; delete `quality_filter.py:306-395`; fix the two colliding dict keys. Impact: medium. Effort: hours.

#### 15. Do not port the Spanish gazetteer — but the part the finder wants to port is the Spanish-specific part

**Confirmed.** `build_vernacular.py:8` hardcodes `FILTER(LANG(?cn)="es")`; `_zipf(…, "es")` at `:116` and `:122`; `vernacular_es.tsv` is 24,050 lines. Unnecessary for English: I probed the OTT index and **14 of 15** English vernaculars resolve as synonym rows — `horses→Equidae`, `humans→Homo`, `moose→Alces americanus`, `tomato→Solanum lycopersicum`, `butterflies→Papilionoidea`, `honey bee→Apis mellifera`, `striped mullet→Chelon tricuspidens`, `deer tick→Ixodes scapularis`; `malaria parasite` is the only miss. Confirmed exactly. The longest-match premise also holds: `leopard→Panthera pardus` and `northern leopard frog→Rana pipiens` are both indexed.

**Correction.** The finder said to port `vernacular_resolver.py:57-92` including "a documented capitalisation guard". That guard (`:72-79`) rejects any Capitalised match, and its own comment states why: *"genuine common-name mentions run lowercase in Spanish prose ('el milano real'), whereas a leading capital signals a proper noun."* In English biomedical prose, common names are routinely capitalised in titles and sentence-initial position — porting this verbatim would silently drop exactly the vernacular mentions it is meant to catch. Port the longest-phrase-first loop (`:60-71`, `:80-92`); drop the guard and replace it with a genus-token check.

**Change:** OTT-backed English mention scanner replacing the regex NER at `fastapi_pipeline.py:100-136`. Impact: medium. Effort: days.

---

### Missed by the finder

1. **The champion's NER head emits zero HOST and zero PATHOGEN.** Measured on the actual checkpoint (6 sentences, 52 non-`O` tags, all `SPECIES`/`INT`). This kills the "the model already emits it" fix and is the true cause of the empty `triples.jsonl`.
2. **The kingdom vocabulary does not connect — this blocks four recommendations at once.** `OTTResolver.resolve('Hordeum vulgare').kingdom` returns **`"Chloroplastida"`** and `Plasmodium falciparum` returns **`"Eukaryota"`**. `robi_prefilter._group` maps both to `"unknown"` (its `_PLANTS` set contains `plantae/viridiplantae/algae/chromista/archaeplastida` — not `chloroplastida`), so `is_pair_plausible` short-circuits to `(True, [])` for **every plant and every protist** even after kingdoms are wired in. The same mismatch already exists inside the classifier: `kg_builder._orient_by_ner:277,279` tests `kingdom in ("Plantae","Viridiplantae")` and will never fire on OTT output; and `results/tier2/kg_stats.json` already mixes `Plantae` (308), `Metazoa` (230), `Viridiplantae` (20), `Chromista` (43), `Archaeplastida` (1). This is a two-line fix to `_PLANTS` plus a rank-climb adjustment for `Eukaryota`, and it is a **prerequisite** for the ORGANISM fix, the ROBI gate, the reranker gate, and the `kingdom_mapper` replacement.
3. **`robi_prefilter._POSSIBLE` silently loses two entries to duplicate `frozenset` keys** (16 keys, not 17); `animal×virus` loses `"vector"` and `"host"`.
4. **`entity_prefix_test.json`'s `"threshold": 0.0` is a hardcoded literal** (`test_entity_prefix.py:122-124`), discarding the computed `best_thr`.
5. **The classifier's systemd unit exists in user scope** and already carries `Restart=on-failure` / `RestartSec=10`.
6. **Launcher/log collision:** `api_distilled.log` carries `fastapi_multitask` output, written by both the user unit and `start_distilled_api.sh:25`.
7. **qa_bert's teacher prompt and EM/F1 normalizer are Spanish too**, not just the checkpoint.
8. **The regex NER is not merely vernacular-blind, it is precision-broken.** Running `fastapi_pipeline.extract_species` over the 100 EP-relax sentences yields ≥2 "species" on **80/100** rows, including `"Biology and"`, `"South and"`, `"Previous studies"`, `"Stain and"`, `"Piptadenia and"`. The KG path is being fed noise, not starved of input — which is why replacing it (finding 15) matters more than the finder argued.

---

### Duplication inventory — verified, with corrections

| Store | Path | Size (measured) | Returns OTT id? | Kingdom? | Called by |
|---|---|---|---|---|---|
| **#1 authoritative** | `classifier/src/data/ott_resolver.py:149-296` → `ott_index.sqlite` | **1,071,099,904 B** — 4,529,129 concepts / 9,316,799 names / 7,579,845 distinct / 4,787,671 synonyms | **yes** | yes (but returns `Chloroplastida`/`Eukaryota` — see missed #2) | `species_qa_service/extractor.py:35` **only** |
| #2 | `classifier/src/data/ott_lookup.py:104-134` → `species_dict.csv` | 165,143,141 B — 7,898,556 names | **never** (`None` at `:132`) | no | `classifier/api/fastapi_pipeline.py:62,144` — **the deployed path** |
| #3 | `experiments/knowledge_graph/improved_ott_resolver.py:75-141` → `ott_local_index.pkl` (+ gnverifier, OTT source id `179` at `:37`) | 322,881,257 B | yes | no | `ampliseq_rerank.py:36,523` |
| #4 | `ampliseq_rerank.py:45-77` `resolve_ott` → SIBiLS autocomplete | network | **20** scattered `ott_cache.json` (12 under `ampliseq/pipeline/results/`, 8 under `classifier/.../ampliseq_rerank/`) — finder said 18 | yes | no | ampliseq rerank main path |
| #5 | `biotx_block/harmonize.py:13-119` → CoL + OTT TNRS | network | yes (+CoL id) | yes | `biotx_block/check.py:187` |
| #6 | `src/data/taxonomy_validator.py:139-168` → `globi_taxonomy_cache.json` | 157,944,334 B — **826,319** species (verified) | no | yes | quality gates only |

**Overlapping on-disk taxon indices: 1.071 + 0.165 + 0.323 + 0.158 = ~1.72 GB**, plus 20 network-derived caches. Retire #2, #3, #4; keep #5 where a CoL id is required and #6's kingdom table as an offline fallback for names OTT misses.

**ROBI:** 3 encodings confirmed — `robi_validator.py:151-211` (typed, needs interaction_type, gates-only), `quality_filter.py:306-395` (pure duplicate, delete), `robi_prefilter.py:123-144` (kingdom-pair, inference-usable, **16-key table with 2 silently overwritten entries**).

**Kingdom assignment:** 4 implementations confirmed — `kingdom_mapper.py:359-364` (wrong, dead), `ott_resolver.py:270-284` (correct mechanism, incompatible vocabulary), `taxonomy_validator.py:153-168` (826,319 species), `harmonize.py:90-95`.

**IR machinery:** single implementation, no second consumer — confirmed. `validate_reranking_globi.py:148-226` and `evaluate_reranking_extrinsic.py:133-165` are wired only to the ampliseq reranker via `ampliseq/pipeline/db.py:61` `validation_runs`.

**Packaging:** classifier has 0 of {pyproject, config-as-data, results DB, run manifest, system-scope unit}, 80 loose scripts, 81 `sys.path` hacks — confirmed, with the user-scope unit correction above.