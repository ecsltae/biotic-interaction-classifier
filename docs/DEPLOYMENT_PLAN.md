# Deployment plan — joint interaction + direction verifier

**Status of what exists today, verified 2026-09-25 by `ss -tlnp` and by reading the launch
scripts — not assumed:**

* **Nothing is listening on ports 8000, 8001 or 8002.** The only live services are 8003
  (`classifier.api.trust_service`, healthy, uptime 35 h), 8010 (species QA) and 8015
  (biotx community check).
* `CLAUDE.md` says *"Port 8001: original ensemble API"* and *"check API health:
  `curl http://localhost:8001/health`"*. Both are stale: nothing answers on 8001, and
  `start_api.sh:16` launches `api_scibert.py`, which loads
  `models/transformer_SciBERT_cv_regularized` — **SciBERT, not the ensemble**.
* `api/fastapi_ensemble.py:343` binds **port 8000**, not 8001.
* `start_api.sh:8` stops the service with `pkill -f "uvicorn.*8001"`, which cannot work:
  that app calls `uvicorn.run()` in-process, so the pattern never matches. A restart leaves
  the old process holding the port.
* Six launch scripts exist for four apps; two of them both claim port 8003.
* There is **no Dockerfile and no conda environment** anywhere in MetaP.
* The joint model is **not served by anything today** — it is a CLI script.

**But there IS a working deployment pattern on this machine, and it is not the `start_*.sh`
scripts.** Three services run as systemd units and survive reboot — verified `active`:

| unit | what | port |
|---|---|---|
| `biotx-community-check.service` | uvicorn from the shared venv, `Restart=always`, config via `Environment=`, journal logs | 8015 |
| `species-qa.service` | same shape | 8010 |
| `biotx-tunnel.service` | `cloudflared tunnel --url http://localhost:8015` | — |

That last one is also the answer to *"how does Emilie reach it"*: a public
`trycloudflare.com` URL, no VPN, no port forwarding. The live one right now is
`https://ist-demand-enclosed-sporting.trycloudflare.com`, read out of the journal by
`biotx_community_check/get_tunnel_url.sh`.

**Copy this pattern, not the `start_*.sh` scripts.** Ready-made units are in `deploy/`.

**Consequence: there is no running classifier service to swap out.** That removes the hardest
part of a deployment and makes the file-based route the obvious first step.

---

## Phase 0 — before anything ships (blocking)

| # | what | who | why it blocks |
|---|---|---|---|
| 0.1 | **Commit the inference code.** `predict_joint.py`, `eval_direction.py`, `train_direction.py`, `xenc_format.py`, `compare_vs_v1_full.py`, `eval_encoder_sweep.py` and `handoff/` are all **untracked** — not gitignored, just never committed. "Clone the repo" delivers none of them. | me | the handoff references files that exist on one disk only |
| 0.2 | **Decide the operating point.** Default 0.50 → P 0.852 / R 0.959. If curation time is the constraint, 0.95 → P 0.882 / R 0.878. This is a curation-economics decision, not a modelling one. | you | it is baked into `predict.py` as `INTERACT_THR` |
| 0.3 | **Confirm the direction convention with Emilie.** `FORWARD`/`REVERSE` are judged against the *canonical* relation, not the surface string. If her downstream store assumes surface order, every direction is inverted. | you + Emilie | silent, systematic wrong answers |

## Phase 1 — file-based batch (ready now)

This is the route I would take. It ships today and has no service to operate.

```bash
# 1. copy the package (422 MB, self-contained, no network at inference)
rsync -a classifier/handoff/biotic_verifier/ <target>:~/biotic_verifier/

# 2. on the target
python -m venv venv && source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.*
pip install -r requirements.txt

# 3. verify the install — must print "3 candidates, 3 accepted"
python predict.py --in example_input.csv --out /tmp/check.csv

# 4. real work
python predict.py --in candidates.csv --out scored.csv --threads 16
```

Throughput ~34 candidates/s on 8 threads on real passages (~120k/hour); roughly linear in
`--threads`. 1.5 GB resident.

**What I would deliberately not build yet:** a service, a container, a queue, or a database
write path. There is no live consumer to satisfy, and a CSV in / CSV out contract is the one
thing that cannot break in a way nobody notices.

## Phase 2 — service, only if a live consumer appears

Everything needed is in `deploy/`:

```
deploy/metap-verifier.service          systemd unit, port 8004, modelled on the working ones
deploy/metap-verifier-tunnel.service   optional public URL via cloudflared
deploy/get_verifier_url.sh             prints the current tunnel hostname
```

Install:

```bash
sudo cp deploy/metap-verifier*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now metap-verifier metap-verifier-tunnel
./deploy/get_verifier_url.sh          # the URL to give Emilie
```

Still to write when it is actually needed: `api/verifier_service.py`. Design notes:

1. **Do not extend `fastapi_ensemble.py`.** It binds 8000, is one of six overlapping launch
   scripts, and its restart path is broken. Write one new app.
2. **Port 8004.** Verified free (8000, 8001, 8002, 8004–8006 are all free; 8003, 8010, 8015 are
   taken).
3. **The request schema must change.** Every existing endpoint takes a bare *sentence*. This
   model takes a *triple plus a passage*. There is no backward-compatible shim — the whole point
   is that the answer depends on which pair is asked about. Make the **batch** endpoint primary:
   the rule layer emits candidates in bulk, and `predict()` is already batched internally.
4. **Response**: `interacts`, `p_interact`, `direction`, `p_species1_is_subject`,
   `direction_confidence`, `both_taxa_located`, `unknown_polarity` — the columns `predict()`
   already returns.
5. **No NER, no GloBI scan.** The existing `/kg` path runs a sentence classifier and then infers
   entities and direction downstream with regex NER + ROBI heuristics. The joint model replaces
   that whole path: the rule layer already supplies both taxa and the relation. Drop those
   layers rather than porting them.
6. **Health endpoint must score a fixed candidate and assert a known value**, not return
   `{"status":"ok"}`. The failure that matters is a missing polarity lexicon, which a liveness
   probe cannot see.
7. **Use the uvicorn CLI** (as the systemd unit does), not in-process `uvicorn.run()` — that is
   why `pkill -f uvicorn` silently fails for `api_scibert.py`.
8. **Quick tunnels change hostname on every restart.** Either document
   `deploy/get_verifier_url.sh` as a step, or set up a named tunnel if Emilie needs a stable URL.

### Checked and NOT a problem

Every existing server lowercases its input (`preprocess_text()` in `fastapi_ensemble.py:20`
and its copies). I tested whether that would corrupt the joint model: **it does not.** The
backbone is BiomedBERT-**uncased** (`do_lower_case=True`), so lowercasing is a no-op — over 60
benchmark rows, max |Δp_interact| = 0.000000 and zero decisions or directions changed. Reuse
or drop that preprocessing as convenient.

## Phase 3 — hygiene, worth doing regardless

* Correct the stale port/model claims in `CLAUDE.md` (§"Port 8001", "Check API health").
  I have not edited it — it is your instruction file.
* Retire or consolidate the six launch scripts; two collide on 8003.
* Write a Dockerfile if this ever needs to run somewhere that is not this machine. Nothing
  exists to reuse.

---

## Risks, in the order they are likely to bite

1. **Direction convention mismatch** (canonical vs surface). Systematic, silent, inverts every
   direction. Mitigation: Phase 0.3, and the two worked examples in the handoff README.
2. **Packaging drops the lexicon.** `polarity.py`, `robi_maps.json` and
   `data/robiext_v2025.json` must travel together. Until today this failed *silently* and
   degraded direction to near-chance with exit code 0; both copies now refuse to start and say
   what is missing. Verified by hiding the file and re-running.
3. **Upstream entity errors.** About half the residual errors are wrong taxon resolution in the
   rule layer. This model verifies whatever pair it is handed; it cannot fix a wrong one.
4. **Co-occurrence in a shared host.** The main residual false-positive class — two organisms
   both related to a third read as interacting. `example_input.csv` row 3 is a live case,
   accepted at p=0.88.
5. **Threshold drift across prevalence.** The model is far more stable than what it replaces
   (F1 spread 0.070 vs 0.169 across prevalence 0.31–0.85), but it is not immune. If the
   candidate mix changes a lot, re-measure rather than assume.
