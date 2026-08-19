#!/usr/bin/env python3
"""
fastapi_pipeline.py — Enriched biotic interaction pipeline (port 8002).

This is a NEW service that adds interpretable layers on top of the existing
ensemble classifier.  It does NOT modify fastapi_ensemble.py (port 8001).

Architecture (per request):
  Layer 1: NER          — Species extraction (regex + OTT validation)
  Layer 2: GloBI scan   — Full 591-term GloBI interaction term detection
  Layer 2b: Lexicon     — STRONG/WEAK interaction lexicon scoring
  Layer 3: ML           — BiomedBERT+RoBERTa ensemble (same models as port 8001)
  Layer 4: Synthesis    — OutcomeCode + human-readable reasoning

Start with:
    bash classifier/start_pipeline.sh
    # or:
    uvicorn classifier.api.fastapi_pipeline:app --port 8002 --reload

Health check:
    curl http://localhost:8002/health
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

# ---------------------------------------------------------------------------
# Path setup (allow running from project root or classifier/ subdir)
# ---------------------------------------------------------------------------
for _candidate in [
    Path(__file__).parent.parent / "src",
    Path(__file__).parent / "src",
]:
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))
        break

from data.interaction_lexicon import score_sentence          # noqa: E402
from data.interaction_taxonomy import (                       # noqa: E402
    scan_globi_terms,
    classify_interaction_type,
    get_interaction_category_for_sentence,
)
from models.flan_t5_enriched import build_prompt, get_yes_no_ids  # noqa: E402
from data.ott_lookup import lookup as ott_lookup, preload as ott_preload  # noqa: E402
from utils.outcome_codes import synthesize_outcome            # noqa: E402
from data.trust_scorer import TrustScorer as _TrustScorer    # noqa: E402

logger = logging.getLogger(__name__)

_trust_scorer: Optional["_TrustScorer"] = None


def _get_trust_scorer() -> "_TrustScorer":
    global _trust_scorer
    if _trust_scorer is None:
        _trust_scorer = _TrustScorer()
    return _trust_scorer

# ---------------------------------------------------------------------------
# Config (same model paths as port 8001 — read-only, no shared state)
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "biomedbert": {
        "path": "/home/egaillac/MetaP/classifier/models/precision_ensemble/biomedbert_precision",
        "weight": 0.70,
    },
    "roberta": {
        "path": "/home/egaillac/MetaP/classifier/models/precision_ensemble/roberta_precision",
        "weight": 0.30,
    },
}
# Generative Layer 3 — FLAN-T5 enriched model (preferred over discriminative ensemble)
# Set GENERATIVE_MODEL_PATH env var to override, or leave empty to auto-detect.
import os as _os
GENERATIVE_MODEL_PATH = _os.environ.get(
    "GENERATIVE_MODEL_PATH",
    "/home/egaillac/MetaP/classifier/models/flan_t5_enriched",
)

ML_THRESHOLD = 0.5
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# NER — regex-based species extraction (TaxoNERD optional)
# ---------------------------------------------------------------------------

# Binomial nomenclature patterns
_BINOMIAL = re.compile(
    r'\b([A-Z][a-z]{2,})\s+([a-z]{3,})\b'   # Genus species
    r'|\b([A-Z])\.\s*([a-z]{3,})\b'          # G. species
)
_QUALIFIER = re.compile(r'\b([A-Z][a-z]{2,})\s+(sp\.|spp\.|cf\.|aff\.)\b')

# Words that look like genus names but aren't
_COMMON_WORDS = {
    "The", "This", "These", "That", "Those", "For", "And", "But", "With",
    "From", "Into", "Over", "When", "Such", "Both", "Each", "Other",
    "Their", "Which", "While", "Where", "Under", "Until", "Upon",
    "Here", "There", "Then", "Thus", "After", "Before",
    "Wolf", "Bear", "Fish", "Bird", "Snake", "Frog", "Deer",
    "Tree", "Grass", "Plant", "Herb", "Seed", "Root", "Leaf",
}


def extract_species(text: str) -> List[dict]:
    """Extract species mentions from text using regex + OTT validation.

    Returns list of {text, start, end, ott_id, valid} dicts, deduped by name.
    """
    found: dict[str, dict] = {}

    for m in _BINOMIAL.finditer(text):
        if m.group(1) and m.group(2):
            name = f"{m.group(1)} {m.group(2)}"
            start, end = m.start(), m.end()
        elif m.group(3) and m.group(4):
            name = f"{m.group(3)}. {m.group(4)}"
            start, end = m.start(), m.end()
        else:
            continue
        if m.group(1) in _COMMON_WORDS:
            continue
        if name not in found:
            ott = ott_lookup(name)
            found[name] = {
                "text": name,
                "start": start,
                "end": end,
                "ott_id": ott["ott_id"] if ott else None,
                "taxon_name": ott["name"] if ott else None,
                "rank": ott["rank"] if ott else None,
                "valid": ott is not None,
            }

    for m in _QUALIFIER.finditer(text):
        name = f"{m.group(1)} {m.group(2)}"
        if m.group(1) not in _COMMON_WORDS and name not in found:
            found[name] = {
                "text": name,
                "start": m.start(),
                "end": m.end(),
                "ott_id": None,
                "taxon_name": None,
                "rank": None,
                "valid": False,
            }

    return list(found.values())


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str
    doi: Optional[str] = None  # optional DOI for trust scoring


class SpeciesEntity(BaseModel):
    text: str
    start: int
    end: int
    ott_id: Optional[str] = None
    taxon_name: Optional[str] = None
    rank: Optional[str] = None
    valid: bool = False


class PipelinePredictionResponse(BaseModel):
    text: str

    # Layer 3: ML classifier
    label: str            # "interaction" | "no_interaction"
    probability: float

    # Layer 1: NER
    species: List[SpeciesEntity]
    n_species: int

    # Layer 2: GloBI + lexicon
    matched_globi_terms: List[str]  # terms from interaction_dict.csv
    interaction_terms: List[str]    # STRONG/WEAK lexicon patterns
    signal_strength: float
    interaction_category: Optional[str]  # canonical category

    # Layer 4: Outcome
    outcome_code: str
    reasoning: str

    # Layer 5: Trust (optional — present when species found or doi provided)
    trust: Optional[dict] = None


class BatchPredictRequest(BaseModel):
    sentences: List[str]


class BatchPredictionResponse(BaseModel):
    predictions: List[PipelinePredictionResponse]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Biotic Interaction Pipeline API",
    description=(
        "Enriched biotic interaction prediction with NER, GloBI term matching, "
        "interaction category classification, and structured outcome codes.  "
        "Port 8002 — does not modify the existing ensemble API on port 8001."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model store
_models: dict = {}
_tokenizers: dict = {}
_models_loaded: bool = False

# Generative model (FLAN-T5 enriched) — preferred Layer 3 when available
_gen_model = None
_gen_tokenizer = None
_is_generative: bool = False
_gen_yes_id: int = None
_gen_no_id: int = None


def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_models() -> None:
    global _models_loaded, _gen_model, _gen_tokenizer, _is_generative, _gen_yes_id, _gen_no_id

    if _models_loaded:
        return

    # ── Try generative model first ────────────────────────────────────────────
    gen_path = Path(GENERATIVE_MODEL_PATH)
    if gen_path.exists() and (gen_path / "config.json").exists():
        logger.info(f"Loading FLAN-T5 enriched model from {gen_path} ...")
        try:
            _gen_tokenizer = AutoTokenizer.from_pretrained(str(gen_path))
            _gen_model = AutoModelForSeq2SeqLM.from_pretrained(str(gen_path))
            _gen_model.to(DEVICE)
            _gen_model.eval()
            _gen_yes_id, _gen_no_id = get_yes_no_ids(_gen_tokenizer)
            _is_generative = True
            logger.info("  Generative Layer 3 active (FLAN-T5 enriched)")
        except Exception as e:
            logger.warning(f"  Failed to load generative model: {e} — falling back to ensemble")
            _gen_model = None
            _is_generative = False

    # ── Fall back to discriminative ensemble ──────────────────────────────────
    if not _is_generative:
        logger.info(f"Loading discriminative ensemble on {DEVICE}...")
        for name, cfg in MODEL_CONFIG.items():
            model_path = cfg["path"]
            if not Path(model_path).exists():
                logger.warning(f"Model not found: {model_path} — skipping {name}")
                continue
            logger.info(f"  {name} (weight {cfg['weight']}) from {model_path}")
            _tokenizers[name] = AutoTokenizer.from_pretrained(model_path)
            _models[name] = AutoModelForSequenceClassification.from_pretrained(model_path)
            _models[name].to(DEVICE)
            _models[name].eval()
        logger.info(f"Discriminative models loaded: {list(_models.keys())}")

    _models_loaded = True


def _ml_predict(
    sentences: List[str],
    species_strs: Optional[List[str]] = None,
    terms_strs: Optional[List[str]] = None,
) -> List[float]:
    """Return probabilities for label=1.

    Uses FLAN-T5 enriched model if available (preferred), else discriminative ensemble.
    species_strs and terms_strs are injected into the enriched prompt when using FLAN-T5.
    """
    if _is_generative and _gen_model is not None:
        # Build enriched prompts and score with log P(yes)/P(no)
        sp = species_strs or ["none"] * len(sentences)
        tr = terms_strs or ["none"] * len(sentences)
        prompts = [build_prompt(t, s, r) for t, s, r in zip(sentences, sp, tr)]

        _gen_model.eval()
        scores = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                enc = _gen_tokenizer(
                    batch, max_length=320, padding=True, truncation=True, return_tensors="pt"
                ).to(DEVICE)
                bos = torch.full(
                    (len(batch), 1),
                    _gen_model.config.decoder_start_token_id,
                    dtype=torch.long,
                ).to(DEVICE)
                out = _gen_model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    decoder_input_ids=bos,
                )
                logits = out.logits[:, 0, :]
                log_p = torch.log_softmax(logits.float(), dim=-1)
                yes_lp = log_p[:, _gen_yes_id].cpu().numpy()
                no_lp  = log_p[:, _gen_no_id].cpu().numpy()
                prob_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
                scores.extend(prob_yes.tolist())
        return scores

    # Discriminative ensemble fallback
    if not _models:
        return [0.5] * len(sentences)

    preprocessed = [_preprocess(s) for s in sentences]
    all_weighted_probs = []
    for name, model in _models.items():
        tokenizer = _tokenizers[name]
        weight = MODEL_CONFIG[name]["weight"]
        inputs = tokenizer(
            preprocessed, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_weighted_probs.append(probs * weight)

    return np.sum(all_weighted_probs, axis=0).tolist()


def _run_pipeline(text: str, prob: float, doi: Optional[str] = None) -> PipelinePredictionResponse:
    """Run all pipeline layers for a single sentence given an ML probability.

    Note: prob is already computed by _ml_predict() using the enriched prompt
    (species + GloBI terms injected) when the generative model is active.
    """
    # Layer 1: NER
    species_raw = extract_species(text)
    n_species = len(species_raw)
    species_entities = [SpeciesEntity(**s) for s in species_raw]
    species_names = [s["text"] for s in species_raw]

    # Layer 2: GloBI term scan
    matched_globi = scan_globi_terms(text)

    # Layer 2b: Interaction lexicon scoring (uses lowercased text)
    has_signal, strength, matched_lex = score_sentence(text.lower())

    # Interaction category
    all_terms = matched_globi + matched_lex
    cat = classify_interaction_type(all_terms) if all_terms else None

    # Negation / methodology detection (from lexicon internals)
    from data.interaction_lexicon import (
        _NEGATION_COMPILED,
        _METHOD_COMPILED,
    )
    t_lower = text.lower()
    has_negation = any(p.search(t_lower) for p in _NEGATION_COMPILED)
    method_count = sum(1 for p in _METHOD_COMPILED if p.search(t_lower))
    has_methodology = method_count >= 2

    # Layer 4: Outcome synthesis
    code, reasoning = synthesize_outcome(
        n_species=n_species,
        species_names=species_names,
        matched_globi_terms=matched_globi,
        interaction_terms=matched_lex,
        signal_strength=strength,
        has_negation=has_negation,
        has_methodology=has_methodology,
        ml_probability=prob,
        ml_threshold=ML_THRESHOLD,
        interaction_category=cat,
    )

    label = "interaction" if prob >= ML_THRESHOLD else "no_interaction"

    # Layer 5: Trust scoring (inline — no inter-service call)
    trust_dict: Optional[dict] = None
    if species_names or doi:
        try:
            sp1 = species_names[0] if len(species_names) > 0 else ""
            sp2 = species_names[1] if len(species_names) > 1 else ""
            itype = cat or ""
            trust_result = _get_trust_scorer().score(
                text=text, species1=sp1, species2=sp2,
                interaction_type=itype, doi=doi,
            )
            trust_dict = trust_result.to_dict()
        except Exception as _te:
            logger.warning("Trust scoring failed: %s", _te)

    return PipelinePredictionResponse(
        text=text,
        label=label,
        probability=round(prob, 4),
        species=species_entities,
        n_species=n_species,
        matched_globi_terms=matched_globi,
        interaction_terms=matched_lex,
        signal_strength=round(strength, 4),
        interaction_category=cat,
        outcome_code=code,
        reasoning=reasoning,
        trust=trust_dict,
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    _load_models()
    # Pre-load OTT species dict in background (non-blocking; first request loads it otherwise)
    import threading
    threading.Thread(target=ott_preload, daemon=True).start()
    logger.info("Pipeline startup complete.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Biotic Interaction Pipeline API",
        "port": 8002,
        "endpoints": ["/health", "/predict", "/predict_batch"],
        "note": "Enriched predictions with NER, GloBI term matching, and outcome codes. "
                "Original ensemble API remains on port 8001.",
    }


@app.get("/health")
def health():
    layer3 = (
        f"FLAN-T5 enriched ({GENERATIVE_MODEL_PATH})"
        if _is_generative
        else f"Discriminative ensemble: {list(_models.keys()) or 'none loaded'}"
    )
    return {
        "status": "ok",
        "device": DEVICE,
        "layer3_backend": "generative" if _is_generative else "discriminative",
        "layer3_detail": layer3,
        "pipeline_layers": [
            "NER (regex + OTT)",
            "GloBI term scan (591 terms)",
            "Interaction lexicon (STRONG/WEAK)",
            layer3,
            "Outcome synthesis",
        ],
    }


@app.post("/predict", response_model=PipelinePredictionResponse)
def predict(request: PredictRequest):
    text = request.text
    # Pre-compute enrichment for the generative Layer 3 prompt
    species_raw = extract_species(text)
    matched_globi = scan_globi_terms(text)
    species_str = ", ".join(s["text"] for s in species_raw) or "none"
    terms_str   = ", ".join(matched_globi) or "none"
    probs = _ml_predict([text], [species_str], [terms_str])
    return _run_pipeline(text, probs[0], doi=request.doi)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictRequest):
    if not request.sentences:
        return BatchPredictionResponse(predictions=[])
    # Pre-compute enrichment context for all sentences
    species_strs, terms_strs = [], []
    for text in request.sentences:
        sp = extract_species(text)
        gl = scan_globi_terms(text)
        species_strs.append(", ".join(s["text"] for s in sp) or "none")
        terms_strs.append(", ".join(gl) or "none")
    probs = _ml_predict(request.sentences, species_strs, terms_strs)
    results = [
        _run_pipeline(text, prob)
        for text, prob in zip(request.sentences, probs)
    ]
    return BatchPredictionResponse(predictions=results)


# ---------------------------------------------------------------------------
# /kg endpoint — typed directed KG edges from NER + interaction_category
# ---------------------------------------------------------------------------

_kg_builder_mod = None

def _get_kg_builder():
    global _kg_builder_mod
    if _kg_builder_mod is None:
        import importlib.util
        kg_path = (
            Path(__file__).resolve().parents[1]
            / "experiments/knowledge_graph/kg_builder.py"
        )
        spec = importlib.util.spec_from_file_location("kg_builder", kg_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _kg_builder_mod = mod
    return _kg_builder_mod


class KGRequest(BaseModel):
    sentences: List[str]
    doi: str = ""
    min_confidence: float = 0.0


class KGEdgeOut(BaseModel):
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    directed: bool
    ro_id: Optional[str]
    confidence: float
    n_sources: int
    flag: Optional[str]


class KGResponse(BaseModel):
    edges: List[KGEdgeOut]
    stats: dict


@app.post("/kg", response_model=KGResponse)
def build_kg(request: KGRequest):
    """
    Build typed directed KG edges from sentences using the full enriched pipeline.

    For each sentence:
      - Layer 1 NER extracts species entities
      - ML layer produces interaction probability
      - interaction_category from GloBI/lexicon provides predicate hint
      - kg_builder.RelationMapper infers directed/undirected edge with ROBI validation

    Returns deduplicated edges with confidence = mean ML score across all supporting sentences.
    """
    mod = _get_kg_builder()
    kg  = mod.KnowledgeGraph()

    for sentence in request.sentences:
        # Run full pipeline to get NER + category + probability
        species_raw   = extract_species(sentence)
        matched_globi = scan_globi_terms(sentence)
        species_str   = ", ".join(s["text"] for s in species_raw) or "none"
        terms_str     = ", ".join(matched_globi) or "none"
        prob          = _ml_predict([sentence], [species_str], [terms_str])[0]

        if prob < request.min_confidence:
            continue

        # interaction_category from _run_pipeline (no second ML call — reuse prob)
        resp = _run_pipeline(sentence, prob)

        # Convert SpeciesEntity list → kg_builder entity dicts
        entities = [
            {"text": s.text, "type": "ORGANISM", "kingdom": None}
            for s in resp.species
        ]

        kg.add_from_api_response({
            "entities":         entities,
            "interaction_type": resp.interaction_category or "",
            "confidence":       prob,
            "sentence":         sentence,
            "doi":              request.doi,
        })

    kg.merge_edges()
    edges = [
        KGEdgeOut(
            subject      = e.subject.text or e.subject.normalized,
            subject_type = e.subject.entity_type,
            predicate    = e.predicate,
            object       = e.object.text or e.object.normalized,
            object_type  = e.object.entity_type,
            directed     = e.directed,
            ro_id        = e.ro_id,
            confidence   = round(e.confidence, 4),
            n_sources    = e.n_sources,
            flag         = e.flag,
        )
        for e in kg.edges
    ]
    return KGResponse(edges=edges, stats=kg.stats())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "fastapi_pipeline:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
    )
