"""
trust_service.py — Literature Trust Scoring microservice (port 8003).

Standalone FastAPI service. No ML model weights loaded — starts in < 2s.
Wraps TrustScorer (5 axes: temporal, evidence, contradiction, KG, retraction).

Start:
    bash classifier/start_trust_service.sh
    # or:
    uvicorn classifier.api.trust_service:app --host 0.0.0.0 --port 8003

Health check:
    curl http://localhost:8003/health

Score a claim:
    curl -X POST http://localhost:8003/score \
      -H 'Content-Type: application/json' \
      -d '{"text":"Ixodes ricinus carries Borrelia.","species1":"Ixodes ricinus","species2":"Borrelia burgdorferi","pub_year":2020}'
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "classifier"))

from src.data.trust_scorer import TrustScorer, lookup_by_doi, _ensure_trust_index
import src.data.trust_scorer as _ts_mod
from src.data.retraction_checker import (
    check_retraction_batch,
    _load_retraction_watch,
    DEFAULT_CACHE_DB,
)
import src.data.retraction_checker as _rc_mod
from src.data.document_trust import score_document, build_dossier
from src.data.llm_judge import assess_document, DEFAULT_MODEL
from src.data.doc_resolver import resolve_reference

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Literature Trust Service",
    description="Multi-axis credibility scoring for ecological interaction claims.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_startup_time: float = 0.0
_scorer: Optional[TrustScorer] = None


def _get_scorer() -> TrustScorer:
    global _scorer
    if _scorer is None:
        _scorer = TrustScorer()
    return _scorer


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    text: str = ""
    species1: str = ""
    species2: str = ""
    interaction_type: str = ""
    pub_year: Optional[int] = None
    doi: Optional[str] = None


class ScoreDoiRequest(BaseModel):
    doi: str
    species1: str = ""
    species2: str = ""
    interaction_type: str = ""


class BatchScoreRequest(BaseModel):
    items: List[ScoreRequest]


class BiotXplorerHitsRequest(BaseModel):
    hits: List[Dict[str, Any]]
    species1: str = ""
    species2: str = ""


class PipelineScoreRequest(BaseModel):
    text: str = ""
    species: List[Dict[str, Any]] = []
    doi: Optional[str] = None


class DocumentRequest(BaseModel):
    ref: str = ""          # any identifier: DOI, PMID, PMCID, or title
    doi: str = ""          # back-compat alias
    max_references: int = 60

    def identifier(self) -> str:
        return (self.ref or self.doi or "").strip()


class DocumentAssessRequest(DocumentRequest):
    model: str = DEFAULT_MODEL


def _resolve_or_404(req: DocumentRequest):
    """Resolve any reference to a DOI, or raise 404 with a helpful message."""
    ident = req.identifier()
    if not ident:
        raise HTTPException(status_code=422, detail="Provide 'ref' (DOI, PMID, PMCID, or title).")
    resolved = resolve_reference(ident)
    if not resolved.matched or not resolved.doi:
        raise HTTPException(
            status_code=404,
            detail=f"Could not resolve '{ident}' to a DOI. {resolved.note}".strip(),
        )
    return resolved


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    global _startup_time
    _startup_time = time.time()
    _get_scorer()          # preload trust index + GloBI cache
    _load_retraction_watch()  # preload local Retraction Watch CSV if present
    logger.info("Trust service ready.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    _ensure_trust_index()
    n_doi = len(_ts_mod._trust_doi_index) if _ts_mod._trust_doi_index else 0
    n_sp = len(_ts_mod._trust_index) if _ts_mod._trust_index else 0
    try:
        import sqlite3
        conn = sqlite3.connect(str(DEFAULT_CACHE_DB))
        n_cached = conn.execute("SELECT COUNT(*) FROM retraction_cache").fetchone()[0]
        conn.close()
    except Exception:
        n_cached = 0
    return {
        "status": "healthy",
        "trust_index_species_pairs": n_sp,
        "trust_index_dois": n_doi,
        "retraction_watch_dois": len(_rc_mod._rw_doi_set) if _rc_mod._rw_doi_set else 0,
        "retraction_cache_entries": n_cached,
        "uptime_s": round(time.time() - _startup_time, 1) if _startup_time else 0,
    }


@app.get("/stats")
async def stats() -> dict:
    _ensure_trust_index()
    n_doi = len(_ts_mod._trust_doi_index) if _ts_mod._trust_doi_index else 0
    n_sp = len(_ts_mod._trust_index) if _ts_mod._trust_index else 0
    itype_counts: Dict[str, int] = {}
    if _ts_mod._trust_index:
        for (_, _, itype), _ in _ts_mod._trust_index.items():
            itype_counts[itype] = itype_counts.get(itype, 0) + 1
    doi_year_coverage = 0
    if _ts_mod._trust_doi_index:
        doi_year_coverage = sum(1 for v in _ts_mod._trust_doi_index.values() if v.get("pub_year"))
    return {
        "species_pair_entries": n_sp,
        "doi_entries": n_doi,
        "doi_with_year": doi_year_coverage,
        "doi_year_coverage_pct": round(100 * doi_year_coverage / n_doi, 1) if n_doi else 0,
        "interaction_type_counts": dict(sorted(itype_counts.items(), key=lambda x: -x[1])[:20]),
    }


@app.post("/score")
async def score(req: ScoreRequest) -> dict:
    """Score a single claim. Accepts text, species pair, DOI, pub_year."""
    try:
        result = _get_scorer().score(
            text=req.text,
            species1=req.species1,
            species2=req.species2,
            interaction_type=req.interaction_type,
            pub_year=req.pub_year,
            doi=req.doi,
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_doi")
async def score_doi(req: ScoreDoiRequest) -> dict:
    """Score by DOI. Resolves pub_year from trust index. Checks retraction via Crossref."""
    try:
        doi_meta = lookup_by_doi(req.doi) or {}
        result = _get_scorer().score(
            text="",
            species1=req.species1,
            species2=req.species2,
            interaction_type=req.interaction_type,
            pub_year=doi_meta.get("pub_year"),
            doi=req.doi,
        )
        out = result.to_dict()
        out["article_type"] = doi_meta.get("article_type", "")
        out["citation_count"] = doi_meta.get("citation_count", 0)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_batch")
async def score_batch(req: BatchScoreRequest) -> dict:
    """Score multiple claims efficiently. Returns per-item results + summary stats."""
    try:
        scorer = _get_scorer()
        results = []
        for item in req.items:
            r = scorer.score(
                text=item.text,
                species1=item.species1,
                species2=item.species2,
                interaction_type=item.interaction_type,
                pub_year=item.pub_year,
                doi=item.doi,
            )
            results.append(r.to_dict())

        if results:
            scores = [r["composite_trust"] for r in results]
            n_retracted = sum(1 for r in results if r.get("retracted"))
            n_low = sum(1 for s in scores if s < 0.4)
            summary = {
                "n_items": len(results),
                "mean_trust": round(sum(scores) / len(scores), 3),
                "min_trust": round(min(scores), 3),
                "max_trust": round(max(scores), 3),
                "n_retracted": n_retracted,
                "n_low_trust": n_low,
            }
        else:
            summary = {"n_items": 0}

        return {"results": results, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_biotxplorer_hits")
async def score_biotxplorer_hits(req: BiotXplorerHitsRequest) -> dict:
    """
    Annotate a list of BiotXplorer hit dicts with trust scores.

    BiotXplorer hits have at minimum:
      {"score": float, "interaction": {"preferred_term": str}}
    They may also have "pmid", "doi", "year", "title" fields.

    Returns the same hits with added trust_score, retracted, retraction_type fields.
    """
    try:
        scorer = _get_scorer()
        hits = req.hits

        # Extract DOIs/PMIDs from hits (BiotXplorer may include them)
        dois = []
        for hit in hits:
            doi = (hit.get("doi") or hit.get("DOI") or "").strip()
            dois.append(doi if doi else "")

        # Batch retraction check for all DOIs that are non-empty
        nonempty_dois = [d for d in dois if d]
        if nonempty_dois:
            ret_results = check_retraction_batch(nonempty_dois)
            ret_map = {r.doi.lower(): r for r in ret_results}
        else:
            ret_map = {}

        annotated = []
        trust_scores = []
        for hit, doi in zip(hits, dois):
            pub_year = hit.get("year") or hit.get("pub_year")
            if isinstance(pub_year, str) and pub_year.isdigit():
                pub_year = int(pub_year)
            elif not isinstance(pub_year, int):
                pub_year = None

            # Get interaction type from hit if available
            itype = ""
            if isinstance(hit.get("interaction"), dict):
                itype = hit["interaction"].get("preferred_term", "")

            r = scorer.score(
                text=hit.get("title", "") or hit.get("text", ""),
                species1=req.species1,
                species2=req.species2,
                interaction_type=itype,
                pub_year=pub_year,
                doi=doi or None,
            )

            annotated_hit = dict(hit)
            annotated_hit["trust_score"] = r.composite_trust
            annotated_hit["retracted"] = r.retracted
            annotated_hit["retraction_type"] = r.retraction_type
            annotated_hit["trust_detail"] = {
                "temporal": r.temporal_score,
                "evidence": r.evidence_score,
                "contradiction_risk": r.contradiction_risk,
                "kg_confirmed": r.kg_confirmed,
                "retraction_score": r.retraction_score,
            }
            annotated.append(annotated_hit)
            trust_scores.append(r.composite_trust)

        n_scored = sum(1 for d in dois if d)
        mean_trust = round(sum(trust_scores) / len(trust_scores), 3) if trust_scores else 0.0

        return {
            "hits": annotated,
            "mean_trust": mean_trust,
            "n_hits": len(annotated),
            "n_hits_scored_by_doi": n_scored,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score_pipeline_response")
async def score_pipeline_response(req: PipelineScoreRequest) -> dict:
    """
    Score trust for a pipeline /predict response (port 8002).

    Accepts the species list from the NER layer and an optional DOI.
    Returns per species-pair trust scores + overall trust.
    """
    try:
        scorer = _get_scorer()
        species_texts = [s.get("text", "") or s.get("taxon_name", "") for s in req.species]

        pair_scores = []
        for i in range(len(species_texts)):
            for j in range(i + 1, len(species_texts)):
                sp1, sp2 = species_texts[i], species_texts[j]
                r = scorer.score(
                    text=req.text,
                    species1=sp1,
                    species2=sp2,
                    doi=req.doi,
                )
                pair_scores.append({
                    "species1": sp1,
                    "species2": sp2,
                    "trust_score": r.composite_trust,
                    "retracted": r.retracted,
                    "retraction_type": r.retraction_type,
                    "kg_confirmed": r.kg_confirmed,
                    "explanation": r.explanation,
                })

        # If no pairs, score the text + DOI alone
        if not pair_scores:
            r = scorer.score(text=req.text, doi=req.doi)
            pair_scores = [{
                "species1": "", "species2": "",
                "trust_score": r.composite_trust,
                "retracted": r.retracted,
                "retraction_type": r.retraction_type,
                "kg_confirmed": r.kg_confirmed,
                "explanation": r.explanation,
            }]

        overall = round(
            sum(p["trust_score"] for p in pair_scores) / len(pair_scores), 3
        )
        n_retracted = sum(1 for p in pair_scores if p.get("retracted"))

        return {
            "trust_scores": pair_scores,
            "overall_trust": overall,
            "n_retracted": n_retracted,
            "doi": req.doi,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Document-level trust (truth in the literature)
# ---------------------------------------------------------------------------

@app.post("/document/resolve")
async def document_resolve(req: DocumentRequest) -> dict:
    """Resolve any reference (DOI, PMID, PMCID, or title) to a canonical DOI."""
    ident = req.identifier()
    if not ident:
        raise HTTPException(status_code=422, detail="Provide 'ref'.")
    return resolve_reference(ident).to_dict()


@app.post("/document/score")
async def document_score(req: DocumentRequest) -> dict:
    """Deterministic document trust: self-retraction + citation-graph evidence.

    Accepts any identifier (DOI, PMID, PMCID, or title). Fast (~1-3s), no LLM.
    """
    resolved = _resolve_or_404(req)
    try:
        report = score_document(resolved.doi, max_references=req.max_references)
        out = report.to_dict()
        out["resolved"] = resolved.to_dict()
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document/dossier")
async def document_dossier(req: DocumentRequest) -> dict:
    """Full evidence dossier: retraction facts + the passages where retracted
    works are cited. Accepts any identifier."""
    resolved = _resolve_or_404(req)
    try:
        dossier = build_dossier(resolved.doi, max_references=req.max_references)
        dossier["resolved"] = resolved.to_dict()
        return dossier
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document/assess")
async def document_assess(req: DocumentAssessRequest) -> dict:
    """LLM-judged document trust. Resolves any identifier to a DOI, builds the
    evidence dossier, then a local LLM (Qwen via Ollama) reasons over it — weighing
    HOW retracted works are cited, not just whether. Returns dossier + judgment."""
    resolved = _resolve_or_404(req)
    try:
        dossier = build_dossier(resolved.doi, max_references=req.max_references)
        judgment = assess_document(dossier, model=req.model)
        return {
            "doi": resolved.doi,
            "resolved": resolved.to_dict(),
            "llm_judgment": judgment.to_dict(),
            "dossier": dossier,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Demo UI
# ---------------------------------------------------------------------------

@app.get("/demo", response_class=HTMLResponse)
async def demo() -> str:
    demo_path = Path(__file__).resolve().parent / "trust_demo.html"
    if demo_path.exists():
        return demo_path.read_text(encoding="utf-8")
    return "<h1>Demo page not found</h1>"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("classifier.api.trust_service:app", host="0.0.0.0", port=8003, reload=False)
