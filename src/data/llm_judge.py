"""
llm_judge.py — LLM-based document trust judgment over an evidence dossier.

Design principle: the deterministic layer (document_trust.build_dossier) gathers
FACTS — is the paper retracted, which cited works are retracted, and the exact
passages where they are cited. It does NOT decide trustworthiness. This module hands
that evidence to a local LLM (Qwen 3.5 122B via Ollama) which does the *reasoning*.

Why the LLM matters: "cites a retracted paper" is not automatically disqualifying. A
review may cite a retracted work to debunk it, to give historical context, or to cite
a part unrelated to the retraction reason. Only reading the citing passage reveals
intent. The LLM weighs this; the deterministic score cannot.

Local only — never uses the Anthropic API (project rule). Model configurable; default
qwen3.5:122b to match the autonomous-work policy.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/generate"
# Default must fit the available GPU. qwen3.5:122b (81 GB) needs an 80 GB card;
# on a 20 GB slice use qwen3-vl:30b-a3b-instruct (19.6 GB, MoE ~3B active → fast).
# Override per-request via the `model` field, or set TRUST_LLM_MODEL.
import os as _os
DEFAULT_MODEL = _os.environ.get("TRUST_LLM_MODEL", "qwen3-vl:30b-a3b-instruct")


@dataclass
class LLMTrustJudgment:
    trust_score: float                 # 0.0–1.0, the LLM's calibrated judgment
    verdict: str                       # "trustworthy" | "use_with_caution" | "untrustworthy"
    confidence: float                  # 0.0–1.0
    reasoning: str                     # prose explanation
    per_issue: List[Dict[str, str]] = field(default_factory=list)  # [{issue, assessment}]
    recommendation: str = ""
    model: str = DEFAULT_MODEL
    raw: str = ""                      # raw LLM output (debugging)

    def to_dict(self) -> dict:
        return {
            "trust_score": round(self.trust_score, 3),
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "per_issue": self.per_issue,
            "recommendation": self.recommendation,
            "model": self.model,
        }


SYSTEM_INSTRUCTIONS = """\
You are a scientific literature trust assessor. You are given an EVIDENCE DOSSIER about \
a research document, assembled by a deterministic pipeline. Your job is to judge how far \
the document can be TRUSTED as a source, and explain WHY, reasoning from the evidence.

Critical judgment rules:
1. If the document ITSELF is retracted, it is untrustworthy as a source of claims \
(trust_score <= 0.15), regardless of anything else. Say so plainly.
2. If the document CITES retracted work, do NOT penalise blindly. Read the citation \
passages provided. Citing a retracted paper is legitimate when the document (a) discusses \
or debunks it, (b) gives historical context, or (c) cites a part unrelated to why it was \
retracted. It is a problem only when the document RELIES on the retracted finding as \
support for its own claims. Judge from the passage; if no passage is available, say the \
intent is unverifiable and treat it as a mild caution, not a condemnation.
3. An old citation base is a weak signal, not disqualifying — weigh it lightly.
4. Be calibrated: reserve very low scores for genuine problems (retraction, reliance on \
fraudulent work). A careful review that debunks bad science is TRUSTWORTHY (high score).

Return ONLY a JSON object, no prose outside it, with exactly these keys:
{
  "trust_score": <float 0..1>,
  "verdict": "trustworthy" | "use_with_caution" | "untrustworthy",
  "confidence": <float 0..1>,
  "reasoning": "<2-4 sentences citing the specific evidence>",
  "per_issue": [{"issue": "<short>", "assessment": "<how you judged it and why>"}],
  "recommendation": "<one sentence for a researcher deciding whether to rely on this>"
}"""


def _format_dossier(dossier: dict) -> str:
    doc = dossier.get("document", {})
    fnd = dossier.get("foundation", {})
    lines = ["EVIDENCE DOSSIER", "=" * 40, "DOCUMENT:"]
    lines.append(f"  Title: {doc.get('title','')}")
    lines.append(f"  Venue: {doc.get('venue','')} ({doc.get('year','')})   "
                 f"Cited by: {doc.get('cited_by_count','?')}   In DOAJ: {doc.get('in_doaj')}")
    if doc.get("abstract"):
        lines.append(f"  Abstract: {doc['abstract'][:800]}")
    lines.append(f"  This document retracted? {doc.get('self_retracted')} "
                 f"({doc.get('retraction_type') or 'n/a'})")
    if doc.get("retraction_notice_title"):
        lines.append(f"  Retraction notice: {doc['retraction_notice_title']}")

    lines.append("")
    lines.append("CITED-LITERATURE FOUNDATION:")
    lines.append(f"  References with DOI: {fnd.get('n_references_with_doi',0)} "
                 f"of {fnd.get('n_references',0)} total")
    lines.append(f"  Median reference age: {fnd.get('median_reference_age')} yr   "
                 f"% from last 10 yr: {fnd.get('pct_references_last_10yr')}")
    rr = fnd.get("retracted_references", [])
    lines.append(f"  RETRACTED works cited: {len(rr)}")
    for i, r in enumerate(rr, 1):
        lines.append(f"    [{i}] {r.get('title','(title unavailable)')[:90]}")
        lines.append(f"        DOI: {r.get('doi')}  ({r.get('retraction_type')})")
        passages = r.get("citation_passages", [])
        if passages:
            for p in passages:
                lines.append(f"        HOW IT IS CITED: \"{p}\"")
        else:
            lines.append("        HOW IT IS CITED: (citing passage unavailable — "
                         "document not open-access; intent unverifiable)")

    signals = dossier.get("signals", [])
    if signals:
        lines.append("")
        lines.append(f"AUTOMATED SIGNALS (facts, not verdicts): {', '.join(signals)}")
    return "\n".join(lines)


def _extract_json(text: str) -> Optional[dict]:
    """Strip qwen <think> blocks and pull the last balanced JSON object."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # find the outermost/last {...}
    depth = 0
    start = None
    candidates = []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    for cand in reversed(candidates):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def assess_document(
    dossier: dict,
    model: str = DEFAULT_MODEL,
    ollama_url: str = _OLLAMA_URL,
    timeout: float = 240.0,
) -> LLMTrustJudgment:
    """Hand the evidence dossier to a local LLM and parse its trust judgment."""
    prompt = f"{SYSTEM_INSTRUCTIONS}\n\n{_format_dossier(dossier)}\n\nJSON judgment:"

    try:
        r = requests.post(
            ollama_url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": False,           # ask Ollama to skip visible reasoning if supported
                "options": {"temperature": 0.2},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        raw = r.json().get("response", "")
    except Exception as e:
        logger.warning("Ollama call failed: %s", e)
        # graceful fallback: mirror the deterministic score, flag LLM unavailable
        det = dossier.get("deterministic_scores", {})
        return LLMTrustJudgment(
            trust_score=float(det.get("overall_trust", 0.5)),
            verdict="use_with_caution",
            confidence=0.2,
            reasoning=f"LLM unavailable ({e}); showing deterministic score only.",
            per_issue=[], recommendation="LLM judge offline — rely on the evidence dossier.",
            model=model, raw="",
        )

    parsed = _extract_json(raw)
    if not parsed:
        det = dossier.get("deterministic_scores", {})
        return LLMTrustJudgment(
            trust_score=float(det.get("overall_trust", 0.5)),
            verdict="use_with_caution", confidence=0.2,
            reasoning="LLM response could not be parsed as JSON; deterministic score shown.",
            per_issue=[], recommendation="Review the evidence dossier manually.",
            model=model, raw=raw[:2000],
        )

    try:
        score = float(parsed.get("trust_score", 0.5))
    except (TypeError, ValueError):
        score = 0.5
    score = min(1.0, max(0.0, score))

    # Enforce rule 1 deterministically: a retracted document cannot score high.
    if dossier.get("document", {}).get("self_retracted"):
        score = min(score, 0.15)

    return LLMTrustJudgment(
        trust_score=score,
        verdict=str(parsed.get("verdict", "use_with_caution")),
        confidence=min(1.0, max(0.0, float(parsed.get("confidence", 0.5) or 0.5))),
        reasoning=str(parsed.get("reasoning", "")).strip(),
        per_issue=parsed.get("per_issue", []) or [],
        recommendation=str(parsed.get("recommendation", "")).strip(),
        model=model,
        raw=raw[:2000],
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.document_trust import build_dossier

    doi = sys.argv[1] if len(sys.argv) > 1 else "10.1089/aid.2020.0095"
    print(f"Building dossier for {doi} ...")
    dossier = build_dossier(doi)
    print("Asking LLM to judge ...")
    j = assess_document(dossier)
    print(json.dumps(j.to_dict(), indent=2))
