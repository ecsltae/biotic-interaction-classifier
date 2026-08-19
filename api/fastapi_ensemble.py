#!/usr/bin/env python3
"""
FastAPI for Biotic Interaction Classifier
BiomedBERT (70%) + RoBERTa (30%) Ensemble
Supports batch predictions
"""

import torch
import numpy as np
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn


def preprocess_text(text: str) -> str:
    """
    Preprocess input text for consistent predictions.
    - Lowercase (BiomedBERT is uncased, RoBERTa works better with consistent casing)
    - Normalize whitespace
    - Strip leading/trailing whitespace
    """
    # Lowercase
    text = text.lower()
    # Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text)
    # Strip
    text = text.strip()
    return text

# Configuration - BiomedBERT gets more weight as domain-specific model
CONFIG = {
    'models': {
        'biomedbert': {
            'path': '/home/egaillac/MetaP/classifier/models/precision_ensemble/biomedbert_precision',
            'weight': 0.70  # Higher weight for domain-specific model
        },
        'roberta': {
            'path': '/home/egaillac/MetaP/classifier/models/precision_ensemble/roberta_precision',
            'weight': 0.30
        }
    },
    'threshold': 0.5,
    'max_length': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Pydantic models
class SinglePredictionRequest(BaseModel):
    text: str

class BatchPredictionRequest(BaseModel):
    sentences: List[str]

class SinglePredictionResponse(BaseModel):
    text: str
    prediction: int
    probability: float
    label: str

class BatchPredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    labels: List[str]

# Initialize FastAPI
app = FastAPI(
    title="Biotic Interaction Classifier API",
    description="BiomedBERT (70%) + RoBERTa (30%) ensemble for detecting biotic interactions in text",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
models = {}
tokenizers = {}

def load_models():
    """Load all ensemble models"""
    global models, tokenizers

    device = CONFIG['device']
    print(f"Loading models on {device}...")

    for name, config in CONFIG['models'].items():
        print(f"  Loading {name} (weight: {config['weight']})...")
        tokenizers[name] = AutoTokenizer.from_pretrained(config['path'])
        models[name] = AutoModelForSequenceClassification.from_pretrained(config['path'])
        models[name].to(device)
        models[name].eval()

    print("All models loaded!")

def predict_batch(sentences: List[str]) -> tuple:
    """
    Predict for a batch of sentences using weighted ensemble
    Returns: (predictions, probabilities)
    """
    device = CONFIG['device']
    all_probs = []

    # Preprocess all sentences for consistent predictions
    sentences = [preprocess_text(s) for s in sentences]

    # Get predictions from each model
    for name, model in models.items():
        tokenizer = tokenizers[name]
        weight = CONFIG['models'][name]['weight']

        # Tokenize batch
        inputs = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=CONFIG['max_length'],
            return_tensors='pt'
        ).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

        all_probs.append(probs * weight)

    # Weighted average
    ensemble_probs = np.sum(all_probs, axis=0)
    predictions = (ensemble_probs >= CONFIG['threshold']).astype(int)

    return predictions.tolist(), ensemble_probs.tolist()

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {
        "message": "Biotic Interaction Classifier API",
        "endpoints": {
            "/predict": "POST - Single sentence prediction",
            "/predict_batch": "POST - Batch prediction (list of sentences)",
            "/health": "GET - Health check"
        },
        "ensemble": {
            "biomedbert": "70%",
            "roberta": "30%"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": CONFIG['device'],
        "models_loaded": list(models.keys()),
        "weights": {name: cfg['weight'] for name, cfg in CONFIG['models'].items()}
    }

@app.post("/predict", response_model=SinglePredictionResponse)
async def predict_single(request: SinglePredictionRequest):
    """Predict for a single sentence"""
    try:
        predictions, probabilities = predict_batch([request.text])
        pred = predictions[0]
        prob = probabilities[0]

        return SinglePredictionResponse(
            text=request.text,
            prediction=pred,
            probability=round(prob, 4),
            label="interaction" if pred == 1 else "no_interaction"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch_endpoint(request: BatchPredictionRequest):
    """
    Predict for a batch of sentences
    Input: {"sentences": ["sentence1", "sentence2", ...]}
    Output: {"predictions": [0, 1, ...], "probabilities": [0.2, 0.8, ...], "labels": ["no_interaction", "interaction", ...]}
    """
    try:
        if not request.sentences:
            raise HTTPException(status_code=400, detail="Empty sentence list")

        predictions, probabilities = predict_batch(request.sentences)
        labels = ["interaction" if p == 1 else "no_interaction" for p in predictions]

        return BatchPredictionResponse(
            predictions=predictions,
            probabilities=[round(p, 4) for p in probabilities],
            labels=labels
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Trust scoring endpoint
# ---------------------------------------------------------------------------

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.trust_scorer import TrustScorer as _TrustScorer

_trust_scorer: "_TrustScorer | None" = None

def _get_trust_scorer() -> "_TrustScorer":
    global _trust_scorer
    if _trust_scorer is None:
        _trust_scorer = _TrustScorer()
    return _trust_scorer


class TrustScoreRequest(BaseModel):
    text: str
    species1: str = ""
    species2: str = ""
    interaction_type: str = ""
    pub_year: int | None = None


@app.post("/trust_score")
async def trust_score(request: TrustScoreRequest):
    """Score the credibility/trust of an interaction claim.

    Returns a composite trust score (0–1) decomposed into:
    temporal decay, evidence type, contradiction risk, and GloBI confirmation.
    """
    try:
        scorer = _get_trust_scorer()
        result = scorer.score(
            text=request.text,
            species1=request.species1,
            species2=request.species2,
            interaction_type=request.interaction_type,
            pub_year=request.pub_year,
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /kg endpoint ──────────────────────────────────────────────────────────────

_kg_builder_mod = None

def _get_kg_builder():
    global _kg_builder_mod
    if _kg_builder_mod is None:
        import importlib.util, sys
        kg_path = Path(__file__).resolve().parents[1] / "experiments/knowledge_graph/kg_builder.py"
        spec = importlib.util.spec_from_file_location("kg_builder", kg_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _kg_builder_mod = mod
    return _kg_builder_mod


class KGRequest(BaseModel):
    sentences: List[str]
    doi: str = ""
    entities: List[List[dict]] = []   # optional pre-extracted entities per sentence


@app.post("/kg")
async def build_kg(request: KGRequest):
    """
    Build typed directed KG edges from a list of sentences.

    For each sentence: runs binary classification, then infers typed directed/undirected
    KG edge using ROBI heuristics + robiext_v2025 ontology.

    Returns list of edges: subject, predicate, object, directed, ro_id, confidence, flag.
    """
    try:
        mod = _get_kg_builder()
        kg  = mod.KnowledgeGraph()

        for i, sentence in enumerate(request.sentences):
            preds, probs = predict_batch([sentence])
            confidence = float(probs[0])

            # Use pre-supplied entities if provided, else empty (NER not available in port 8001)
            ents = request.entities[i] if i < len(request.entities) else []

            kg.add_from_api_response({
                "entities":         ents,
                "interaction_type": "",   # no interaction_type at port 8001; enrich at port 8002
                "confidence":       confidence,
                "sentence":         sentence,
                "doi":              request.doi,
            })

        kg.merge_edges()
        edges = []
        for e in kg.edges:
            edges.append({
                "subject":         e.subject.text or e.subject.normalized,
                "subject_type":    e.subject.entity_type,
                "predicate":       e.predicate,
                "object":          e.object.text or e.object.normalized,
                "object_type":     e.object.entity_type,
                "directed":        e.directed,
                "ro_id":           e.ro_id,
                "confidence":      round(e.confidence, 4),
                "n_sources":       e.n_sources,
                "flag":            e.flag,
            })
        return {"edges": edges, "stats": kg.stats()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import socket

    # Get the machine's IP
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    print(f"Starting API on http://{ip}:8000")
    print("Endpoints:")
    print("  POST /predict - Single sentence")
    print("  POST /predict_batch - Batch of sentences")
    print("  GET /health - Health check")

    uvicorn.run(app, host="0.0.0.0", port=8000)
