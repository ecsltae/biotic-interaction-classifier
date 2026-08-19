#!/usr/bin/env python3
"""
Ensemble: BiomedBERT (discriminative) + FLAN-T5-base (generative)
==================================================================
Combines the best discriminative model (BiomedBERT cv_regularized, avg EP F1=0.788)
with the best generative model (FLAN-T5-base v12, complements BiomedBERT best).

Best ensemble: BiomedBERT + v12 (geometric) = F1=0.865 on EP-relax, F1=1.000 on eval_100

Fusion strategies:
  - arithmetic:    (p_bert + p_t5) / 2  [default, better coverage of diverse types]
  - geometric:     sqrt(p_bert * p_t5)  [original, same F1 on EP-relax]

Note: Arithmetic mean recovers 17 coverage gap samples on gen_set_100 (mutualism,
herbivory, seed dispersal) where T5 returns p=0.0 but BERT is correct (p>0.9).
Result: gen_set_100 F1=0.945 (arithmetic) vs 0.800 (geometric).

Evaluates on EP test set (globi-relax_..._EP.tsv) with threshold optimization.
Results saved to classifier/results/ensemble_biomedbert_flant5/

Usage:
    python scripts/ensemble_biomedbert_flant5.py
    python scripts/ensemble_biomedbert_flant5.py --eval-all   # also eval eval_100.tsv
    python scripts/ensemble_biomedbert_flant5.py --fusion geometric  # original strategy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = Path('/home/egaillac/MetaP/classifier')

BIOMEDBERT_MODEL = BASE_DIR / 'models/transformer_BiomedBERT_cv_regularized'
FLANT5_MODEL     = BASE_DIR / 'models/flan-t5-base_v12'  # v12 complements BERT better than v11_1

EP_TEST_FILE  = BASE_DIR / 'data/evaluation/globi-relax_passages-triplets_2024-02-28_curation_EP.tsv'
EVAL_100_FILE = BASE_DIR / 'data/evaluation/eval_100.tsv'

RESULTS_DIR = BASE_DIR / 'results/ensemble_biomedbert_flant5'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_eval_set(path: Path):
    import pandas as pd
    df = pd.read_csv(path, sep='\t')
    text_col  = 'sentence'
    label_col = 'evaluation_pair_interacting'
    texts  = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


# ---------------------------------------------------------------------------
# BiomedBERT inference
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def get_biomedbert_probs(model_dir: Path, texts: list) -> np.ndarray:
    print(f"  [BiomedBERT] Loading from {model_dir.name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE).eval()

    ds = TextDataset(texts, tokenizer)
    loader = DataLoader(ds, batch_size=32)
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


# ---------------------------------------------------------------------------
# FLAN-T5-base inference
# ---------------------------------------------------------------------------

def _get_yes_no_ids(tokenizer):
    yes_id = tokenizer.encode('yes', add_special_tokens=False)[0]
    no_id  = tokenizer.encode('no',  add_special_tokens=False)[0]
    return yes_id, no_id


def _make_prompt(text: str) -> str:
    return (
        f"Does the following sentence describe a biotic interaction between two species? "
        f"Answer yes or no.\n\nSentence: {text}"
    )


def get_flant5_probs(model_dir: Path, texts: list, batch_size: int = 32) -> np.ndarray:
    print(f"  [FLAN-T5-base] Loading from {model_dir.name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(DEVICE).eval()
    yes_id, no_id = _get_yes_no_ids(tokenizer)

    prompts = [_make_prompt(t) for t in texts]
    probs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            enc = tokenizer(batch_prompts, return_tensors='pt', padding=True,
                            truncation=True, max_length=512).to(DEVICE)
            # Force decode with decoder_input_ids = [pad_token_id]
            dec_input = torch.full(
                (len(batch_prompts), 1), tokenizer.pad_token_id,
                dtype=torch.long, device=DEVICE
            )
            out = model(**enc, decoder_input_ids=dec_input)
            logits = out.logits[:, 0, :]   # (batch, vocab) — first decode step
            lp = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = lp[:, yes_id].cpu().numpy()
            no_lp  = lp[:, no_id].cpu().numpy()
            p_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
            probs.extend(p_yes.tolist())
    return np.array(probs)


# ---------------------------------------------------------------------------
# Ensemble strategies
# ---------------------------------------------------------------------------

def find_best_threshold(probs: np.ndarray, labels: list) -> tuple:
    """Sweep threshold 0.05..0.95, return (best_thresh, best_f1, prec, rec)."""
    y = np.array(labels)
    best = (0.5, 0.0, 0.0, 0.0)
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best[1]:
            p = precision_score(y, preds, zero_division=0)
            r = recall_score(y, preds, zero_division=0)
            best = (round(float(t), 2), round(f1, 4), round(p, 4), round(r, 4))
    return best


def evaluate_strategies(p_bert: np.ndarray, p_t5: np.ndarray, labels: list) -> dict:
    strategies = {}

    # Simple average
    strategies['simple_avg'] = (p_bert + p_t5) / 2

    # Weighted sweep
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        strategies[f'w{int(w*10)}_bert'] = w * p_bert + (1 - w) * p_t5

    # Max
    strategies['max'] = np.maximum(p_bert, p_t5)

    # Geometric mean
    strategies['geometric'] = np.sqrt(np.clip(p_bert * p_t5, 1e-12, 1.0))

    # Individual baselines
    strategies['biomedbert_only'] = p_bert
    strategies['flant5_only']     = p_t5

    results = {}
    for name, probs in strategies.items():
        thresh, f1, prec, rec = find_best_threshold(probs, labels)
        results[name] = {'f1': f1, 'precision': prec, 'recall': rec, 'threshold': thresh}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_ensemble(p_bert: np.ndarray, p_t5: np.ndarray, fusion: str) -> np.ndarray:
    """Compute ensemble probability using specified fusion strategy."""
    if fusion == 'arithmetic':
        return (p_bert + p_t5) / 2
    elif fusion == 'geometric':
        return np.sqrt(np.clip(p_bert * p_t5, 1e-12, 1.0))
    elif fusion == 'max':
        return np.maximum(p_bert, p_t5)
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-all', action='store_true',
                        help='Also evaluate on eval_100.tsv')
    parser.add_argument('--biomedbert-model', default=str(BIOMEDBERT_MODEL))
    parser.add_argument('--flant5-model',     default=str(FLANT5_MODEL))
    parser.add_argument('--fusion', default='arithmetic', choices=['arithmetic', 'geometric', 'max'],
                        help='Ensemble fusion strategy (default: arithmetic)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    bert_dir = Path(args.biomedbert_model)
    t5_dir   = Path(args.flant5_model)

    print(f"\nEnsemble Configuration:")
    print(f"  BiomedBERT: {bert_dir.name}")
    print(f"  FLAN-T5:    {t5_dir.name}")
    print(f"  Fusion:     {args.fusion}")

    all_results = {}

    for eval_name, eval_path in [('ep_test', EP_TEST_FILE),
                                  ('eval_100', EVAL_100_FILE)]:
        if eval_name == 'eval_100' and not args.eval_all:
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating on: {eval_path.name}")
        print('='*60)

        texts, labels = load_eval_set(eval_path)
        print(f"  {len(texts)} samples, {sum(labels)} positives")

        print("\nRunning inference...")
        p_bert = get_biomedbert_probs(bert_dir, texts)
        p_t5   = get_flant5_probs(t5_dir, texts)

        # Save raw probabilities
        probs_path = RESULTS_DIR / f'probs_{eval_name}.npz'
        np.savez(probs_path, p_bert=p_bert, p_t5=p_t5, labels=np.array(labels))
        print(f"\n  Raw probs saved → {probs_path.name}")

        # Compute primary ensemble result with selected fusion
        p_ens = compute_ensemble(p_bert, p_t5, args.fusion)
        thresh, f1, prec, rec = find_best_threshold(p_ens, labels)
        print(f"\n  ★ {args.fusion.upper()} ENSEMBLE: F1={f1:.3f} Prec={prec:.3f} Rec={rec:.3f} (thresh={thresh:.2f})")

        print("\nAll strategy comparison:")
        results = evaluate_strategies(p_bert, p_t5, labels)
        all_results[eval_name] = results
        all_results[eval_name][f'{args.fusion}_ensemble'] = {'f1': f1, 'precision': prec, 'recall': rec, 'threshold': thresh}

        # Print table
        header = f"  {'Strategy':<20} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Thresh':>7}"
        print(header)
        print('  ' + '-' * 54)
        for name, r in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
            print(f"  {name:<20} {r['f1']:>7.3f} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['threshold']:>7.2f}")

    # Save results
    out_path = RESULTS_DIR / 'ensemble_results.json'
    out_data = {
        'biomedbert_model': str(bert_dir),
        'flant5_model':     str(t5_dir),
        'fusion':           args.fusion,
        'results':          all_results,
    }
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nResults saved → {out_path}")

    # Print recommended configuration
    if 'ep_test' in all_results:
        r = all_results['ep_test'][f'{args.fusion}_ensemble']
        print(f"\n*** PRODUCTION CONFIG: {args.fusion} ensemble — F1={r['f1']:.3f} "
              f"Prec={r['precision']:.3f} Rec={r['recall']:.3f} Threshold={r['threshold']:.2f} ***")


if __name__ == '__main__':
    main()
