#!/usr/bin/env python3
"""
Score every row in a CSV with the BiomedBERT + FLAN-T5-base teacher ensemble.

Unlike teacher_scorer.py, no confidence filtering is applied — every row receives
a p_bert, p_t5, and p_ensemble (geometric mean) regardless of score.

Usage:
    python classifier/scripts/score_all.py \
        --input  classifier/data/training/v7_data.csv \
        --output classifier/data/training/v7_softlabels.csv \
        --gpu

    # Custom model paths (e.g. for Option D retrained teacher):
    python classifier/scripts/score_all.py \
        --input  classifier/data/training/distillation_44k.csv \
        --output classifier/data/training/distillation_v14teacher.csv \
        --bert-model classifier/models/transformer_BiomedBERT_v14 \
        --gpu
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parents[2]

BERT_DEFAULT  = ROOT / "classifier/models/transformer_BiomedBERT_cv_regularized"
T5_DEFAULT    = ROOT / "classifier/models/flan-t5-base_v12"
BATCH_SIZE    = 32
MAX_INPUT_LEN = 256

PROMPT_TEMPLATE = (
    "Does this sentence describe a biotic interaction between two organisms?\n"
    "Sentence: {sentence}\n"
    "Answer:"
)


def predict_seq_cls(model_path: Path, texts: list, device) -> np.ndarray:
    tok   = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True).to(device).eval()
    probs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t.lower() for t in texts[i:i + BATCH_SIZE]]
        enc = tok(batch, truncation=True, max_length=MAX_INPUT_LEN, padding=True,
                  return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(
                model(input_ids=enc["input_ids"],
                      attention_mask=enc["attention_mask"]).logits,
                -1)[:, 1].cpu().tolist()
        probs.extend(p)
        if (i // BATCH_SIZE) % 50 == 0:
            print(f"  bert  {i + len(batch)}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.array(probs)


def predict_flan_t5(model_path: Path, texts: list, device) -> np.ndarray:
    tok   = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True).to(device).eval()
    yes_id = tok.encode("yes", add_special_tokens=False)[0]
    no_id  = tok.encode("no",  add_special_tokens=False)[0]
    prompts = [PROMPT_TEMPLATE.format(sentence=t) for t in texts]
    scores = []
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i + BATCH_SIZE]
            enc = tok(batch, max_length=MAX_INPUT_LEN, padding=True,
                      truncation=True, return_tensors="pt").to(device)
            bos = torch.full((len(batch), 1), model.config.decoder_start_token_id,
                             dtype=torch.long).to(device)
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        decoder_input_ids=bos)
            logits = out.logits[:, 0, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            yes_lp = log_probs[:, yes_id].cpu().numpy()
            no_lp  = log_probs[:, no_id].cpu().numpy()
            prob_yes = np.exp(yes_lp) / (np.exp(yes_lp) + np.exp(no_lp))
            scores.extend(prob_yes.tolist())
            if (i // BATCH_SIZE) % 50 == 0:
                print(f"  t5    {i + len(batch)}/{len(texts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="Score all rows with teacher ensemble (no filtering)")
    parser.add_argument("--input",       required=True, help="Input CSV")
    parser.add_argument("--output",      required=True, help="Output CSV (all input rows + p_bert, p_t5, p_ensemble)")
    parser.add_argument("--text-col",    default="text", help="Column name for sentence text")
    parser.add_argument("--bert-model",  default=str(BERT_DEFAULT), help="Path to BiomedBERT model dir")
    parser.add_argument("--flant5-model", default=str(T5_DEFAULT),  help="Path to FLAN-T5 model dir")
    parser.add_argument("--gpu",         action="store_true", help="Use CUDA:0")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    in_path  = Path(args.input)
    out_path = Path(args.output)
    bert_path = Path(args.bert_model)
    t5_path   = Path(args.flant5_model)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        sys.exit(1)
    if not bert_path.exists():
        print(f"ERROR: BiomedBERT model not found: {bert_path}")
        sys.exit(1)
    if not t5_path.exists():
        print(f"ERROR: FLAN-T5 model not found: {t5_path}")
        sys.exit(1)

    df = pd.read_csv(in_path)
    texts = df[args.text_col].astype(str).tolist()
    print(f"Loaded {len(texts):,} rows from {in_path.name}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nScoring with BiomedBERT ({bert_path.name}) ...", flush=True)
    p_bert = predict_seq_cls(bert_path, texts, device)

    print(f"\nScoring with FLAN-T5 ({t5_path.name}) ...", flush=True)
    p_t5 = predict_flan_t5(t5_path, texts, device)

    p_ensemble = np.sqrt(p_bert * p_t5)

    df["p_bert"]     = p_bert
    df["p_t5"]       = p_t5
    df["p_ensemble"] = p_ensemble

    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows → {out_path}")
    print(f"p_ensemble  mean={p_ensemble.mean():.4f}  median={np.median(p_ensemble):.4f}")
    print(f"p_ensemble > 0.5: {(p_ensemble > 0.5).sum():,} ({100*(p_ensemble > 0.5).mean():.1f}%)")


if __name__ == "__main__":
    main()
