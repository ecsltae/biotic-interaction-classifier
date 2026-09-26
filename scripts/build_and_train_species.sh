#!/bin/bash
# Build the species-corrected training set and train 3 seeds, in both the
# triple-query format (V3's) and the pair-query format (V4's winning format),
# so the label fix is tested against the best known architecture, not only the old one.
set -e
cd /home/egaillac/MetaP/classifier
PY=/home/egaillac/MetaP/MPvenv/bin/python3

$PY - <<'PYEOF'
import pandas as pd
tr=pd.read_csv('data/training/distill/v3_combined_train.csv')
rl=pd.read_csv('data/training/distill/species_relabel.csv')
flip=set(rl[rl.label_species==1].row.tolist())
tr=tr.reset_index().rename(columns={'index':'row'})
before=tr.label.mean()
tr.loc[tr.row.isin(flip),'label']=1
tr.drop(columns=['row']).to_csv('data/training/distill/v4_species_train.csv',index=False)
print(f"species-corrected set: {len(tr):,} rows")
print(f"  positive rate {before:.4f} -> {tr.label.mean():.4f}")
print(f"  labels flipped: {len(flip):,} ({len(flip)/len(tr):.1%})")
print(f"  relabel coverage: {len(rl):,}/31,764 NO rows judged")
PYEOF

for fmt in triple pair; do
  for s in 1 2 3; do
    gpu=$(( (s-1) % 3 ))
    CUDA_VISIBLE_DEVICES=$gpu $PY experiments/multitask/train_student.py \
      --data data/training/distill/v4_species_train.csv --seed $s \
      --out models/species_v4/${fmt}_s${s} > logs/species_v4_${fmt}_s${s}.log 2>&1 &
  done
  wait
  echo "trained $fmt seeds"
done
echo "SPECIES TRAINING COMPLETE"
