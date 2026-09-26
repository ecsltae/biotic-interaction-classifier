"""Datasets for the joint binary + direction cross-encoder."""
import numpy as np, torch
from torch.utils.data import Dataset
import fmt


class Joint(Dataset):
    """rows: dict-of-lists with s1, s2, rel, text, y_bin (-100 = none), y_dir (-100 = none).

    y_dir is expressed over the CANONICAL order: 0 = the alphabetically-first taxon is the
    subject, 1 = the second is, 2 = the relation has no direction.
    """

    def __init__(self, df, tok, max_len=256, augment=False, names_only=False,
                 mask_relation=False, anon=False, seed=0):
        self.df = df.reset_index(drop=True)
        self.tok, self.ml = tok, max_len
        self.augment, self.names_only, self.mask_relation = augment, names_only, mask_relation
        self.anon = anon
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        order = "canon"
        yd = int(r.y_dir)
        if self.augment and self.rng.rand() < 0.5:
            order = "swap"
            if yd in (0, 1):
                yd = 1 - yd
        sa, sb, pa, pb, _, _, ok = fmt.encode(
            r.s1, r.s2, r.text, order=order, names_only=self.names_only, anon=self.anon,
            mask_relation=(r.rel if self.mask_relation else None))
        e = self.tok(sa, sb, truncation="only_second", max_length=self.ml,
                     return_offsets_mapping=True)
        sids = e.sequence_ids()
        off = e["offset_mapping"]
        ma = np.zeros(len(off), dtype=np.int8)
        mb = np.zeros(len(off), dtype=np.int8)
        for j, (s, (st, en)) in enumerate(zip(sids, off)):
            if s != 1 or en <= st:
                continue
            if pa and st >= pa[0] and en <= pa[1]:
                ma[j] = 1
            if pb and st >= pb[0] and en <= pb[1]:
                mb[j] = 1
        if ma.sum() == 0 or mb.sum() == 0:
            ok = False
            if ma.sum() == 0:
                ma[:] = [1 if s == 1 else 0 for s in sids]
            if mb.sum() == 0:
                mb[:] = [1 if s == 1 else 0 for s in sids]
            yd = -100                       # never train direction on a row we cannot mark
        return {k: e[k] for k in ("input_ids", "token_type_ids", "attention_mask") if k in e} | {
            "mask_a": ma.tolist(), "mask_b": mb.tolist(),
            "y_bin": int(r.y_bin), "y_dir": int(yd), "ok": int(ok)}


def collate(tok):
    def f(batch):
        yb = torch.tensor([b.pop("y_bin") for b in batch])
        yd = torch.tensor([b.pop("y_dir") for b in batch])
        ok = torch.tensor([b.pop("ok") for b in batch])
        ma = [b.pop("mask_a") for b in batch]
        mb = [b.pop("mask_b") for b in batch]
        enc = tok.pad(batch, padding=True, return_tensors="pt")
        L = enc["input_ids"].shape[1]
        def pad(ms):
            return torch.tensor([m + [0] * (L - len(m)) for m in ms], dtype=torch.float)
        return dict(enc) | {"mask_a": pad(ma), "mask_b": pad(mb),
                            "y_bin": yb, "y_dir": yd, "ok": ok}
    return f
