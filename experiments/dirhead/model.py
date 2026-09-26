"""Cross-encoder with a symmetric binary head and a structurally antisymmetric direction head.

Input format (`mark_canon`, already in scripts/xenc_format.py):
    segment A : the two taxa sorted alphabetically, "a [SEP] b"
    segment B : the passage with `a` wrapped in @ ... @ and `b` in # ... #

Because segment A carries no argument order, the BINARY question ("do these two taxa
interact?") is exactly order-invariant at the input, which is the semantics the species-level
label actually has.  All the order sensitivity lives in the direction head, which reads the
two marked span representations and is odd under swapping them by construction:

    s = u.g(hA, hB, hc) - u.g(hB, hA, hc)          FORWARD/REVERSE logit  (antisymmetric)
    n = v.m(hc, hA+hB, hA*hB)                      NO-DIRECTION logit      (symmetric)
    logits = [s, -s, n]

Swapping the two spans maps s -> -s and leaves n alone, so P(A is subject) and
P(B is subject) exchange exactly and P(no direction) is unchanged.  There is no loss
weight or gradient gate holding the symmetry in place; it is a property of the algebra.
"""
import torch, torch.nn as nn
from transformers import AutoModel, AutoConfig


def masked_mean(h, m):
    m = m.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)


class DirModel(nn.Module):
    def __init__(self, encoder, dir_hidden=256, n_dir=3, dropout=0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(encoder)
        d = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        # binary head: symmetric features only
        self.bin = nn.Sequential(nn.Linear(3 * d, d), nn.Tanh(), nn.Linear(d, 2))
        # direction head: shared scorer applied to both orderings, then subtracted
        self.g = nn.Sequential(nn.Linear(3 * d, dir_hidden), nn.Tanh(), nn.Linear(dir_hidden, 1))
        # no-direction head: symmetric
        self.nod = nn.Sequential(nn.Linear(3 * d, dir_hidden), nn.Tanh(), nn.Linear(dir_hidden, 1))
        self.n_dir = n_dir

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                mask_a=None, mask_b=None, detach_dir=False):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        h = self.enc(**kw).last_hidden_state
        hc = h[:, 0]
        hA = masked_mean(h, mask_a)
        hB = masked_mean(h, mask_b)
        hc, hA, hB = self.drop(hc), self.drop(hA), self.drop(hB)
        sym = torch.cat([hc, hA + hB, hA * hB], -1)
        bin_logits = self.bin(sym)
        hd = (hc.detach(), hA.detach(), hB.detach()) if detach_dir else (hc, hA, hB)
        c, a, b = hd
        symd = torch.cat([c, a + b, a * b], -1)
        s = (self.g(torch.cat([a, b, c], -1)) - self.g(torch.cat([b, a, c], -1))).squeeze(-1)
        n = self.nod(symd).squeeze(-1)
        dir_logits = torch.stack([s, -s, n], -1) if self.n_dir == 3 \
            else torch.stack([s, -s], -1)
        return bin_logits, dir_logits
