"""Single-head window model and mini Track B MoE (guide §6).

Encoder: 2-layer LSTM(64) -> 4-head self-attention (+residual, LayerNorm) -> z = [last step, mean] (128-d).
The guide's Neural ODE stage is omitted in this proof of concept; z is still 128-d.
"""
import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, n_feat, hidden=64, heads=4):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        h, _ = self.lstm(x)
        a, _ = self.attn(h, h, h)
        a = self.ln(h + a)
        z = torch.cat([a[:, -1], a.mean(1)], 1)
        disp = (h[:, -1] - h[:, 0]).norm(dim=1, keepdim=True)  # stands in for the guide's Neural-ODE displacement
        return z, disp


def head(d_in=128, out="sigmoid", h1=64, h2=32):
    """128 -> 64 -> 32 -> 1, LayerNorm (never BatchNorm), dropout 0.1. h1/h2 only change for E3 v2's matched head."""
    act = nn.Sigmoid() if out == "sigmoid" else nn.Softplus()
    return nn.Sequential(nn.Linear(d_in, h1), nn.LayerNorm(h1), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(h1, h2), nn.LayerNorm(h2), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(h2, 1), act)


class SingleHead(nn.Module):
    """E2 model. E5 adds an optional extra input: aux (+ latent displacement) -> Linear + LeakyReLU -> 32, concatenated
    to z before the head. With d_aux=0 the modules (and their init order) are exactly E2's."""

    def __init__(self, n_feat, out="sigmoid", d_aux=0, aux_disp=False, branch=32, width=64):
        super().__init__()
        self.enc = Encoder(n_feat)
        self.head = head(d_in=128 + (branch if d_aux else 0), out=out, h1=width, h2=max(width // 2, 1))
        self.aux = nn.Sequential(nn.Linear(d_aux + int(aux_disp), branch), nn.LeakyReLU()) if d_aux else None
        self.aux_disp = aux_disp

    def forward(self, x, aux=None):
        z, disp = self.enc(x)
        if self.aux is not None:
            a = torch.cat([aux, disp], 1) if self.aux_disp else aux
            z = torch.cat([z, self.aux(a)], 1)
        return self.head(z).squeeze(-1)


class MoE(nn.Module):
    """E small experts + always-on shared expert; pred = (1-beta) * sum_e g_e f_e(z) + beta * f_shared(z)."""

    def __init__(self, n_feat, d_text, n_dyn, n_experts, bias):
        super().__init__()
        self.enc = Encoder(n_feat)
        self.experts = nn.ModuleList(head() for _ in range(n_experts))
        self.shared = head()
        self.text = nn.Sequential(nn.Linear(d_text, 64), nn.LeakyReLU())
        self.router = nn.Sequential(nn.Linear(64 + n_dyn + 1, 32), nn.LeakyReLU(), nn.Linear(32, n_experts))
        self.beta_logit = nn.Parameter(torch.zeros(1))
        self.bias = bias
        self.E = n_experts

    def forward(self, x, dyn, text, match, tau, mode):
        z, disp = self.enc(x)
        f = torch.cat([e(z) for e in self.experts], 1)
        fs = self.shared(z).squeeze(-1)
        # unit-length text embedding (run 2): raw Llama vectors are unscaled, the suspected cause of near one-hot gates in run 1
        logits = self.router(torch.cat([self.text(F.normalize(text, dim=1)), dyn, disp], 1))
        if mode == "uniform":       # Stage A: router frozen at uniform weights
            g = torch.full_like(f, 1 / self.E)
        elif mode == "bias":        # ours: additive category bias, softmax over all experts
            g = torch.softmax(logits / tau + self.bias * match, 1)
        elif mode == "mask":        # PBT-style: softmax, zero non-matching, renormalise (models/PBT.py:156-172)
            p = torch.softmax(logits / tau, 1) * match
            g = p / (p.sum(1, keepdim=True) + 1e-9)
        else:
            raise ValueError(mode)
        beta = torch.sigmoid(self.beta_logit)
        pred = (1 - beta) * (g * f).sum(1) + beta * fs
        if mode == "mask":          # no matching specialist: shared expert alone (PBT: general experts only)
            pred = torch.where(match.sum(1) == 0, fs, pred)
        return dict(pred=pred, g=g, f=f, fs=fs, z=z, logits=logits, beta=beta)

    def add_expert(self, init):
        """E3 v2 few-shot: spawn one expert for the held-out chemistry, a copy of the shared expert (init="shared")
        or of expert `init` (an index). The router's output layer gains one row: a copy of expert `init`'s row, or
        the mean row for "shared"; existing experts' logits are unchanged."""
        import copy
        self.experts.append(copy.deepcopy(self.shared if init == "shared" else self.experts[init]))
        old = self.router[-1]
        new = nn.Linear(old.in_features, old.out_features + 1).to(old.weight.device)
        with torch.no_grad():
            new.weight[:-1], new.bias[:-1] = old.weight, old.bias
            new.weight[-1] = old.weight.mean(0) if init == "shared" else old.weight[init]
            new.bias[-1] = old.bias.mean() if init == "shared" else old.bias[init]
        self.router[-1] = new
        self.E += 1
