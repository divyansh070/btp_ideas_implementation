"""Single-head window model and mini Track B MoE (guide §6).

Encoder: 2-layer LSTM(64) -> 4-head self-attention (+residual, LayerNorm) -> z = [last step, mean] (128-d).
The guide's Neural ODE stage is omitted in this proof of concept; z is still 128-d.
"""
import torch
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


def head(d_in=128, out="sigmoid"):
    """128 -> 64 -> 32 -> 1, LayerNorm (never BatchNorm), dropout 0.1."""
    act = nn.Sigmoid() if out == "sigmoid" else nn.Softplus()
    return nn.Sequential(nn.Linear(d_in, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(32, 1), act)


class SingleHead(nn.Module):
    def __init__(self, n_feat, out="sigmoid"):
        super().__init__()
        self.enc, self.head = Encoder(n_feat), head(out=out)

    def forward(self, x, *_):
        return self.head(self.enc(x)[0]).squeeze(-1)


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
        logits = self.router(torch.cat([self.text(text), dyn, disp], 1))
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
