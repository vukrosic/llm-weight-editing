"""The small language model under edit: RoPE, QK-norm, SwiGLU, RMSNorm, no biases. Tensor names are what the editor sees."""
import os
import torch, torch.nn as nn, torch.nn.functional as F

DEV, FULL = os.environ.get("WE_DEV", "cpu"), os.environ.get("WE_FULL") == "1"  # default: light on a laptop, one low-priority CPU thread
if DEV == "cpu": torch.set_num_threads(int(os.environ.get("WE_THREADS", 1))); os.nice(19)


class RMSNorm(nn.Module):
    def __init__(s, d):
        super().__init__(); s.scale = nn.Parameter(torch.ones(d))

    def forward(s, x):
        return x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * s.scale


def rotate(x, base=10000.0):
    """RoPE over the last dimension of (batch, heads, length, head width)."""
    d, L = x.shape[-1], x.shape[-2]
    f = 1.0 / base ** (torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    a = torch.outer(torch.arange(L, device=x.device, dtype=x.dtype), f)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * a.cos() - x2 * a.sin(), x1 * a.sin() + x2 * a.cos()], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s, w, heads):
        super().__init__(); s.heads = heads; h = w // heads * 8 // 3 * heads
        s.norm1, s.norm2 = RMSNorm(w), RMSNorm(w)
        s.q, s.k, s.v, s.attn_out = (nn.Linear(w, w, bias=False) for _ in range(4))
        s.mlp_gate, s.mlp_up, s.mlp_out = nn.Linear(w, h, bias=False), nn.Linear(w, h, bias=False), nn.Linear(h, w, bias=False)

    def forward(s, x):
        B, L, W = x.shape; n = s.norm1(x)
        split = lambda t: t.view(B, L, s.heads, W // s.heads).transpose(1, 2)
        q, k, v = F.normalize(rotate(split(s.q(n))), dim=-1), F.normalize(rotate(split(s.k(n))), dim=-1), split(s.v(n))
        a = (q @ k.transpose(-1, -2)) * (W // s.heads) ** .5
        a = a.masked_fill(torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1), -1e9).softmax(-1)
        x = x + s.attn_out((a @ v).transpose(1, 2).reshape(B, L, W))
        n = s.norm2(x)
        return x + s.mlp_out(F.silu(s.mlp_gate(n)) * s.mlp_up(n))


class LM(nn.Module):
    def __init__(s, V, width, layers, heads):
        super().__init__(); s.cfg = dict(V=V, width=width, layers=layers, heads=heads)
        s.embed = nn.Embedding(V, width); s.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        s.final_norm = RMSNorm(width); s.head = nn.Linear(width, V, bias=False)

    def forward(s, x, capture=None):
        h = s.embed(x)
        for i, b in enumerate(s.blocks):
            h = b(h)
            if capture is not None: capture[f"resid.{i}"] = h.detach()
        return s.head(s.final_norm(h))


@torch.no_grad()
def generate(m, prompts, n_new, temperature, top_k, g, device, ban=None):
    """prompts: (n, L) long tensor, all the same length. Sampling draws from a CPU generator, so it is the same on any device. ban is a token that is never sampled."""
    x = prompts.to(device); m.eval()
    for _ in range(n_new):
        l = m(x[:, -128:])[:, -1].cpu().float() / temperature
        if ban is not None: l[:, ban] = -1e9
        kth = l.topk(top_k, -1).values[:, -1:]
        x = torch.cat([x, l.masked_fill(l < kth, -1e9).softmax(-1).multinomial(1, generator=g).to(device)], 1)
    return x[:, prompts.shape[1]:].cpu()


@torch.no_grad()
def row_nll(m, rows, device, batch=128):
    """Per-row mean next-token loss. rows: (n, L) long tensor."""
    m.eval(); out = []
    for i in range(0, len(rows), batch):
        x = rows[i:i + batch].to(device)
        out.append(F.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(len(x), -1).mean(1).cpu())
    return torch.cat(out)
