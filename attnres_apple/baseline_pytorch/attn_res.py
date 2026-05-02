"""
Attention Residuals (AttnRes) — Core Modules (v3)
===================================================
Reproduction of "Attention Residuals" (Kimi Team, arXiv:2603.15031)
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learnable scale."""
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, dim: int, ff_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, ff_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def depth_attention(
    sources: torch.Tensor,
    pseudo_query: torch.Tensor,
    key_norm: RMSNorm,
) -> torch.Tensor:
    """
    Core AttnRes operation: Eq. 2-4.
    sources: [N_src, B, T, D], pseudo_query: [D], key_norm: RMSNorm
    Returns: [B, T, D]
    """
    V = sources
    K = key_norm(V)
    logits = torch.einsum('d, n b t d -> n b t', pseudo_query, K)
    weights = logits.softmax(dim=0)
    h = torch.einsum('n b t, n b t d -> b t d', weights, V)
    return h


class FullAttnResLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.attn_res_query = nn.Parameter(torch.zeros(dim))
        self.mlp_res_query = nn.Parameter(torch.zeros(dim))
        self.attn_res_norm = RMSNorm(dim)
        self.mlp_res_norm = RMSNorm(dim)
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.mlp = SwiGLUMLP(dim, ff_dim, dropout=dropout)

    def forward(self, sources: list[torch.Tensor],
                attn_mask: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        stacked = torch.stack(sources)
        h = depth_attention(stacked, self.attn_res_query, self.attn_res_norm)
        h_normed = self.attn_norm(h)
        if attn_mask is None:
            T = h.size(1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        attn_out, _ = self.attn(h_normed, h_normed, h_normed,
                                attn_mask=attn_mask, is_causal=True)
        sources = sources + [attn_out]
        stacked = torch.stack(sources)
        h = depth_attention(stacked, self.mlp_res_query, self.mlp_res_norm)
        mlp_out = self.mlp(self.mlp_norm(h))
        sources = sources + [mlp_out]
        return sources


class BlockAttnResLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int,
                 layer_idx: int, block_size: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.attn_res_query = nn.Parameter(torch.zeros(dim))
        self.mlp_res_query = nn.Parameter(torch.zeros(dim))
        self.attn_res_norm = RMSNorm(dim)
        self.mlp_res_norm = RMSNorm(dim)
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.mlp = SwiGLUMLP(dim, ff_dim, dropout=dropout)

    @staticmethod
    def _block_depth_attention(blocks, partial_block, pseudo_query, key_norm):
        if partial_block is None:
            sources = blocks
        else:
            sources = blocks + [partial_block]
        V = torch.stack(sources)
        return depth_attention(V, pseudo_query, key_norm)

    def forward(self, blocks: list[torch.Tensor], partial_block: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):
        h = self._block_depth_attention(
            blocks, partial_block, self.attn_res_query, self.attn_res_norm)

        layer_number = self.layer_idx + 1
        layers_per_block = self.block_size // 2
        if layer_number % layers_per_block == 0:
            blocks = blocks + [partial_block]
            partial_block = None

        h_normed = self.attn_norm(h)
        if attn_mask is None:
            T = h.size(1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(T, device=h.device)
        attn_out, _ = self.attn(h_normed, h_normed, h_normed,
                                attn_mask=attn_mask, is_causal=True)
        if partial_block is not None:
            partial_block = partial_block + attn_out
        else:
            partial_block = attn_out

        h = self._block_depth_attention(
            blocks, partial_block, self.mlp_res_query, self.mlp_res_norm)
        mlp_out = self.mlp(self.mlp_norm(h))
        partial_block = partial_block + mlp_out
        return blocks, partial_block


class StandardTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.mlp = SwiGLUMLP(dim, ff_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.attn_norm(x)
        if attn_mask is None:
            T = x.size(1)
            attn_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, is_causal=True)
        x = x + attn_out
        h = self.mlp_norm(x)
        x = x + self.mlp(h)
        return x


class FullAttnResTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, ff_dim,
                 max_seq_len=8192, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList([
            FullAttnResLayer(dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_query = nn.Parameter(torch.zeros(dim))
        self.final_norm = RMSNorm(dim)
        self.out_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        skip = {n for n, _ in self.named_parameters()
                if 'res_query' in n or 'final_query' in n}
        initialized: set[int] = set()
        for name, param in self.named_parameters():
            if name in skip:
                continue
            if param.data_ptr() in initialized:
                continue
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            initialized.add(param.data_ptr())

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        sources = [x]
        for layer in self.layers:
            sources = layer(sources)
        stacked = torch.stack(sources)
        h = depth_attention(stacked, self.final_query, self.final_norm)
        logits = self.lm_head(self.out_norm(h))
        result = {"logits": logits}
        if targets is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return result

    def num_parameters(self, exclude_embeddings=True):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n

    @property
    def num_sources(self):
        return 1 + 2 * self.num_layers


class BlockAttnResTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, ff_dim,
                 max_seq_len=8192, block_size=16, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList([
            BlockAttnResLayer(dim, num_heads, ff_dim, layer_idx=i,
                              block_size=block_size, dropout=dropout)
            for i in range(num_layers)
        ])
        self.final_query = nn.Parameter(torch.zeros(dim))
        self.final_norm = RMSNorm(dim)
        self.out_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        layers_per_block = block_size // 2
        if layers_per_block < 2:
            warnings.warn(f"block_size={block_size} results in layers_per_block={layers_per_block}.")
        self._init_weights()

    def _init_weights(self):
        skip = {n for n, _ in self.named_parameters()
                if 'res_query' in n or 'final_query' in n}
        initialized: set[int] = set()
        for name, param in self.named_parameters():
            if name in skip:
                continue
            if param.data_ptr() in initialized:
                continue
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            initialized.add(param.data_ptr())

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        blocks = [x]
        partial_block = x
        for layer in self.layers:
            blocks, partial_block = layer(blocks, partial_block)
        sources = blocks + [partial_block] if partial_block is not None else blocks
        stacked = torch.stack(sources)
        h = depth_attention(stacked, self.final_query, self.final_norm)
        logits = self.lm_head(self.out_norm(h))
        result = {"logits": logits}
        if targets is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return result

    def num_parameters(self, exclude_embeddings=True):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n


class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, num_heads, ff_dim,
                 max_seq_len=8192, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList([
            StandardTransformerLayer(dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        initialized: set[int] = set()
        for name, param in self.named_parameters():
            if param.data_ptr() in initialized:
                continue
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.02)
            initialized.add(param.data_ptr())

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.out_norm(x))
        result = {"logits": logits}
        if targets is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return result

    def num_parameters(self, exclude_embeddings=True):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n
