import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SarikaConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    d_model: int = 768
    ff_hidden: int = 3072
    n_blocks: int = 12
    n_head: int = 12


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.d_model = config.d_model

    def forward(self, x):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (d_model)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: SarikaConfig) -> None:
        super().__init__()
        self.config = SarikaConfig
        self.c_fc = nn.Linear(in_features=config.d_model, out_features=config.ff_hidden)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(
            in_features=config.ff_hidden, out_features=config.d_model
        )

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: SarikaConfig) -> None:
        super().__init__()
        self.config = SarikaConfig
        self.ln_1 = nn.LayerNorm(normalized_shape=config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(normalized_shape=config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Sarika(nn.Module):
    def __init__(self, config: SarikaConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                # Embeddings
                wte=nn.Embedding(
                    num_embeddings=config.vocab_size, embedding_dim=config.d_model
                ),
                # Positional Embeddings
                wpe=nn.Embedding(
                    num_embeddings=config.block_size, embedding_dim=config.d_model
                ),
                # Repeating blocks
                h=nn.ModuleList([Block(config) for _ in range(config.n_blocks)]),
                ln_f=nn.LayerNorm(normalized_shape=config.d_model),
            )
        )
        self.lm_head = nn.Linear(
            in_features=config.d_model, out_features=config.vocab_size, bias=False
        )

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(0, T, dtype=torch.long, device=x.device)
        embeddings = self.transformer.wte(x)
        positional_embeddings = self.transformer.wpe(positions)
        x = embeddings + positional_embeddings
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        x = self.lm_head(x)
        return x

    def generate(self, tokens, max_new_tokens=128, top_k=30):
        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = self.forward(tokens)
                last_logit = x[:, -1, :]
                probs = F.softmax(last_logit, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=top_k)
                ix = torch.multinomial(topk_probs, 1)
                next_token = torch.gather(topk_indices, -1, ix)
                tokens = torch.cat((tokens, next_token), dim=1)
        return tokens

    @classmethod
    def from_pretrained(cls, model_type):
        # TODO
        pass
