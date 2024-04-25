import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from ._nn.attention import CausalSelfAttention
from ._nn.mlp import Mlp
from ._nn.layer_norm import LayerNorm
from ._mixin.optimizer import OptimizerMixin
from pathlib import Path
import sys
import time


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.d, bias=config.bias)
        self.ln_2 = LayerNorm(config.d, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.mlp = Mlp(config)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(OptimizerMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.n_vocab, config.d)
        self.wpe = nn.Embedding(config.sequence_length, config.d)
        self.lm_head = nn.Linear(config.d, config.n_vocab, bias=False)
        self.wte.weight = self.lm_head.weight
        self.drop = nn.Dropout(config.dropout)
        self.ln = LayerNorm(config.d, bias=config.bias)
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )
        self.loss_function = F.cross_entropy
        self.load_weight(config)

    def load_weight(self, config):
        experiment_path = Path(sys.argv[2])
        checkpoint_path = experiment_path / "checkpoint.ckpt"
        if checkpoint_path.exists():
            print("checkpoint exists, load from checkpoint, ignore init_from")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(checkpoint["model"])
            return

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
        # report number of parameters
        n_params = self.get_num_params()
        print("number of parameters: %.2fM" % (n_params / 1e6,))
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.config.dtype == "float16")
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_optimizer(self):
        optimizer = self.configure_optimizers(
            self.config.weight_decay,
            self.config.learning_rate,
            (self.config.beta1, self.config.beta2),
            self.config.device,
        )
        return optimizer

    def training_step(self, data):
        x, y = data
        logits = self(x)
        loss = self.loss_function(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
        )
        return logits, loss

    @torch.no_grad()
    def evaluation_step(self, data):
        x, y = data
        logits = self(x)
        loss = self.loss_function(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
        )
        # ids = torch.argmax(logits, dim=-1)
        return logits, loss

    def forward(self, idx):
        a = time.time()
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=1000, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.sequence_length
                else idx[:, -self.config.sequence_length :]
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
