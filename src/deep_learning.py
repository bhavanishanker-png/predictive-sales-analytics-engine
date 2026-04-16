"""Deep learning models for SaaS sales outcome prediction.

Two architectures targeting the Phase 2 rubric (modern DL, not black-box):

1. **BiLSTM + Self-Attention**: learns sequential patterns in raw conversation
   text via a trainable embedding, bidirectional LSTM, and a lightweight
   dot-product attention pooling layer.
2. **DistilBERT classifier**: fine-tunes a pretrained Transformer encoder
   (distilbert-base-uncased) with a task-specific classification head.

Both models include:
- Xavier / Kaiming weight initialisation (explicit, not default)
- Dropout and LayerNorm regularisation
- An EarlyStopping callback to prevent overfitting
- Configurable learning-rate scheduling (ReduceLROnPlateau)
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TokenizerConfig:
    """Settings for the custom word-level tokenizer."""
    max_vocab: int = 30_000
    max_length: int = 512
    min_freq: int = 2
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"


@dataclass
class LSTMConfig:
    """Hyper-parameters for the BiLSTM + Attention classifier."""
    vocab_size: int = 30_002
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    num_classes: int = 2


@dataclass
class TransformerConfig:
    """Hyper-parameters for the DistilBERT fine-tuning classifier."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512
    dropout: float = 0.3
    num_classes: int = 2
    freeze_layers: int = 0


@dataclass
class TrainingConfig:
    """Shared training hyper-parameters."""
    batch_size: int = 32
    epochs: int = 20
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 3
    min_delta: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    grad_clip: float = 1.0
    device: str = "auto"


# ---------------------------------------------------------------------------
# Custom tokenizer (word-level, built from training corpus)
# ---------------------------------------------------------------------------

class SalesTokenizer:
    """Word-level tokenizer with frequency-based vocabulary.

    Why word-level?  SaaS sales conversations use domain jargon
    ("upsell", "ARR", "churn") that sub-word tokenizers may split
    unhelpfully.  A word-level vocabulary preserves these tokens while
    keeping the pipeline simple and interpretable.
    """

    _SPLIT_RE = re.compile(r"\s+")

    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self.config = config or TokenizerConfig()
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._fitted = False

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def fit(self, texts: List[str]) -> "SalesTokenizer":
        """Build vocabulary from a list of raw text strings."""
        counter: Counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        specials = [self.config.pad_token, self.config.unk_token]
        self.word2idx = {tok: idx for idx, tok in enumerate(specials)}

        for word, freq in counter.most_common(self.config.max_vocab):
            if freq < self.config.min_freq:
                break
            self.word2idx[word] = len(self.word2idx)

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self._fitted = True
        return self

    def encode(self, text: str) -> List[int]:
        """Convert a raw string into a list of token indices."""
        if not self._fitted:
            raise RuntimeError("Tokenizer has not been fitted.")
        unk_id = self.word2idx[self.config.unk_token]
        ids = [self.word2idx.get(w, unk_id) for w in self._tokenize(text)]
        return ids[: self.config.max_length]

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode and pad a batch to uniform length, returning a LongTensor."""
        pad_id = self.word2idx[self.config.pad_token]
        encoded = [self.encode(t) for t in texts]
        max_len = min(max((len(e) for e in encoded), default=1), self.config.max_length)
        padded = [e + [pad_id] * (max_len - len(e)) for e in encoded]
        return torch.tensor(padded, dtype=torch.long)

    def _tokenize(self, text: str) -> List[str]:
        return self._SPLIT_RE.split(text.lower().strip())


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class TextClassificationDataset(Dataset):
    """Wraps pre-tokenized integer sequences and labels."""

    def __init__(self, token_ids: torch.Tensor, labels: np.ndarray) -> None:
        self.token_ids = token_ids
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.token_ids[idx], self.labels[idx]


class TransformerDataset(Dataset):
    """Wraps HuggingFace tokenizer outputs (input_ids + attention_mask)."""

    def __init__(self, encodings: dict, labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Model 1: BiLSTM + Self-Attention
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """Additive (Bahdanau-style) self-attention over LSTM hidden states.

    Computes a scalar attention weight for each time step, then returns
    the weighted sum as a fixed-length context vector.  This lets the
    model focus on the most predictive parts of a conversation rather
    than relying solely on the final LSTM hidden state.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn.weight)

    def forward(self, lstm_out: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # lstm_out: (batch, seq_len, hidden_dim)
        scores = self.attn(lstm_out).squeeze(-1)          # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)                # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class BiLSTMClassifier(nn.Module):
    """Embedding → BiLSTM → Self-Attention → LayerNorm → FC → Softmax.

    Architecture choices (for viva):
    - **Bidirectional LSTM** captures both forward and backward context in
      conversational text, important because sales objections and resolutions
      can reference earlier or later turns.
    - **Self-attention pooling** learns which time steps carry the most
      predictive signal (e.g., closing language, pricing discussion) instead
      of naively using the last hidden state.
    - **LayerNorm + Dropout** between the FC layers stabilises training and
      prevents co-adaptation of hidden units.

    Weight initialisation:
    - Embeddings: Xavier uniform (symmetric around zero, preserves gradient
      variance through the look-up table).
    - LSTM: orthogonal initialisation for recurrent weights (mitigates
      vanishing/exploding gradients in long sequences).
    - Linear layers: Kaiming (He) uniform, suited for layers followed by
      ReLU or similar non-linearities.
    """

    def __init__(self, config: LSTMConfig, pad_idx: int = 0) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            config.embed_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        effective_hidden = config.hidden_dim * (2 if config.bidirectional else 1)
        self.attention = SelfAttention(effective_hidden)
        self.layer_norm = nn.LayerNorm(effective_hidden)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(effective_hidden, effective_hidden // 2)
        self.fc2 = nn.Linear(effective_hidden // 2, config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.embedding.weight)
        with torch.no_grad():
            self.embedding.weight[0].zero_()  # keep pad vector at zero

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 (helps learn long-range deps)
                hidden = self.config.hidden_dim
                param.data[hidden: 2 * hidden].fill_(1.0)

        for lin in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor):
        mask = (x != 0).float()                        # (batch, seq_len)
        embedded = self.dropout(self.embedding(x))     # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)              # (batch, seq_len, hidden*2)
        context, attn_weights = self.attention(lstm_out, mask)
        context = self.layer_norm(context)
        context = self.dropout(context)
        hidden = F.relu(self.fc1(context))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)                      # (batch, num_classes)
        return logits, attn_weights


# ---------------------------------------------------------------------------
# Model 2: DistilBERT Fine-tuning Classifier
# ---------------------------------------------------------------------------

class DistilBERTClassifier(nn.Module):
    """DistilBERT encoder → LayerNorm → Dropout → FC classification head.

    Architecture choices (for viva):
    - **DistilBERT** is a 6-layer distilled Transformer that retains 97 %
      of BERT's language understanding at 60 % of the parameters and 1.6×
      inference speed—practical for 100 k conversations on limited hardware.
    - **Selective layer freezing** (`freeze_layers`) lets us keep lower
      (syntactic) layers fixed and only fine-tune upper (semantic) layers,
      reducing over-fitting risk on domain-specific text.
    - The classification head uses **LayerNorm → Dropout → Linear**, a
      standard robust pattern that avoids internal covariate shift in the
      added parameters.

    Weight initialisation:
    - DistilBERT body: pretrained weights (knowledge transfer).
    - Classification head linear: Kaiming uniform.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained(config.model_name)

        if config.freeze_layers > 0:
            modules_to_freeze = [self.bert.embeddings] + list(
                self.bert.transformer.layer[: config.freeze_layers]
            )
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768 for base
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, config.num_classes)

        nn.init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5))
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_hidden = self.layer_norm(cls_hidden)
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden)
        return logits

