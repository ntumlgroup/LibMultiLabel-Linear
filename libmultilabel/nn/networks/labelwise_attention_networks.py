from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel

from .modules import (
    AttentionRNNLinearOutput,
    CNNEncoder,
    Embedding,
    GRUEncoder,
    LabelwiseAttention,
    LabelwiseLinearOutput,
    LabelwiseMultiHeadAttention,
    LSTMEncoder,
    diff_QV_LabelwiseAttention,
)


class LabelwiseAttentionNetwork(ABC, nn.Module):
    """Base class for Labelwise Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        embed_dropout (float): The dropout rate of the word embedding.
        encoder_dropout (float): The dropout rate of the encoder output.
        hidden_dim (int): The output dimension of the encoder.
    """

    def __init__(
        self,
        embed_vecs: Tensor,
        num_classes: int,
        embed_dropout: float,
        encoder_dropout: float,
        hidden_dim: int,
    ):
        super(LabelwiseAttentionNetwork, self).__init__()
        self.embedding = Embedding(embed_vecs, embed_dropout)
        self.encoder = self._get_encoder(embed_vecs.shape[1], encoder_dropout)
        self.attention = self._get_attention()
        self.output = LabelwiseLinearOutput(hidden_dim, num_classes)

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def _get_encoder(self, input_size, dropout):
        raise NotImplementedError

    @abstractmethod
    def _get_attention(self):
        raise NotImplementedError


class RNNLWAN(LabelwiseAttentionNetwork):
    """Base class for RNN Labelwise Attention Network"""

    def forward(self, inputs):
        # (batch_size, sequence_length, embed_dim)
        x = self.embedding(inputs["text"])
        # (batch_size, sequence_length, hidden_dim)
        x = self.encoder(x, inputs["length"])
        x, _ = self.attention(x)  # (batch_size, num_classes, hidden_dim)
        x = self.output(x)  # (batch_size, num_classes)
        return {"logits": x}


class BiGRULWAN(RNNLWAN):
    """BiGRU Labelwise Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the GRU network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): The number of recurrent layers. Defaults to 1.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
    """

    def __init__(self, embed_vecs, num_classes, rnn_dim=512, rnn_layers=1, embed_dropout=0.2, encoder_dropout=0):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        super(BiGRULWAN, self).__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return GRUEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        return LabelwiseAttention(self.rnn_dim, self.num_classes)


class AttentionRNN(RNNLWAN):
    def __init__(
        self,
        embed_vecs,
        num_classes: int,
        rnn_dim: int,
        linear_size: list[int, ...],
        freeze_embed_training: bool = False,
        rnn_layers: int = 1,
        embed_dropout: float = 0.2,
        encoder_dropout: float = 0.5,
    ):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        super().__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim)
        self.embedding = Embedding(
            embed_vecs, dropout=embed_dropout, freeze=freeze_embed_training, use_sparse_embed=True
        )
        self.output = AttentionRNNLinearOutput([self.rnn_dim] + linear_size, 1)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return LSTMEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        # return LabelwiseAttention(self.rnn_dim * 2, self.num_classes, init_fn=xavier_uniform_)
        return LabelwiseAttention(self.rnn_dim, self.num_classes)

    def forward(self, inputs):
        # N: num_batches, L: sequence_length, E: emb_size, v: vocab_size, C: num_classes, H: hidden_dim
        # input : dict["text", (N, L, V), "labels", (N, L, C: csr_matrix)]
        x, lengths, masks = self.embedding(inputs["text"])  # (N, L, E)
        x = self.encoder(x, lengths)  # (N, L, 2 * H)
        x, _ = self.attention(x, masks)  # (N, C, 2 * H)
        x = self.output(x)  # (N, C)
        return {"logits": x}


class BiLSTMLWAN(RNNLWAN):
    """BiLSTM Labelwise Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the LSTM network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): The number of recurrent layers. Defaults to 1.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
    """

    def __init__(self, embed_vecs, num_classes, rnn_dim=512, rnn_layers=1, embed_dropout=0.2, encoder_dropout=0):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        super(BiLSTMLWAN, self).__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return LSTMEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        return LabelwiseAttention(self.rnn_dim, self.num_classes)


class BiLSTMLWMHAN(LabelwiseAttentionNetwork):
    """BiLSTM Labelwise Multihead Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the LSTM network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): The number of recurrent layers. Defaults to 1.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
        num_heads (int): The number of parallel attention heads. Defaults to 8.
        attention_dropout (float): The dropout rate for the attention. Defaults to 0.0.
    """

    def __init__(
        self,
        embed_vecs,
        num_classes,
        rnn_dim=512,
        rnn_layers=1,
        embed_dropout=0.2,
        encoder_dropout=0,
        num_heads=8,
        attention_dropout=0.0,
    ):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        super(BiLSTMLWMHAN, self).__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return LSTMEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        return LabelwiseMultiHeadAttention(self.rnn_dim, self.num_classes, self.num_heads, self.attention_dropout)

    def forward(self, inputs):
        # (batch_size, sequence_length, embed_dim)
        x = self.embedding(inputs["text"])
        # (batch_size, sequence_length, hidden_dim)
        x = self.encoder(x, inputs["length"])
        # (batch_size, num_classes, hidden_dim)
        x, _ = self.attention(x, attention_mask=inputs["text"] == 0)
        x = self.output(x)  # (batch_size, num_classes)
        return {"logits": x}


class CNNLWAN(LabelwiseAttentionNetwork):
    """CNN Labelwise Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        filter_sizes (list): Size of convolutional filters.
        num_filter_per_size (int): The number of filters in convolutional layers in each size. Defaults to 50.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
        activation (str): Activation function to be used. Defaults to 'tanh'.
    """

    def __init__(
        self,
        embed_vecs,
        num_classes,
        filter_sizes=None,
        num_filter_per_size=50,
        embed_dropout=0.2,
        encoder_dropout=0,
        activation="tanh",
    ):
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filter_per_size = num_filter_per_size
        self.activation = activation
        self.hidden_dim = num_filter_per_size * len(filter_sizes)
        super(CNNLWAN, self).__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, self.hidden_dim)

    def _get_encoder(self, input_size, dropout):
        return CNNEncoder(
            input_size, self.filter_sizes, self.num_filter_per_size, self.activation, dropout, channel_last=True
        )

    def _get_attention(self):
        return LabelwiseAttention(self.hidden_dim, self.num_classes)

    def forward(self, inputs):
        # (batch_size, sequence_length, embed_dim)
        x = self.embedding(inputs["text"])
        x = self.encoder(x)  # (batch_size, sequence_length, hidden_dim)
        x, _ = self.attention(x)  # (batch_size, num_classes, hidden_dim)
        x = self.output(x)  # (batch_size, num_classes)
        return {"logits": x}


class CNNLWAN_exps(LabelwiseAttentionNetwork):
    """CNN Labelwise Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        filter_sizes (list): Size of convolutional filters.
        num_filter_per_size (int): The number of filters in convolutional layers in each size. Defaults to 50.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
        activation (str): Activation function to be used. Defaults to 'tanh'.
    """

    def __init__(
        self,
        embed_vecs,
        num_classes,
        attn_mode,
        output_mode,
        freeze_embed,
        d_a=None,
        linear_size=None,
        filter_sizes=None,
        num_filter_per_size=50,
        embed_dropout=0.2,
        encoder_dropout=0,
        activation="tanh",
    ):
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filter_per_size = num_filter_per_size
        self.activation = activation
        self.hidden_dim = num_filter_per_size * len(filter_sizes)
        self.d_a = d_a
        self.attn_mode = attn_mode
        self.output_mode = output_mode
        super(CNNLWAN_exps, self).__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, self.hidden_dim)
        self.embedding = Embedding(embed_vecs, dropout=embed_dropout, freeze=freeze_embed)
        if self.attn_mode == "tanhW":
            self.W = nn.Linear(self.hidden_dim, self.d_a, bias=False)
        if self.output_mode == "relu":
            self.output = AttentionRNNLinearOutput([self.hidden_dim] + linear_size, 1)

    def _get_encoder(self, input_size, dropout):
        return CNNEncoder(
            input_size, self.filter_sizes, self.num_filter_per_size, self.activation, dropout, channel_last=True
        )

    def _get_attention(self):
        if self.attn_mode == "tanhW":
            return diff_QV_LabelwiseAttention(self.d_a, self.num_classes)
        elif self.attn_mode == "tanh":
            return diff_QV_LabelwiseAttention(self.hidden_dim, self.num_classes)
        else:
            return LabelwiseAttention(self.hidden_dim, self.num_classes)

    def forward(self, inputs):
        # (batch_size, sequence_length, embed_dim)
        x = self.embedding(inputs["text"])
        H = self.encoder(x)  # (batch_size, sequence_length, hidden_dim)
        if self.attn_mode == "tanhW":
            Z = torch.tanh(self.W(H))
            x, _ = self.attention(Z, H)  # (batch_size, num_classes, hidden_dim)
        if self.attn_mode == "tanh":
            Z = torch.tanh(H)
            x, _ = self.attention(Z, H)  # (batch_size, num_classes, hidden_dim)
        if self.attn_mode == "vanilla":
            x, _ = self.attention(H)  # (batch_size, num_classes, hidden_dim)
        x = self.output(x)  # (batch_size, num_classes)
        return {"logits": x}


class BiLSTMLWAN_exps(RNNLWAN):
    """BiLSTM Labelwise Attention Network

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the LSTM network
            is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): The number of recurrent layers. Defaults to 1.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
    """

    def __init__(
        self,
        freeze_embed,
        embed_vecs,
        num_classes,
        attn_mode,
        output_mode,
        d_a=None,
        linear_size=None,
        rnn_dim=512,
        rnn_layers=1,
        embed_dropout=0.2,
        encoder_dropout=0,
    ):
        self.num_classes = num_classes
        self.rnn_dim = rnn_dim
        self.rnn_layers = rnn_layers
        self.d_a = d_a
        self.linear_size = linear_size
        self.attn_mode = attn_mode
        self.output_mode = output_mode
        super(BiLSTMLWAN_exps, self).__init__(embed_vecs, num_classes, embed_dropout, encoder_dropout, rnn_dim)
        self.embedding = Embedding(embed_vecs, embed_dropout, freeze=freeze_embed)
        if self.attn_mode == "tanhW":
            self.W = nn.Linear(rnn_dim, self.d_a, bias=False)
        if self.output_mode == "relu":
            self.output = AttentionRNNLinearOutput([self.rnn_dim] + self.linear_size, 1)

    def _get_encoder(self, input_size, dropout):
        assert self.rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        return LSTMEncoder(input_size, self.rnn_dim // 2, self.rnn_layers, dropout)

    def _get_attention(self):
        if self.attn_mode == "tanhW":
            return diff_QV_LabelwiseAttention(self.d_a, self.num_classes)
        elif self.attn_mode == "tanh":
            return diff_QV_LabelwiseAttention(self.rnn_dim, self.num_classes)
        elif self.attn_mode == "vanilla":
            return LabelwiseAttention(self.rnn_dim, self.num_classes)
        else:
            return None

    def forward(self, inputs):
        # (batch_size, sequence_length, embed_dim)
        x = self.embedding(inputs["text"])
        # (batch_size, sequence_length, hidden_dim)
        H = self.encoder(x, inputs["length"])
        if self.attn_mode == "tanhW":
            Z = torch.tanh(self.W(H))
            x, _ = self.attention(Z, H)  # (batch_size, num_classes, hidden_dim)
        if self.attn_mode == "tanh":
            Z = torch.tanh(H)
            x, _ = self.attention(Z, H)  # (batch_size, num_classes, hidden_dim)
        if self.attn_mode == "vanilla":
            x, _ = self.attention(H)  # (batch_size, num_classes, hidden_dim)
        x = self.output(x)  # (batch_size, num_classes)
        return {"logits": x}


class BERTLWAN_exps(nn.Module):
    """BERT Labelwise Attention Network

    Args:
        num_classes (int): Total number of classes.
        encoder_hidden_dropout (float): The dropout rate of the feed forward sublayer in each BERT layer. Defaults to 0.1.
        encoder_attention_dropout (float): The dropout rate of the attention sublayer in each BERT layer. Defaults to 0.1.
        post_encoder_dropout (float): The dropout rate of the dropout layer after the BERT model. Defaults to 0.
        lm_weight (str): Pretrained model name or path. Defaults to 'bert-base-cased'.
        lm_window (int): Length of the subsequences to be split before feeding them to
            the language model. Defaults to 512.
    """

    def __init__(
        self,
        num_classes,
        attn_mode,  # for labelwise attn
        output_mode,  # for labelwise attn
        encoder_hidden_dropout=0.1,
        encoder_attention_dropout=0.1,
        lm_weight="bert-base-cased",
        d_a=None,  # W
        linear_size=None,  # mlp
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes

        # for labelwise attn
        self.d_a = d_a
        self.attn_mode = attn_mode
        self.output_mode = output_mode

        self.encoder = AutoModel.from_pretrained(
            lm_weight,
            torchscript=True,
            hidden_dropout_prob=encoder_hidden_dropout,
            attention_probs_dropout_prob=encoder_attention_dropout,
        )
        # no need to do post encoder dropout as we have hidden_dropout_prob?
        # self.post_encoder_dropout = nn.Dropout(p=post_encoder_dropout)
        self.hidden_dim = self.encoder.config.hidden_size
        self.attention = self._get_attention()
        self.output = LabelwiseLinearOutput(self.hidden_dim, num_classes)

        if self.attn_mode == "tanhW":
            self.W = nn.Linear(self.hidden_dim, self.d_a, bias=False)
        if self.output_mode == "relu":
            self.output = AttentionRNNLinearOutput([self.hidden_dim] + linear_size, 1)

    def _get_attention(self):
        if self.attn_mode == "tanhW":
            return diff_QV_LabelwiseAttention(self.d_a, self.num_classes)
        elif self.attn_mode == "tanh":
            return diff_QV_LabelwiseAttention(self.hidden_dim, self.num_classes)
        else:
            return LabelwiseAttention(self.hidden_dim, self.num_classes)

    def forward(self, input):
        input_ids = input["text"]  # (batch_size, sequence_length)
        # (batch_size, sequence_length, hidden_dim)
        H = self.encoder(input_ids, attention_mask=input_ids != self.encoder.config.pad_token_id)[0]

        if self.attn_mode == "tanhW":
            Z = torch.tanh(self.W(H))
            x, _ = self.attention(Z, H)  # (batch_size, num_classes, hidden_dim)
        if self.attn_mode == "tanh":
            Z = torch.tanh(H)
            x, _ = self.attention(Z, H)  # (batch_size, num_classes, hidden_dim)
        if self.attn_mode == "vanilla":
            x, _ = self.attention(H)  # (batch_size, num_classes, hidden_dim)
        x = self.output(x)  # (batch_size, num_classes)

        return {"logits": x}
