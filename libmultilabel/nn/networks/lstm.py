import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Embedding, LSTMEncoder


class BiLSTM_classifier(nn.Module):
    """BiLSTM_classifier

    Args:
        embed_vecs (torch.Tensor): The pre-trained word vectors of shape (vocab_size, embed_dim).
        num_classes (int): Total number of classes.
        rnn_dim (int): The size of bidirectional hidden layers. The hidden size of the LSTM network is set to rnn_dim//2. Defaults to 512.
        rnn_layers (int): The number of recurrent layers. Defaults to 1.
        encoder_dropout (float): The dropout rate of the encoder output. Defaults to 0.
        embed_dropout (float): The dropout rate of the word embedding. Defaults to 0.2.
        activation (str): Activation function to be used. Defaults to 'relu'.
    """

    def __init__(
        self,
        embed_vecs,
        num_classes,
        rnn_dim=512,
        rnn_layers=1,
        embed_dropout=0.2,
        encoder_dropout=0,
        activation="relu",
    ):
        super(BiLSTM_classifier, self).__init__()
        self.embedding = Embedding(embed_vecs, dropout=embed_dropout)
        assert rnn_dim % 2 == 0, """`rnn_dim` should be even."""
        self.encoder = LSTMEncoder(embed_vecs.shape[1], rnn_dim // 2, rnn_layers, encoder_dropout)
        self.linear = nn.Linear(rnn_dim, num_classes)
        self.activation = getattr(torch, activation, getattr(F, activation)) if activation else None    

    def forward(self, input):
        x = self.embedding(input["text"])  # (batch_size, length, embed_dim)
        x = self.encoder(x, input["length"])[:,-1,:]
        x = self.activation(x) if self.activation else x
        x = self.linear(x) 
        return {"logits": x}
