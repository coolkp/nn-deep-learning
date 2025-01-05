import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Setting embedding_dim same as model dimension
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    
    # input is (batch_size, sequence_length)
    def forward(self, token_matrix: torch.Tensor):
        return self.embedding(input) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int , dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
    