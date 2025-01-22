import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Setting embedding_dim same as model dimension
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=d_model)

    # input is (batch_size, sequence_length) => (batch_size, sequence_length, embedding_dim)
    def forward(self, token_matrix: torch.Tensor):
        return self.embedding(token_matrix) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a positional encoding matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # PE (pos,2i) =sin(pos/10000 2i/d model)
        # Create a vector of shape (seq_len, 1) position of each token in the sequence (from 0 to seq_len - 1).
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1)  # shape [seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))  # denominator in log space

        # Apply the sine. to even positions of encoding space
        # apply the cos to odd positions of encoding space
        pe[:, 0::2] = torch.sin(position * div_term)  # step 0,2,4
        pe[:, 1::2] = torch.cos(position * div_term)  # step, 1,3,5

        # add batch dimension (seq_len, d_model) => (1, Seq_len, d_model)
        pe = pe.unsqueeze(0)  #  (1, Seq_len, d_model)

        self.register_buffer('pe', pe)  # register as part of state to be saved

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(
            False
        )  # apply positional encoding and set requires_grad to False because the positional encodings are fixed and don't need to be learned."
        return self.dropout(
            x
        )  # Dropout to avoid overfitting by randomly setting a fraction(dropout ratio set in init) of positional encodings to 0 at each update


# add and norm
class LayerNorm(nn.Module):

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multiplied
        self.bias = nn.Parameter(torch.ones(1))  #added

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# 2 layer NN outer with 512 and inner with 2048
class FeedForward(nn.Module):
    # d_model is outer layer dim, d_ff is hidden layer
    def __init__(self,
                 dropout: float,
                 d_model: int = 512,
                 d_ff: int = 2048) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x: torch.Tensor):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
        layer1_activation = self.linear_1(x)
        layer_2_x = torch.relu(layer1_activation)
        layer_2_x = self.dropout(layer_2_x)
        return self.linear_2(layer_2_x)


# Input replicated to Q, K , V
# each has learned parameter W
# QW, KW, VW is split into number of heads along the embedding dimension
# Concat Heads and multiple with Wq = Multi head attention Wo
class MultiHeadAttention(nn.Module):

    def __init__(self, dropout, num_heads, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads, decay cannot be obtained precisely"

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)  # Wq
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)  # Wk
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)  # Wv

        self.w_o = nn.Linear(in_features=d_model, out_features=d_model)  #Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # (Batch, h, seq_len, seq_len) 
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1)  # Softmax all attention scores for Q
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    # Mask set during attention computation in decoder
    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor):
        query = self.w_q(
            q
        )  # (batch, seq_len, d_model) x (d_model, d_model) => (Batch, seq_len, d_model)
        key = self.w_k(
            k
        )  # (batch, seq_len, d_model) x (d_model, d_model) => (Batch, seq_len, d_model)
        value = self.w_v(
            v
        )  # (batch, seq_len, d_model) x (d_model, d_model) => (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) => (Batch, seq_len, h, d_k) => (Batch, h, seq_len, d_k) 
        # Transpose to facilitate parallelism across attention heads.
        query = query.view(query.shape[0], query.shape[1], self.num_heads,
                           self.d_k).transpose(1, 2)

        # (Batch, seq_len, d_model) => (Batch, seq_len, h, d_k) => (Batch, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.num_heads,
                       self.d_k).transpose(1, 2)

        # (Batch, seq_len, d_model) => (Batch, seq_len, h, d_k) => (Batch, h, seq_len, d_k)
        value = key.view(key.shape[0], key.shape[1], self.num_heads,
                         self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout)
        # (Batch, h, seq_len, d_k) => (Batch, Seq_len, h, d_k) =>  (Batch, Seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape(0), -1,
                                                self.h * self.d_k)

        # (Batch, Seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    # pre-normalization approach for greater stability, original paper uses post-normalization
    def forward(self, x:torch.Tensor, sublayer:nn.Module):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout=dropout) for _ in range(2)])

    def forward(self, x:torch.Tensor, src_mask:torch.Tensor):
        x = self.residual_connection[0](
            #                                       q,k,v
            x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention,
                 cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connection = nn.ModuleList(
            [ResidualConnection(dropout=dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](
            x, lambda x: self.cross_attention_block(x, encoder_output,
                                                    encoder_output, src_mask))
        x = self.residual_connection[2](
            x, lambda x: self.self_attention_block(x, self.feed_forward_block))
        return x


class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


class OutputProjection(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(in_features=d_model, out_features=vocab_size)

    # input is (batch_size, sequence_length, d_model) -> (batch_Size, seq_len, vocab_size) (applied logsoftmax)
    def forward(self, x: torch.Tensor):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbeddings, target_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, target_pos: PositionalEncoding,
                 proj: OutputProjection):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.proj = proj

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.target_embed(tgt)
        tgt = self.target_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.proj(x)


def train_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      N: int = 6,
                      h: int = 8,
                      dropout: float = 0.1,
                      d_ff: int = 2084) -> Transformer:
    src_embed = InputEmbeddings(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embed = InputEmbeddings(d_model=d_model, vocab_size=tgt_vocab_size)

    src_pos = PositionalEncoding(d_model=d_model,
                                 seq_len=src_seq_len,
                                 dropout=dropout)
    tgt_pos = PositionalEncoding(d_model=d_model,
                                 seq_len=tgt_seq_len,
                                 dropout=dropout)

    encoder_blocks = nn.ModuleList()
    decoder_blocks = nn.ModuleList()
    for _ in range(N):

        self_attention_block = MultiHeadAttention(dropout=dropout,
                                                  num_heads=h,
                                                  d_model=d_model)
        feed_forward_block = FeedForward(dropout=dropout,
                                         d_model=d_model,
                                         d_ff=d_ff)
        e = EncoderBlock(self_attention_block=self_attention_block,
                         feed_forward_block=feed_forward_block,
                         dropout=dropout)
        encoder_blocks.append(e)

        multi_head_attention = MultiHeadAttention(dropout=dropout,
                                                  num_heads=h,
                                                  d_model=d_model)
        cross_attention = MultiHeadAttention(dropout=dropout,
                                             num_heads=h,
                                             d_model=d_model)
        feed_forward = FeedForward(dropout=dropout, d_model=d_model, d_ff=d_ff)
        d = DecoderBlock(self_attention_block=multi_head_attention,
                         cross_attention_block=cross_attention,
                         feed_forward_block=feed_forward,
                         dropout=dropout)
    proj = OutputProjection(d_model=d_model, vocab_size=tgt_vocab_size)
    encoder = Encoder(layers=encoder_blocks)
    decoder = Decoder(layers=decoder_blocks)
    transformer = Transformer(encoder=encoder,
                              decoder=decoder,
                              src_embed=src_embed,
                              target_embed=tgt_embed,
                              src_pos=src_pos,
                              target_pos=tgt_pos,
                              proj=proj)

    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return transformer
