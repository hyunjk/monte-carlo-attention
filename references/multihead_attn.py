import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, attention_dropout_prob, hidden_dropout_prob, layer_norm_eps):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.attention_softmax = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(attention_dropout_prob)
        )

        self.dense = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(hidden_dropout_prob)
        )

        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(self, in_features, tgt_features=None, tgt_mask=None):

        # Self attention
        if tgt_features is None:
            tgt_features = in_features

        # (1, n, i) * (i, m * p) -> (1, n, m * p)
        # (N, L, E) -> (N, L, num_heads * head_dim)
        query = self.query_proj(in_features)
        key = self.key_proj(tgt_features)
        value = self.value_proj(tgt_features)

        # (N, L, num_heads * head_dim) -> (N, num_heads, L, head_dim)
        query = self._separate_heads(query)
        key = self._separate_heads(key)
        value = self._separate_heads(value)

        # Calculate the attention scores
        # (N, num_heads, L, head_dim) * (N, num_head, head_dim, L) -> (N, num_head, L, L)
        attention = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # Apply softmax to the attention scores
        if tgt_mask is not None:
            attention += self._extend_mask(tgt_mask)
        attention = self.attention_softmax(attention)

        # Applying attention weights
        # (N, num_heads, L, L) * (N, num_heads, L, head_dim) -> (N, num_heads, L, head_dim)
        attention_value = torch.matmul(attention, value)

        # (N, num_heads, L, head_dim) -> (N, L, num_heads * head_dim)
        attention_value = self._merge_heads(attention_value)

        out_features = self.dense(attention_value)
        out_features = self.norm(out_features + in_features)

        return out_features

    def _separate_heads(self, features):
        # (N, L, num_heads * head_dim) -> (N, L, num_heads, head_dim)
        batch_size, input_len = features.shape[:2]
        features = features.view(batch_size, input_len, self.num_heads, self.head_dim)

        # (N, L, num_heads, head_dim) -> (N, num_heads, L, head_dim)
        return torch.transpose(features, 2, 1).contiguous()

    def _merge_heads(self, features):
        # (N, num_heads, L, head_dim) -> (N, L, num_heads, head_dim)
        features = torch.transpose(features, 2, 1).contiguous()

        # (N, L, num_heads, head_dim) -> (N, L, num_heads * head_dim)
        batch_size, input_len = features.shape[:2]
        return features.view(batch_size, input_len, self.num_heads * self.head_dim)

    def _extend_mask(self, mask):
        # (N, L) -> (N, 1, 1, L)
        batch_size, input_len = mask.shape[:2]
        extended_mask = mask.view(batch_size, 1, 1, input_len)

        # Adding -1e5 makes masked locations zeroed out during softmax
        return (1.0 - extended_mask) * -1e5