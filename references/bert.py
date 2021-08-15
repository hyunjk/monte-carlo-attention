import torch
import torch.nn as nn
import math
import transformer_proxy


class CombinerConfig:

    def __init__(self):
        self.embed_dim = 768

        # Embedding
        self.num_vocabs = 30522
        self.max_num_tokens = 512
        self.max_num_segments = 64

        # Encoding
        self.hidden_dim = 3072
        self.num_encoding_layers = 12
        self.num_attention_heads = 12
        self.layer_norm_eps = 1e-12

        self.attention_dropout_prob = 0.3
        self.hidden_dropout_prob = 0.4

    @staticmethod
    def from_transformers(transformers):
        assert len(transformers) > 0
        assert len(set([t.config.num_hidden_layers for t in transformers])) == 1

        config = CombinerConfig()
        primary = transformers[0]

        config.num_vocabs = primary.config.vocab_size
        config.embed_dim = primary.config.hidden_size
        config.layer_norm_eps = primary.config.layer_norm_eps
        config.max_num_tokens = primary.config.max_position_embeddings
        config.num_encoding_layers = primary.config.num_hidden_layers

        return config


class CombinerForQuestionAnswering(nn.Module):

    def __init__(self, config, transformers):
        super().__init__()
        self.combiner = Combiner(config, transformers)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def forward(self, token_ids_list, mask, segments):
        features = self.combiner(token_ids_list, mask, segments)

        logits = self.qa_outputs(features)
        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class Combiner(nn.Module):

    def __init__(self, config, transformers):
        super().__init__()

        self.transformers = transformers

        self.embedding = CombinerEmbedding(config, transformers)
        self.encoder = CombinerEncoder(config, transformers)

    def forward(self, token_ids_list, mask, segments):
        features, tgt_features, segment_bias = self.embedding(token_ids_list, segments)
        features = self.encoder(features, mask, segments, segment_bias)

        return features


class CombinerEmbedding(nn.Module):
    def __init__(self, config, transformers):
        super().__init__()

        self.transformers = transformers

        self.word_embedding = nn.Embedding(config.num_vocabs, config.embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_num_tokens, config.embed_dim)
        self.segment_embedding = nn.Embedding(config.max_num_segments, config.embed_dim, padding_idx=0)

        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids_list, segments):

        # Calculate segment bias
        segment_ids_list = []
        for i, seg_size in enumerate(segments):
            segment_ids_list.append(torch.ones(seg_size, dtype=torch.long) * (i + 1))

        token_ids = token_ids_list[0]
        segment_ids = torch.cat(segment_ids_list).unsqueeze(0).to(token_ids_list[0].device)
        position_ids = torch.arange(segments[0]).unsqueeze(0).to(token_ids_list[0].device)

        word_embeddings = self.word_embedding(token_ids)
        position_embeddings = self.position_embedding(position_ids)
        segment_bias = self.segment_embedding(segment_ids)

        embeddings = self.norm(word_embeddings + position_embeddings)
        embeddings = self.dropout(embeddings)

        # Embedding
        tgt_embeddings_list = []

        for transformer, token_ids in zip(self.transformers, token_ids_list):
            embeddings = transformer.embedding(token_ids)
            tgt_embeddings_list.append(embeddings)

        return embeddings, torch.cat(tgt_embeddings_list, dim=1), segment_bias


class CombinerTokenizer:
    def __init__(self, transformers):
        self.tokenizers = [transformer.tokenizer for transformer in transformers]

    def analyze(self, text):
        token_ids_list = []
        mask_list = []
        segments = []
        for i, tokenizer in enumerate(self.tokenizers):
            tok = tokenizer(text, padding=True, return_tensors='pt')
            token_ids_list.append(tok.input_ids)
            mask_list.append(tok.attention_mask)
            segments.append(tok.attention_mask.size(1))

        mask = torch.cat(mask_list, dim=1)

        return token_ids_list, mask, segments


class CombinerEncoder(nn.Module):

    def __init__(self, config, transformers):
        super().__init__()

        self.transformers = transformers

        self.num_layers = config.num_encoding_layers

        self.shared_layer = EncodingLayer(config)
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def forward(self, in_features, tgt_features, mask, segments, segment_bias):

        for i in range(self.num_layers):

            # Global attention (CLS token)
            # (N, 1, E) @ (N, L, E) -> (N, 1, E)

            biased = self.norm(tgt_features + segment_bias)
            in_features = self.shared_layer(in_features, biased, mask)
            # out_features.append(feature)

            # Segment local attention
            # (N, L, E) @ (N, L+Lh, E) -> (N, L, E)
            tgt_out_features = []
            seg_idx = 0
            for seg_size, transformer in zip(segments, self.transformers):
                seg_in = list(range(seg_idx, seg_idx + seg_size))
                seg_idx += seg_size
                feature = transformer.layer(i, tgt_features[:, seg_in], mask[:, seg_in])
                tgt_out_features.append(feature)

            tgt_features = torch.cat(tgt_out_features, dim=1)

        return in_features


class EncodingLayer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.attention = MultiheadAttention(
            config.embed_dim,
            config.num_attention_heads,
            config.attention_dropout_prob,
            config.hidden_dropout_prob,
            config.layer_norm_eps
        )

        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.Dropout(config.hidden_dropout_prob)
        )

        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)

    def forward(self, in_features, tgt_features=None, tgt_mask=None):
        interim_features = self.attention(in_features, tgt_features, tgt_mask)

        out_features = self.ffn(interim_features)
        out_features = self.norm(out_features + interim_features)

        return out_features


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