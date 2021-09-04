import math
import torch
import torch.nn as nn
import utils
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

import monte_carlo_attention as mca

class StatTracker:
    def __init__(self):
        self.flops = []


def inject_bert(model, alpha=1.0, use_mca=False):
    tracker = StatTracker()
    layers = model.encoder.layer
    model.tracker = tracker
    for i in range(len(layers)):
        old = layers[i].attention.self
        layers[i].attention.self = RandomizedBertSelfAttention(old, alpha, use_mca, tracker)


def inject_distilbert(model, alpha=1.0, use_mca=False):
    tracker = StatTracker()
    layers = model.transformer.layer
    model.tracker = tracker
    for i in range(len(layers)):
        old = layers[i].attention
        layers[i].attention = RandomizedDistilBertSelfAttention(old, alpha, use_mca, tracker)


def inject_longformer(model):
    layers = model.encoder.layer
    for i in range(len(layers)):
        old = layers[i].attention.self
        layers[i].attention.self = RandomizedBertSelfAttention(old)


class RandomizedBertSelfAttention(nn.Module):
    def __init__(self, attn, alpha, use_mca, tracker):

        super().__init__()

        self.attn = attn
        self.alpha = alpha
        self.use_mca = use_mca
        self.pure_len = None
        self.p_cdf = None
        self.tracker = tracker

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):

        # (N, L, E) -> (N, L, num_heads * head_dim)
        # (N, L, num_heads * head_dim) -> (N, num_heads, L, head_dim)
        key_layer = self.attn.transpose_for_scores(self.attn.key(hidden_states))
        query_layer = self.attn.transpose_for_scores(self.attn.query(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attn.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # remove batch
        if self.p_cdf is None:
            self.p_cdf = utils.get_p_cdf_from_w(self.attn.value.weight.t().contiguous(), self.attn.num_attention_heads)

        attention_probs = torch.squeeze(attention_probs, dim=0)
        hidden_states = torch.squeeze(hidden_states, dim=0)
        context_layer, flops = utils.multi_rand_attn_cuda(attention_probs, hidden_states, self.attn.value.weight,
                                                   self.attn.value.bias, self.p_cdf, self.alpha, self.use_mca)
        self.tracker.flops.append(flops)
        # add virtual batch of size one
        context_layer = torch.unsqueeze(context_layer, dim=0)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class RandomizedDistilBertSelfAttention(nn.Module):
    def __init__(self, attn, alpha, use_mca, tracker):

        super().__init__()

        self.attn = attn
        self.alpha = alpha
        self.use_mca = use_mca
        self.tracker = tracker
        self.pure_len = None
        self.p_cdf = None

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)
        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.attn.dim // self.attn.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """separate heads"""
            return x.view(bs, -1, self.attn.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.attn.n_heads * dim_per_head)

        q = shape(self.attn.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.attn.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.attn.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        # remove batch
        if self.p_cdf is None:
            self.p_cdf = utils.get_p_cdf_from_w(self.attn.v_lin.weight.t().contiguous(),
                                                self.attn.n_heads)

        attention_probs = torch.squeeze(weights, dim=0)
        hidden_states = torch.squeeze(value, dim=0)
        context_layer, flops = utils.multi_rand_attn_cuda(attention_probs, hidden_states, self.attn.v_lin.weight,
                                                   self.attn.v_lin.bias, self.p_cdf, self.alpha, self.use_mca)

        self.tracker.flops.append(flops)

        # add virtual batch of size one
        context = torch.unsqueeze(context_layer, dim=0)

        # v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        #
        #
        # context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        # context = unshape(context)  # (bs, q_length, dim)
        context = self.attn.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class RandomizedLongformerSelfAttention(nn.Module):
    def __init__(self, attn):

        super().__init__()

        self.attn = attn

        self.pure_len = None
        self.p_cdf = None

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_head_mask=None,
            is_index_masked=None,
            is_index_global_attn=None,
            is_global_attn=None,
            output_attentions=False,
    ):
        """
        :class:`LongformerSelfAttention` expects `len(hidden_states)` to be multiple of `attention_window`. Padding to
        `attention_window` happens in :meth:`LongformerModel.forward` to avoid redoing the padding on each layer.
        The `attention_mask` is changed in :meth:`LongformerModel.forward` from 0, 1, 2 to:
            * -10000: no attention
            * 0: local attention
            * +10000: global attention
        """
        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.attn.query(hidden_states)
        key_vectors = self.attn.key(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.size()

        # normalize query
        query_vectors /= math.sqrt(self.attn.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.attn.num_heads, self.attn.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.attn.num_heads, self.attn.head_dim).transpose(0, 1)

        attn_scores = self.attn._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.attn.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, -10000.0
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self.attn._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask, self.attn.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.attn.num_heads,
            self.attn.one_sided_attn_window_size * 2 + 1,
        ], f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.attn.num_heads}, {self.attn.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self.attn._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self.attn._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores

        attn_probs = nn.functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores

        # apply dropout
        attn_probs = nn.functional.dropout(attn_probs, p=self.attn.dropout, training=self.attn.training)

        #####################################################################################################################
        # attn_probs (1, seq_len, num_heads, window * 2 + 1)
        # (1, seq_len, num_heads, window * 2 + 1) -> (seq_len, num_heads, window * 2 + 1) - >

        window_size = self.attn.one_sided_attn_window_size
        num_heads = self.attn.num_heads

        # (1, seq_len, num_heads, ws * 2 + 1) -> (seq_len, num_heads, ws * 2 + 1)
        ap = torch.squeeze(attn_probs, dim=0)

        # (seq_len, num_heads, ws * 2 + 1) -> (num_heads, seq_len, ws * 2 + 1)
        ap = ap.transpose(0, 1).contiguous()

        if is_global_attn:
            ap = ap.narrow(-1, max_num_global_attn_indices, window_size * 2 + 1)

        zero_pad = torch.zeros((num_heads, window_size, window_size * 2 + 1))

        # (num_heads, seq_len, ws * 2 + 1) -> (num_heads, seq_len + window_size * 2, window_size * 2 + 1)
        ap = torch.cat((zero_pad, ap, zero_pad))

        # (num_heads, seq_len + window_size * 2, ws * 2 + 1) -> (num_heads, seq_len, window_size * 2 + 1)
        ap = ap.as_strided((num_heads, seq_len, window_size * 2 + 1),
                           ((seq_len + window_size * 2) * (window_size * 2 + 1), window_size * 2 + 1, window_size * 2),
                           window_size * 2)

        # (num_heads, seq_len)
        num_trials = torch.max(ap, dim=-1)

        # remove batch
        if self.p_cdf is None:
            self.p_cdf = utils.get_p_cdf_from_w(self.attn.value.weight.t().contiguous(), self.attn.num_heads)

        # (seq_len, batch_size, embed_dim) -> (seq_len, embed_dim)
        input_feature = torch.squeeze(hidden_states, dim=1)

        value_vectors = mca.simple_mca(input_feature, self.attn.value.weight, self.attn.value.bias, num_trials,
                                       self.p_cdf)

        # add virtual batch of size one
        value_vectors = torch.unsqueeze(value_vectors, dim=1)
        #####################################################################################################################

        value_vectors = value_vectors.view(seq_len, batch_size, self.attn.num_heads, self.attn.head_dim).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self.attn._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self.attn._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.attn.one_sided_attn_window_size
            )

        assert attn_output.size() == (batch_size, seq_len, self.attn.num_heads, self.attn.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output, global_attn_probs = self.attn._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                                         is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
                                         ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0

        outputs = (attn_output.transpose(0, 1),)

        if output_attentions:
            outputs += (attn_probs,)

        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs
