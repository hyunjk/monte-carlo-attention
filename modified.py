import math
import torch
import torch.nn as nn
import utils
from transformers.models.bert.modeling_bert import BertSelfAttention


def inject(bert_model):
    layers = bert_model.encoder.layer
    for i in range(len(layers)):
        old = layers[i].attention.self
        layers[i].attention.self = RandomizedBertSelfAttention(old)


class RandomizedBertSelfAttention(nn.Module):
    def __init__(self, attn):

        super().__init__()

        self.attn = attn

        self.pure_len = None
        self.p_cdf = None

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
        context_layer = utils.multi_rand_attn_cuda(attention_probs, hidden_states, self.attn.value.weight,
                                                   self.attn.value.bias, self.p_cdf)

        # add virtual batch of size one
        context_layer = torch.unsqueeze(context_layer, dim=0)


        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
