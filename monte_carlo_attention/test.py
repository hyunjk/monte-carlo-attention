import torch
import monte_carlo_attention as mca

# weight

h = 4
n = 73
d_in = h * 32
d_out = h * 32
hd_out = d_out // h

# (H, N, N)
a = torch.softmax(torch.randn((h, n, n), device='cuda'), dim=2)
#
# # (N, D_IN)
x = torch.randn((n, d_in), device='cuda')
#
# # (D_IN, D_OUT)
w = torch.randn((d_in, d_out), device='cuda')
bias = torch.randn((d_out), device='cuda')
# (H, N)
v, _ = torch.max(a, 1)
res = (v * d_in).int()  # min-> 1

# (H, D_IN + 1)

p_cdf = mca.eval_sampling_prob_cdf(w.t().contiguous(), h)

result = mca.monte_carlo_multihead_attention(a, x, w, bias, res, p_cdf)

# def get_res(attn, d_in):
#     v, _ = torch.max(attn, 1)
#     return (v * d_in).int()
#
#
# def get_p_cdf(w, h):
#     d_in, d_out = w.shape
#     hd_out = d_out // h
#
#     w = w.view((d_in, h, hd_out))
#     p = torch.norm(w, p=2, dim=2)
#     p = p / torch.sum(p, dim=0)
#
#     p = torch.cat((torch.zeros((1, h), device=w.device), p), dim=0)
#
#     p_cdf = torch.transpose(torch.cumsum(p, dim=0), 0, 1).contiguous()
#
#     return p_cdf
#
#
# # (H, N, N)
# attn = torch.softmax(torch.randn((h, n, n), device='cuda'), dim=2)
#
# # (N, D_IN)
# input = torch.randn((n, d_in), device='cuda')
#
# # (D_IN, D_OUT)
# weight = torch.randn((d_in, d_out), device='cuda')
#
# # (H, N)
# res = get_res(attn, d_in)
#
# # (H, D_IN + 1)
# p_cdf = get_p_cdf(weight, h)
#
# # (h, n, n) * (n, d_out) = (h, n, d_out) = NOT
# # (h, n, n) * (h, n, hd_out) = (h, n, hd_out)
# result = raa_cuda.raa(attn, res, input, weight, p_cdf)

print(result)
print(result.shape)
