import torch
import monte_carlo_attention as mca
import torch.linalg as linalg


# RandNLA


# a: (m, n)
# b: (n, k)
# c: sample count (0 < c < n)
def rand_matmul(a, b, r, uniform_sample=False):
    m, n = a.shape
    n_, k = b.shape

    # assert compatible size!
    assert n == n_

    # generate sketch matrix s
    p = torch.empty(n, dtype=torch.float).cuda()

    if uniform_sample:
        p = p.fill_(1 / n)
    else:
        for i in range(n):
            # a_norm = torch.norm(a[:, i], p=2)
            b_norm = torch.norm(b[i], p=2)
            p[i] = b_norm * b_norm

        p /= torch.sum(p)

    # construct sketch by random sampling
    sketch = torch.multinomial(p, r, replacement=not uniform_sample)

    out = torch.zeros((m, k)).cuda()

    for i in range(r):
        k = sketch[i]
        out += torch.outer(a[:, k], b[k]) / (r * p[k])

    return out


# randomized attention
# a: (n, n)
# x: (n, p)
# w: (p, q)
# out: (n, q)
def rand_attn(a, x, w):
    # AXW

    n, n_ = a.shape
    n__, p = x.shape
    p_, q = w.shape

    assert n == n_
    assert n == n__
    assert p == p_

    # get importance matrix
    # max-reduce last dimension
    v, _ = torch.max(a, dim=0)  # torch.ones(n)
    # v = torch.ones(n).cuda()

    xw = torch.empty((n, q)).cuda()

    # print(f'started!! {a.shape}')

    for i in range(n):
        res = max(int(p * v[i]), 1)
        xw[i] = torch.squeeze(rand_matmul(x[i].unsqueeze(0), w, res))

    axw = torch.matmul(a, xw)

    return axw


# multi-head randomized attention
# a: (h, n, n)
# x: (n, p)
# w: (p, q)
# out: (n, q)
def multi_rand_attn(a, x, w):
    # convention
    # x * w = (n, p) * (p, q) = (n, q)
    #                         = (n, h * q') = (n, h, q')
    #                                      -> (h, n, q')
    # a * x * w = (h, n, n) * (h, n, q') = (h, n, q')
    #                                   -> (n, h, q')
    #                                   -> (n, h * q')
    #                                   -> (n, q)
    h, n, n_ = a.shape
    n__, p = x.shape
    p_, q = w.shape

    assert n == n_
    assert n == n__
    assert p == p_

    q_frag = q // h
    assert q_frag * h == q

    # get importance matrix
    # max-reduce last dimension (m, n)
    v, _ = torch.max(a, dim=1)  # torch.ones(n)
    # v = torch.ones((h, n)).cuda()

    # (p, q) -> (p, h, q')
    w = w.view(p, h, q_frag)
    xw = torch.empty((h, n, q_frag)).cuda()

    for i in range(h):
        for j in range(n):
            res = max(int(p * v[i][j]), 1)
            xw[i][j] = torch.squeeze(rand_matmul(x[j].unsqueeze(0), w[:, i, :], res))

    # (h, n, n) * (h, n, q') = (h, n, q')
    axw = torch.matmul(a, xw)

    # (h, n, q') -> (n, h, q')
    axw = torch.transpose(axw, dim0=0, dim1=1).contiguous()

    return axw.view(n, q)


# multi-head randomized attention
# a: (h, n, n)
# x: (n, p)
# w: (p, q)
# out: (n, q)
def multi_rand_attn_cuda(a, x, w, bias, p_cdf=None):
    h, n, n_ = a.shape
    n__, d_in = x.shape
    d_out, d_in_ = w.shape

    assert n == n_
    assert n == n__
    assert d_in == d_in_

    hd_out = d_out // h
    assert hd_out * h == d_out
    # (H, N)

    v, _ = torch.max(a, 1)  # hard fix!
    # v = torch.ones_like(v)

    # res = (v * d_in).int()  # resolution이 0이 되면 안된다.
    res = torch.clamp_max(v * d_in * 3, d_in).int()

    # (H, D_IN)
    # 한번만 계산하면 됨.
    if p_cdf is None:
        p_cdf = get_p_cdf_from_w(w.t().contiguous(), h)

    result = mca.monte_carlo_multihead_attention(a, x, w, bias, res, p_cdf)

    return result


def get_p_cdf_from_w(w, h):
    return mca.eval_sampling_prob_cdf(w, h)

    # w = w.t()
    #
    # d_in, d_out = w.shape
    # hd_out = d_out // h
    #
    # hw = w.view(d_in, h, hd_out)
    #
    # p = torch.pow(linalg.vector_norm(hw, ord=2, dim=2), 2)
    # p = p / torch.sum(p, dim=0)
    # p_cdf = torch.cumsum(p, dim=0)
    # p_cdf = torch.transpose(p_cdf, 0, 1).contiguous()
    #
    # return p_cdf


def multi_attn(a, x, w, b):
    h, n, n_ = a.shape
    n__, d_in = x.shape
    d_in_, d_out = w.shape
    d_out_ = b.shape[0]

    assert n == n_
    assert n == n__
    assert d_in == d_in_
    assert d_out == d_out_

    hd_out = d_out // h
    assert hd_out * h == d_out

    # (n, d_in) * (d_in, d_out) = (n, d_out)
    v = torch.matmul(x, w.t()) + b

    # (n, d_out) -> (n, h, hd_out) -> (h, n, hd_out)
    v = v.view(n, h, hd_out)
    v = torch.transpose(v, dim0=0, dim1=1).contiguous()

    # (h, n, n) * (h, n, hd_out) = (h, n, hd_out)
    out = torch.matmul(a, v)

    # (h, n _hd_out) - > (n, h, hd_out) -> (n, d_out)
    out = torch.transpose(out, dim0=0, dim1=1).contiguous()
    out = out.view(n, d_out)

    return out


def attn(a, x, w):
    n, n_ = a.shape
    n__, p = x.shape
    p_, q = w.shape

    assert n == n_
    assert n == n__
    assert p == p_

    return torch.matmul(a, torch.matmul(x, w))
