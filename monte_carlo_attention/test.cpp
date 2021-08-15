#include <iostream>
#include <torch/torch.h>

#include "test.h"
#include "attention.cuh"

int main() {

    int h = 12;
    int n = 5;
    int d_in = 144;
    int d_out = 144;

    // (H, N, N)
    torch::Tensor attn = torch::softmax(torch::randn({h, n, n}, torch::device(torch::kCUDA)), 2);

    // (N, D_IN)
    torch::Tensor input = torch::randn({n, d_in}, torch::device(torch::kCUDA));

    // (D_IN, D_OUT)
    torch::Tensor weight = torch::randn({d_in, d_out}, torch::device(torch::kCUDA));

    torch::Tensor bias = torch::randn({d_out}, torch::device(torch::kCUDA));

    // (H, N)
    torch::Tensor num_trials = get_res(attn, d_in);

    // (H, D_IN + 1)
    torch::Tensor sampling_prob_cdf = mca::eval_sampling_prob_cdf(weight.t().contiguous(), h);

    auto result = mca::monte_carlo_multihead_attention(attn, input, weight, bias, num_trials, sampling_prob_cdf);


}


/**
 * attention matrix로부터 resolution 구하기
 * @param attn (H, N, N)
 * @param d_in (H, N)
 * @return
 */
torch::Tensor get_res(torch::Tensor attn, int d_in) {

    return (std::get<0>(torch::max(attn, 1)) * d_in).toType(torch::kInt32);
}
