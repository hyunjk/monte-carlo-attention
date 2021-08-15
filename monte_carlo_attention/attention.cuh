//
// Created by hyunjk on 21. 8. 12..
//

#ifndef MCA_ATTENTION_CUH
#define MCA_ATTENTION_CUH

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define UPDIV(x, y) (x / y + (x % y != 0))

namespace mca {

    torch::Tensor eval_sampling_prob_cdf(const torch::Tensor &weight, int num_heads);

    torch::Tensor monte_carlo_multihead_attention(
            const torch::Tensor &attn, const torch::Tensor &input, const torch::Tensor &weight,
            const torch::Tensor &bias,
            const torch::Tensor &num_trials, const torch::Tensor &sampling_prob_cdf);

    __global__ void monte_carlo_multihead_attention_kernel(
            float *input, float *weight, int *num_trials,
            float *sampling_prob_cdf, float *random_samples,
            float *value,
            int num_heads, int seq_length, int input_dim, int output_dim);
}
#endif //MCA_ATTENTION_CUH
