

#include "attention.cuh"

namespace mca {

    curandGenerator_t gen;
    bool curand_init = false;

    void random_uniform(float *out, int n) {
        if (!curand_init) {
            curand_init = true;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen, time(nullptr));
        }
        curandGenerateUniform(gen, out, n);
    }

    /**
     *
     * @param weight (output_dim, input_dim)
     * @param num_heads
     * @return
     */
    torch::Tensor eval_sampling_prob_cdf(const torch::Tensor &weight, int num_heads) {
        CHECK_INPUT(weight)

        auto w = weight.t();

        const int input_dim = (int) weight.size(0);
        const int output_dim = (int) weight.size(1);
        const int head_dim = output_dim / num_heads;

        w = w.view({input_dim, num_heads, head_dim});

        auto p = torch::norm(w, 2, 2);
        p = torch::pow(p, 2);
        p = p / torch::sum(p, 0);
        auto p_cdf = torch::cumsum(p, 0);
        p_cdf = torch::transpose(p_cdf, 0, 1).contiguous();

        return p_cdf;
    }


    /**
     *
     * Approximate multi-head attention
     *
     * @param attn (num_heads, seq_length, seq_length)
     * @param input (seq_length, input_dim) 입력 feature map
     * @param weight (output_dim, input_dim) PyTorch와 바로 호환 가능하도록 output이 먼저 오는 형식.
     * @param bias (output_dim)
     * @param num_trials (num_heads, seq_length) 몇 개의 행을 랜덤 샘플링할지의 시행 횟수
     * @param sampling_prob_cdf (num_heads, input_dim) 몇 개의 행을 랜덤 샘플링할지의 시행 횟수
     * @return (seq_length, output_dim)
     */
    torch::Tensor monte_carlo_multihead_attention(
            const torch::Tensor &attn, const torch::Tensor &input, const torch::Tensor &weight,
            const torch::Tensor &bias,
            const torch::Tensor &num_trials, const torch::Tensor &sampling_prob_cdf) {

        CHECK_INPUT(input)
        CHECK_INPUT(weight)
        CHECK_INPUT(bias)
        CHECK_INPUT(num_trials)
        CHECK_INPUT(sampling_prob_cdf)

        const int num_heads = (int) num_trials.size(0);
        const int seq_length = (int) num_trials.size(1);
        const int output_dim = (int) weight.size(0);
        const int input_dim = (int) weight.size(1);
        const int head_dim = output_dim / num_heads;

        TORCH_CHECK(output_dim % num_heads == 0, "invalid head size in num_trials")

        torch::Tensor value = torch::empty({num_heads, seq_length, head_dim},
                                           torch::device(weight.device()).dtype(torch::kFloat32));

        float *random_samples;
        cudaMalloc(&random_samples, num_heads * input_dim * sizeof(float));
        random_uniform(random_samples, num_heads * input_dim);

        dim3 grid_dim(num_heads, seq_length);
        const int buffer_size = (3 * input_dim) * (int) sizeof(float);

        monte_carlo_multihead_attention_kernel<<<grid_dim, head_dim, buffer_size>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                num_trials.data_ptr<int>(),
                sampling_prob_cdf.data_ptr<float>(),
                random_samples,
                value.data_ptr<float>(),
                num_heads, seq_length, input_dim, output_dim);
        cudaFree(random_samples);

        cudaDeviceSynchronize();

        // check for error
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            // print the CUDA error message and exit
            printf("after CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }

        torch::Tensor output = torch::matmul(attn, value + bias.view({num_heads, 1, head_dim}));
        output = torch::transpose(output, 0, 1).reshape({seq_length, output_dim});

        return output;

    }


    /**
     *
     * @param input (seq_length, input_dim)
     * @param weight (output_dim, input_dim)
     * @param num_trials (num_heads, seq_length)
     * @param sampling_prob_cdf (num_heads, input_dim)
     * @param random_samples (num_heads, input_dim)
     * @param value (num_heads, seq_length, head_dim)
     * @param num_heads
     * @param seq_length
     * @param input_dim
     * @param output_dim
     */
    __global__ void monte_carlo_multihead_attention_kernel(
            float *input, float *weight, int *num_trials,
            float *sampling_prob_cdf, float *random_samples,
            float *value,
            int num_heads, int seq_length, int input_dim, int output_dim) {

        extern __shared__ float buffer[];

        int *sampled_indices = (int *) buffer;
        float *sampling_prob_cdf_buf = &buffer[input_dim];
        float *ramdom_samples_buf = &buffer[input_dim * 2];

        const int head_idx = (int) blockIdx.x;
        const int seq_idx = (int) blockIdx.y;
        const int t_idx = (int) threadIdx.x;

        const int head_dim = output_dim / num_heads;


        // 기본 가정: D_IN > HD_OUT
        const int chunk_size = UPDIV(input_dim, head_dim); // 만약 input_dim == output_dim 이었다면 chunk_size는 h임.

        const int num_samples = num_trials[head_idx * seq_length + seq_idx];
        // outer product
        float prod_sum = 0.0;

        if (num_samples >= input_dim) {

            for (int i = 0; i < num_samples; i++) {

                int inp_idx = seq_idx * input_dim + i;
                int w_idx = head_idx * head_dim * input_dim + t_idx * input_dim + i;

                prod_sum += input[inp_idx] * weight[w_idx];
            }
            prod_sum /= (float) num_samples / (float) input_dim;

        } else if (num_samples > 0) {

            // sampling_prob_cdf 로드
            for (int i = 0; i < chunk_size; i++) {
                int elem_idx = t_idx * chunk_size + i;
                if (elem_idx < input_dim) {
                    sampling_prob_cdf_buf[elem_idx] = sampling_prob_cdf[head_idx * input_dim + elem_idx];
                }
                if (elem_idx < num_samples) {
                    ramdom_samples_buf[elem_idx] = random_samples[head_idx * input_dim + elem_idx];
                }
            }
            __syncthreads();

            // weighted random pick
            for (int i = 0; i < num_samples; i++) {
                float p_rand = ramdom_samples_buf[i];


                for (int j = 0; j < chunk_size; j++) {
                    int elem_idx = t_idx * chunk_size + j;
                    if (elem_idx < input_dim) {

                        float p_cdf_l = 0.0;
                        float p_cdf_r = 1.0;

                        if (elem_idx > 0) {
                            p_cdf_l = sampling_prob_cdf_buf[elem_idx - 1];
                        }

                        if (elem_idx < input_dim - 1) {
                            p_cdf_r = sampling_prob_cdf_buf[elem_idx];
                        }

                        // c.f.  0.0 is excluded and 1.0 is included in cuRAND
                        if (p_cdf_l < p_rand && p_rand <= p_cdf_r) {
                            sampled_indices[i] = elem_idx;
                            break;
                        }
                    }
                }
            }
            __syncthreads();


            for (int i = 0; i < num_samples; i++) {
                int sample_idx = sampled_indices[i];
                float prob = sampling_prob_cdf_buf[sample_idx];
                if (sample_idx > 0) {
                    prob -= sampling_prob_cdf_buf[sample_idx - 1];
                }

                int inp_idx = seq_idx * input_dim + sample_idx;
                int w_idx = head_idx * head_dim * input_dim + t_idx * input_dim + sample_idx;

                prod_sum += input[inp_idx] * weight[w_idx] / ((float) num_samples * prob);
            }

        }

        // 계산결과 메모리에 쓰기
        int out_idx = head_idx * seq_length * head_dim + seq_idx * head_dim + t_idx;
        value[out_idx] = prod_sum;

    }

}
