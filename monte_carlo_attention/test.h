//
// Created by hyunjk on 21. 8. 13..
//

#ifndef RAA_CUDA_MAIN_H
#define RAA_CUDA_MAIN_H

torch::Tensor get_res(torch::Tensor attn, int d_in);


torch::Tensor get_p_cdf(torch::Tensor w, int h);


torch::Tensor
raa(torch::Tensor attn, torch::Tensor res, torch::Tensor input, torch::Tensor weight, torch::Tensor p_cdf);

#endif