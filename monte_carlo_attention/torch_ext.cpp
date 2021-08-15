//
// Created by hyunjk on 21. 8. 13..
//

#include <torch/extension.h>
#include "attention.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("monte_carlo_multihead_attention", &mca::monte_carlo_multihead_attention);
    m.def("eval_sampling_prob_cdf", &mca::eval_sampling_prob_cdf);
}
