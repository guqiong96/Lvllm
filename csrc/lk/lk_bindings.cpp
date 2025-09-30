#include "moe.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(_lk_C, m) {
    m.doc() = "MOE (Mixture of Experts) bindings";
 
 
    py::class_<MOEConfig>(m, "MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size,
                         int intermediate_size, int stride, int group_min_len,
                         int group_max_len, intptr_t gate_proj,
                         intptr_t up_proj, intptr_t down_proj, int gate_type,
                         int up_type, int down_type, int hidden_type) {
            return MOEConfig(expert_num, routed_expert_num, hidden_size,
                             intermediate_size, stride, group_min_len,
                             group_max_len, (void *)gate_proj, (void *)up_proj,
                             (void *)down_proj, (ggml_type)gate_type,
                             (ggml_type)up_type, (ggml_type)down_type,
                             (ggml_type)hidden_type);
        }));
 
    py::class_<MOE>(m, "MOE")
        .def(py::init<MOEConfig>())
        .def("warm_up", &MOE::warm_up)
        .def("sync_with_cuda_stream",
            [](MOE& self, intptr_t user_cuda_stream) {
                self.sync_with_cuda_stream(user_cuda_stream);
            })
        .def("submit_with_cuda_stream", 
            [](MOE& self, intptr_t user_cuda_stream, int qlen, int k, intptr_t expert_ids, 
            intptr_t weights, intptr_t input, intptr_t output, intptr_t batch_size_tensor) {
                self.submit_with_cuda_stream(
                    user_cuda_stream, 
                    qlen, 
                    k, 
                    (const uint64_t*)expert_ids, 
                    (const float*)weights, 
                    (const void*)input, 
                    (void*)output,
                    (int *)batch_size_tensor
                );
            }
        )
        .def("forward",
            [](MOE& self,
               int qlen,
               int k,
               intptr_t expert_ids,
               intptr_t weights,
               intptr_t input,
               intptr_t output,
               intptr_t batch_size_tensor) {
                 
                
                self.forward(
                    qlen,
                    k,
                    (const uint64_t *)expert_ids,
                    (const float *)weights,
                    (const void *)input,
                    (void *)output,
                    (int *)batch_size_tensor
                );
            });
}