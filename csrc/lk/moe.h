/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : guqiong96
 * @LastEditTime : 2025-08-12 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_MOE_H
#define CPUINFER_OPERATOR_MOE_H
 
#include <cuda_runtime.h> 

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>

#include "backend_numa.h"
#include "conversion.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "llamafile/sgemm.h"
// #define __AMX_INT8__ 1
// #define __AVX512VNNI__ 1
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    #include "amx_gemm.hpp"
#endif
#include <queue> 

class MOE;
struct ForwardParams {
    MOE* moe_ptr;
    int qlen;
    int k;
    const uint64_t* expert_ids;
    const float* weights;
    const void* input;
    void* output;
    int* bsz_tensor;
    cudaEvent_t* forward_done_event;
};
 
struct MOEConfig {
    int expert_num;
    int routed_expert_num;
    int hidden_size;
    int intermediate_size;
    int stride;
    int group_min_len;
    int group_max_len;
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type up_type;
    ggml_type down_type;
    ggml_type hidden_type;

    MOEConfig() {}

    MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, void* gate_proj, void* up_proj, void* down_proj, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type)
        : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size), intermediate_size(intermediate_size), stride(stride), group_min_len(group_min_len), group_max_len(group_max_len), gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {}
};

class MOE {
   public:
    MOE(MOEConfig);
    ~MOE();
    void warm_up();
    void forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, int* bsz_tensor);
    void submit_with_cuda_stream(intptr_t user_cuda_stream, int qlen, int k, const uint64_t* expert_ids, 
                             const float* weights, const void* input, void* output, int* bsz_tensor);
    void sync_with_cuda_stream(intptr_t user_cuda_stream);
    void sync();
   private:
    void forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output);
    void forward_many_m(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output);
    using ForwardOneImpl = void (MOE::*)(int, const uint64_t*, const float*, const void*, void*);
    using ForwardManyImpl = void (MOE::*)(int, int, const uint64_t*, const float*, const void*, void*);
    ForwardOneImpl forward_one_impl;
    ForwardManyImpl forward_many_impl;
    MOEConfig config_;
    void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]
 
    std::vector<void*> gate_numa_;  // [numa_num, nth * stride * expert_num * hidden_size ( /32 if quantized)]
    std::vector<void*> up_numa_;    // [numa_num, nth * stride * expert_num * hidden_size ( /32 if quantized)]
    std::vector<void*> down_numa_;  // [numa_num, nth * stride * expert_num *  intermediate_size ( /32 if quantized)]
    std::vector<size_t> gate_numa_size_;
    std::vector<size_t> up_numa_size_;
    std::vector<size_t> down_numa_size_; 
    size_t stride_gate_bytes_;
    size_t stride_up_bytes_; 
    size_t stride_down_bytes_;
    size_t amx_stride_gate_bytes_;
    size_t amx_stride_up_bytes_; 
    size_t amx_stride_down_bytes_;
    struct NumaBlock {
        int node_id;
        int start_block;
        int num_blocks;
    };
    std::vector<NumaBlock> gate_up_blocks_;
    std::vector<NumaBlock> down_blocks_;

    float* s_input_fp32_;                      // [hidden_size]
    float* s_gate_output_;        // [routed_expert_num, intermediate_size]
    float* s_up_output_;          // [routed_expert_num, intermediate_size]
    float* s_down_output_;        // [routed_expert_num, hidden_size] 
    uint8_t* s_gate_input_;                    // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* s_up_input_;                      // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    uint8_t* s_down_input_;       // [routed_expert_num, intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
  
 

    size_t hidden_type_size;
    size_t hidden_blk_size;
    size_t hidden_bytes;

    ggml_type gate_vec_type;
    size_t gate_type_size;
    size_t gate_blk_size;
    size_t gate_bytes;

    ggml_type up_vec_type;
    size_t up_type_size;
    size_t up_blk_size;
    size_t up_bytes;

    ggml_type down_vec_type;
    size_t down_type_size;
    size_t down_blk_size;
    size_t down_bytes;

    float* input_fp32_;        //[ group_max_len * hidden_size]
    uint8_t* gate_input_;      //[ group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* up_input_;        //[ group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    float* gate_output_;       //[ group_max_len * routed_expert_num * intermediate_size]
    float* up_output_;         //[ group_max_len * routed_expert_num * intermediate_size]
    uint8_t* down_input_;      //[ group_max_len * routed_expert_num * intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    float* down_output_;       //[ group_max_len * routed_expert_num * hidden_size]
    float* output_fp32_;       //[ group_max_len * hidden_size]

    void* m_gate_input_;      //[ group_max_len * routed_expert_num * hidden_size //* ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    void* m_up_input_;        //[ group_max_len * routed_expert_num * hidden_size //* ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    bool use_fp32_buffer_; 

    std::atomic<bool> sync_flag{false};

    
};




#endif