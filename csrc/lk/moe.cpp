/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : guqiong96
 * @LastEditTime : 2025-08-12 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "moe.h"
#include <iostream>
#include <cstdint>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif 
#include <mutex>
static std::mutex print_mutex;


MOE::MOE(MOEConfig config) { 
    
    config_ = config;

    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    config_.stride = 32;
    use_fp32_buffer_ = false;
    #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    // std::cout << "AMX enabled ...... " << std::endl;
    if(config_.gate_type == GGML_TYPE_F16){
        use_fp32_buffer_ = true;
        if(config_.up_type != GGML_TYPE_F16 || config_.down_type != GGML_TYPE_F16){
            throw std::runtime_error("GGML_TYPE_F16 not same with gate,up,down");
        }
    }    
    #endif
    // if(config_.up_type == GGML_TYPE_BF16 && config_.hidden_type == GGML_TYPE_BF16){
    //     #if defined(__AMX_INT8__) || defined(__AVX512VNNI__) || defined(__AVX512BF16__) || defined(__AVX512F__)

    //         // std::cout << "amx is enabled for bf16 ...... " << std::endl;
    //     #else
    //         use_fp32_buffer_ = true;
    //     #endif
    // }

    
    hidden_type_size = ggml_type_size(config_.hidden_type);
    hidden_blk_size = ggml_blck_size(config_.hidden_type);
    hidden_bytes = config_.hidden_size * hidden_type_size / hidden_blk_size;

    gate_vec_type = ggml_internal_get_type_traits(config_.gate_type).vec_dot_type;
    gate_type_size = ggml_type_size(gate_vec_type);
    gate_blk_size = ggml_blck_size(gate_vec_type); 
    gate_bytes = config_.hidden_size * gate_type_size / gate_blk_size;

    up_vec_type = ggml_internal_get_type_traits(config_.up_type).vec_dot_type;
    up_type_size = ggml_type_size(up_vec_type);
    up_blk_size = ggml_blck_size(up_vec_type); 
    up_bytes = config_.hidden_size * up_type_size / up_blk_size;

    down_vec_type = ggml_internal_get_type_traits(config_.down_type).vec_dot_type;
    down_type_size = ggml_type_size(down_vec_type);
    down_blk_size = ggml_blck_size(down_vec_type); 
    down_bytes = config_.intermediate_size * down_type_size / down_blk_size;
    // std::cout << "config_.stride : " << config_.stride << " down_blk_size :" << down_blk_size << " hidden_blk_size :" << hidden_blk_size << std::endl;
    // std::cout << "config_.hidden_type : " << ggml_internal_get_type_traits(config_.hidden_type).type_name << std::endl;
    // std::cout << "config_.gate_type : " << ggml_internal_get_type_traits(config_.gate_type).type_name << std::endl;
    // std::cout << "config_.up_type : " << ggml_internal_get_type_traits(config_.up_type).type_name << std::endl;
    // std::cout << "config_.down_type : " << ggml_internal_get_type_traits(config_.down_type).type_name << std::endl; 
    // std::cout << "gate_type_size: " << gate_type_size << std::endl;
    // std::cout << "gate_blk_size: " << gate_blk_size << std::endl;
    // std::cout << "gate_bytes: " << gate_bytes << std::endl;
    // std::cout << "up_type_size: " << up_type_size << std::endl;
    // std::cout << "up_blk_size: " << up_blk_size << std::endl;
    // std::cout << "up_bytes: " << up_bytes << std::endl;
    // std::cout << "down_type_size: " << down_type_size << std::endl;
    // std::cout << "down_blk_size: " << down_blk_size << std::endl;
    // std::cout << "down_bytes: " << down_bytes << std::endl;
    // std::cout << "config_.hidden_size: " << config_.hidden_size << std::endl;
    // std::cout << "config_.intermediate_size: " << config_.intermediate_size << std::endl; 
    // std::cout << "gate_vec_type: " << gate_vec_type << std::endl; 
    
    gate_numa_.resize(numa_nodes_);
    up_numa_.resize(numa_nodes_);
    down_numa_.resize(numa_nodes_); 

    gate_numa_size_.resize(numa_nodes_);
    up_numa_size_.resize(numa_nodes_);
    down_numa_size_.resize(numa_nodes_);  

    int nth = config_.intermediate_size / config_.stride;
    stride_gate_bytes_ = config_.stride * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
    stride_up_bytes_ = config_.stride * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
    
    #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    amx_stride_gate_bytes_ = get_amx_packed_size(config_.gate_type, config_.hidden_size, config_.stride);
    amx_stride_up_bytes_ = get_amx_packed_size(config_.up_type, config_.hidden_size, config_.stride);
    #endif
    int base = nth / numa_nodes_;
    int remain = nth % numa_nodes_;

    gate_up_blocks_.resize(numa_nodes_);
    int current_block = 0; 
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int n_blocks = (base + (nid < remain));
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        gate_numa_size_[nid] = config_.expert_num * n_blocks * amx_stride_gate_bytes_;
        up_numa_size_[nid] = config_.expert_num * n_blocks * amx_stride_up_bytes_;
        #else
        gate_numa_size_[nid] = config_.expert_num * n_blocks * stride_gate_bytes_;
        up_numa_size_[nid] = config_.expert_num * n_blocks * stride_up_bytes_;
        #endif
        gate_up_blocks_[nid] = NumaBlock{
            .node_id = nid,
            .start_block = current_block,
            .num_blocks = n_blocks
        };
        current_block += n_blocks; 
    }
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, numa_nodes_, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;  
        int start_block = gate_up_blocks_[nid].start_block;
        int num_blocks = gate_up_blocks_[nid].num_blocks; 

        if (num_blocks == 0) return;
        assert(nid == task_id);
        gate_numa_[nid] = allocate_aligned_numa(gate_numa_size_[nid], nid);
        up_numa_[nid] = allocate_aligned_numa(up_numa_size_[nid], nid);
    }, nullptr);
   
     Backend_NUMA::getInstance().do_k_work_stealing_job(config_.expert_num, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = gate_up_blocks_[nid].start_block;
        int num_blocks = gate_up_blocks_[nid].num_blocks; 

        if (num_blocks == 0) return;
 
        int x = task_id - start_block * config_.expert_num;
        int expert_id = x / num_blocks;  

        int offset = x % num_blocks;
        int ith = start_block + offset;  
        
        void* gate_ptr = (uint8_t*)gate_proj_ + (expert_id * nth + ith) * stride_gate_bytes_;
        void* up_ptr = (uint8_t*)up_proj_ +  (expert_id * nth + ith) * stride_up_bytes_;
      
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__) 
        uint8_t* local_gate_ptr = (uint8_t*)gate_numa_[nid] + (expert_id * num_blocks + offset) * amx_stride_gate_bytes_;
        uint8_t* local_up_ptr = (uint8_t*)up_numa_[nid] + (expert_id * num_blocks + offset) * amx_stride_up_bytes_;
        convert_weight_to_amx_format(
            local_gate_ptr,
            gate_ptr,
            config_.gate_type,
            config_.hidden_size,
            config_.stride
        );
        convert_weight_to_amx_format(
            local_up_ptr,
            up_ptr,
            config_.up_type,
            config_.hidden_size,
            config_.stride
        ); 
#else
        void* local_gate_ptr = (uint8_t*)gate_numa_[nid] + (expert_id * num_blocks + offset) * stride_gate_bytes_;
        void* local_up_ptr = (uint8_t*)up_numa_[nid] + (expert_id * num_blocks + offset) * stride_up_bytes_;
        memcpy(local_gate_ptr, gate_ptr, stride_gate_bytes_);
        memcpy(local_up_ptr, up_ptr, stride_up_bytes_);
#endif
    }, nullptr);

    nth = config_.hidden_size / config_.stride;
    stride_down_bytes_ = config_.stride * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
    
    #if defined(__AMX_INT8__) && defined(__AVX512VNNI__) 
    amx_stride_down_bytes_ = get_amx_packed_size(config_.down_type, config_.intermediate_size, config_.stride);
    #endif
  
    base = nth / numa_nodes_;
    remain = nth % numa_nodes_;
    down_blocks_.resize(numa_nodes_);
    current_block = 0;
    for (int nid = 0; nid < numa_nodes_; nid++) { 
        int n_blocks = (base + (nid < remain));
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__) 
        down_numa_size_[nid] = config_.expert_num * n_blocks * amx_stride_down_bytes_;
        #else
        down_numa_size_[nid] = config_.expert_num * n_blocks * stride_down_bytes_;
        #endif
        down_blocks_[nid] = NumaBlock{
            .node_id = nid,
            .start_block = current_block,
            .num_blocks = n_blocks
        };
        
        current_block += n_blocks;  
    }  
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, numa_nodes_, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;
 
        if (num_blocks == 0) return; 
        assert(nid == task_id);
        down_numa_[nid] = allocate_aligned_numa(down_numa_size_[nid], nid);
    }, nullptr);
    
   
 
    Backend_NUMA::getInstance().do_k_work_stealing_job(config_.expert_num, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;
 
        if (num_blocks == 0) return;
 
        int x = task_id - start_block * config_.expert_num;
        int expert_id = x / num_blocks;  

        int offset = x % num_blocks;
        int ith = start_block + offset; 

        
        void* down_ptr = (uint8_t*)down_proj_ + (expert_id * nth + ith) * stride_down_bytes_;  

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)  
        uint8_t* local_down_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks + offset) * amx_stride_down_bytes_;
        
        convert_weight_to_amx_format(
            local_down_ptr,
            down_ptr,
            config_.down_type,
            config_.intermediate_size,
            config_.stride
        );
#else  
        void* local_down_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks + offset) * stride_down_bytes_;
        memcpy(local_down_ptr, down_ptr, stride_down_bytes_);
#endif
    }, nullptr);
    
    s_input_fp32_ = (float*)allocate_aligned(sizeof(float) * config_.hidden_size);
    s_gate_output_ = (float*)allocate_aligned(config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    s_up_output_ = (float*)allocate_aligned(config_.routed_expert_num * sizeof(float) * config_.intermediate_size); 
    s_down_output_ = (float*)allocate_aligned(config_.routed_expert_num * sizeof(float) * config_.hidden_size);
    if(!use_fp32_buffer_){
        s_gate_input_ = (uint8_t*)allocate_aligned(gate_bytes);
        s_up_input_ = (uint8_t*)allocate_aligned(up_bytes);
        s_down_input_ = (uint8_t*)allocate_aligned(config_.routed_expert_num * down_bytes);
    }
    
 
    input_fp32_ = (float*)allocate_aligned(config_.group_max_len * sizeof(float) * config_.hidden_size);
    gate_output_ = (float*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    up_output_ = (float*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    down_output_ = (float*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.hidden_size);
    output_fp32_ = (float*)allocate_aligned(config_.group_max_len * sizeof(float) * config_.hidden_size);  
    if(!use_fp32_buffer_){
        gate_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * gate_bytes);
        up_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * up_bytes);
        down_input_ = (uint8_t*)allocate_aligned(config_.group_max_len * config_.routed_expert_num * down_bytes);
        m_gate_input_ = (uint8_t*)allocate_aligned( config_.group_max_len * config_.routed_expert_num * hidden_bytes);
        m_up_input_ = (uint8_t*)allocate_aligned( config_.group_max_len * config_.routed_expert_num * hidden_bytes);
    }else{
        m_gate_input_ = (float*)allocate_aligned( config_.group_max_len * config_.routed_expert_num * sizeof(float) *  config_.hidden_size); 
    }
  
    forward_one_impl = &MOE::forward_one;
    forward_many_impl = &MOE::forward_many_m;

    config_.gate_proj = nullptr;
    config_.up_proj = nullptr;
    config_.down_proj = nullptr;

    
}

MOE::~MOE() {
    for (int nid = 0; nid < numa_nodes_; nid++) {  
        free_aligned_numa(gate_numa_[nid], gate_numa_size_[nid]);
        free_aligned_numa(up_numa_[nid], up_numa_size_[nid]);
        free_aligned_numa(down_numa_[nid], down_numa_size_[nid]);
    }  
    free_aligned(s_input_fp32_, sizeof(float) * config_.hidden_size);
    free_aligned(s_gate_output_, config_.group_max_len * sizeof(float) * config_.intermediate_size);
    free_aligned(s_up_output_, config_.group_max_len * sizeof(float) * config_.intermediate_size); 
    free_aligned(s_down_output_, config_.group_max_len * sizeof(float) * config_.hidden_size);
    if(!use_fp32_buffer_){
        free_aligned(s_gate_input_, gate_bytes);
        free_aligned(s_up_input_, up_bytes);
        free_aligned(s_down_input_, config_.group_max_len * down_bytes);
    }

    free_aligned(input_fp32_, config_.group_max_len * sizeof(float) * config_.hidden_size);
    free_aligned(gate_output_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    free_aligned(up_output_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.intermediate_size);
    free_aligned(down_output_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.hidden_size);
    free_aligned(output_fp32_, config_.group_max_len * sizeof(float) * config_.hidden_size); 
    if(!use_fp32_buffer_){
        free_aligned(gate_input_ , config_.group_max_len * gate_bytes);
        free_aligned(up_input_ , config_.group_max_len * up_bytes);
        free_aligned(down_input_ , config_.group_max_len * config_.routed_expert_num * down_bytes); 
        free_aligned(m_gate_input_, config_.group_max_len * config_.routed_expert_num * hidden_bytes);
        free_aligned(m_up_input_ , config_.group_max_len * config_.routed_expert_num * hidden_bytes);
    }else{
        free_aligned(m_gate_input_, config_.group_max_len * config_.routed_expert_num * sizeof(float) * config_.hidden_size);
    }
}

void MOE::warm_up() {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(hidden_bytes);
    std::vector<uint8_t> output(hidden_bytes);
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
    for (int i = 0; i < config_.expert_num; i++) {
        uint64_t expert_ids = i;
        float weights = 0;
        forward_one(1, &expert_ids, &weights, input.data(), output.data());  
    }
}



static void act_fn(float* up, float* gate, int n) {
 
#if defined(__AVX2__)
    constexpr int VEC_SIZE = 8;
    const __m256 v_log2e = _mm256_set1_ps(1.44269504089f);  
    const __m256 v_ln2 = _mm256_set1_ps(0.69314718056f);  
    const __m256 v_one = _mm256_set1_ps(1.0f);
    const __m256 v_neg_inf = _mm256_set1_ps(-128.0f);       
    const __m256 v_pos_inf = _mm256_set1_ps(127.0f);       
    for (int i = 0; i < n; i += VEC_SIZE) {
        __m256 v_gate = _mm256_load_ps(gate + i);  
        __m256 v_up = _mm256_load_ps(up + i);      

        __m256 v_x = _mm256_mul_ps(v_gate, v_log2e);

        v_x = _mm256_max_ps(_mm256_min_ps(v_x, v_pos_inf), v_neg_inf);

        __m256i v_k = _mm256_cvtps_epi32(v_x);         
        __m256 v_k_f = _mm256_cvtepi32_ps(v_k);        
        __m256 v_r = _mm256_sub_ps(v_x, v_k_f);         

        __m256i v_k_bias = _mm256_add_epi32(v_k, _mm256_set1_epi32(127));  
        __m256i v_k_bits = _mm256_slli_epi32(v_k_bias, 23);              
        __m256 v_two_k = _mm256_castsi256_ps(v_k_bits);                 

        __m256 v_t = _mm256_mul_ps(v_r, v_ln2);                       
        __m256 v_t2 = _mm256_mul_ps(v_t, v_t);                        
        __m256 v_t3 = _mm256_mul_ps(v_t2, v_t);                     
        __m256 v_t4 = _mm256_mul_ps(v_t3, v_t);                     
        __m256 v_two_r = _mm256_add_ps(v_one, 
            _mm256_fmadd_ps(v_t, v_one, 
                _mm256_fmadd_ps(v_t2, _mm256_set1_ps(1.0f/2.0f), 
                    _mm256_fmadd_ps(v_t3, _mm256_set1_ps(1.0f/6.0f), 
                        _mm256_mul_ps(v_t4, _mm256_set1_ps(1.0f/24.0f))))));

        __m256 v_two_x = _mm256_mul_ps(v_two_k, v_two_r);

        __m256 v_denom = _mm256_add_ps(v_one, v_two_x);
        __m256 v_sigmoid = _mm256_div_ps(v_two_x, v_denom);

        __m256 v_swish = _mm256_mul_ps(v_gate, v_sigmoid);
        __m256 v_out = _mm256_mul_ps(v_up, v_swish);

        _mm256_store_ps(up + i, v_out);
    }
#else 
    for (int i = 0; i < n; ++i) {
        up[i] = up[i] * (gate[i] / (1.0f + expf(-gate[i])));
    }
#endif
}

void MOE::forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output) {
    
    const void* gate_input_ptr;
    const void* up_input_ptr;
    size_t gate_input_em = config_.hidden_size / ggml_blck_size(config_.gate_type);
    size_t up_input_em = config_.hidden_size / ggml_blck_size(config_.up_type);
    size_t down_input_em = config_.intermediate_size / ggml_blck_size(config_.down_type);

    if(use_fp32_buffer_){
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        gate_input_ptr = up_input_ptr = s_input_fp32_;
        gate_input_em = up_input_em = config_.hidden_size;
        down_input_em = config_.intermediate_size; 
    }else{
        if (config_.hidden_type == gate_vec_type && config_.hidden_type == up_vec_type) {
            gate_input_ptr = up_input_ptr = input;
        } else {
            to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
            if (gate_vec_type == up_vec_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, gate_vec_type);
                gate_input_ptr = up_input_ptr = s_gate_input_;
            } else {
                if (config_.hidden_type != gate_vec_type) {
                    from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, gate_vec_type);
                    gate_input_ptr = s_gate_input_;
                } else {
                    gate_input_ptr = input;
                }
                if (config_.hidden_type != up_vec_type) {
                    from_float(s_input_fp32_, s_up_input_, config_.hidden_size, up_vec_type);
                    up_input_ptr = s_up_input_;
                } else {
                    up_input_ptr = input;
                }
            } 
        }
    }  
    
    size_t nth = config_.intermediate_size / config_.stride; 
    Backend_NUMA::getInstance().do_k_work_stealing_job(k, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = gate_up_blocks_[nid].start_block;
        int num_blocks = gate_up_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;

        int x = task_id - start_block * k;
        int expert_idx = x / num_blocks; 
        int expert_id = expert_ids[expert_idx];
        int offset = x % num_blocks; 
        int ith = start_block + offset;
        size_t n_stride = config_.stride;

        size_t offsets_i = expert_idx * config_.intermediate_size;
        
        float* gate_output_ptr = s_gate_output_ + offsets_i + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        uint8_t* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (expert_id * num_blocks + offset) * amx_stride_gate_bytes_; 
        amx_gemm_compute(config_.gate_type, gate_proj_ptr, gate_input_ptr, gate_output_ptr, 1, n_stride, config_.hidden_size, n_stride);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (expert_id * num_blocks + offset) * stride_gate_bytes_;
        bool is_supported = llamafile_sgemm(n_stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, gate_input_em, gate_output_ptr, n_stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, use_fp32_buffer_ ? GGML_TYPE_F32 : gate_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        #endif
        
        float* up_output_ptr = s_up_output_ + offsets_i + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__) 
        uint8_t* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (expert_id * num_blocks  + offset) * amx_stride_up_bytes_;
        amx_gemm_compute(config_.up_type, up_proj_ptr, up_input_ptr, up_output_ptr, 1, n_stride, config_.hidden_size, n_stride);
        #else
        void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (expert_id * num_blocks  + offset) * stride_up_bytes_;
        llamafile_sgemm(n_stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, up_input_em, up_output_ptr, n_stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, use_fp32_buffer_ ? GGML_TYPE_F32 : up_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        #endif
        act_fn(up_output_ptr, gate_output_ptr , n_stride);  
        if (config_.stride % down_blk_size == 0 && !use_fp32_buffer_) {
            void* down_input_ptr = s_down_input_ + (offsets_i + ith * config_.stride) * down_type_size / down_blk_size;
            from_float(up_output_ptr, down_input_ptr, n_stride, down_vec_type);
        }
    }, nullptr);
    if (config_.stride % down_blk_size != 0 && !use_fp32_buffer_) {
        Backend_NUMA::getInstance().do_k_work_stealing_job(1, k, nullptr, [&](int task_id) {
            int expert_idx = task_id;
            float* up_output_ptr = s_up_output_ + expert_idx * config_.intermediate_size;
            void* down_input_ptr = s_down_input_ + expert_idx * config_.intermediate_size * down_type_size / down_blk_size;
            from_float(up_output_ptr, down_input_ptr, config_.intermediate_size, down_vec_type);
        }, nullptr);
    }
    nth = config_.hidden_size / config_.stride; 
    Backend_NUMA::getInstance().do_k_work_stealing_job(k, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;

        int x = task_id - start_block * k;
        int expert_idx = x / num_blocks; 
        int expert_id = expert_ids[expert_idx];
        int offset = x % num_blocks; 
        int ith = start_block + offset;
        size_t n_stride = config_.stride;
        void* down_input_ptr;
        if(use_fp32_buffer_){
            down_input_ptr = s_up_output_ + expert_idx * config_.intermediate_size; 
        }else{
            down_input_ptr = s_down_input_ + expert_idx * config_.intermediate_size * down_type_size / down_blk_size; 
        }
        float* down_output_ptr = s_down_output_ + expert_idx * config_.hidden_size + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__) 
        uint8_t* down_proj_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks  + offset) * amx_stride_down_bytes_;
        amx_gemm_compute(config_.down_type, down_proj_ptr, down_input_ptr, down_output_ptr, 1, config_.stride, config_.intermediate_size, n_stride);    
        #else
        void* down_proj_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks  + offset) * stride_down_bytes_;
        llamafile_sgemm(n_stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, down_input_em, down_output_ptr, n_stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, use_fp32_buffer_ ? GGML_TYPE_F32 : down_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        #endif
    }, nullptr); 
    nth = config_.hidden_size / config_.stride;  
    
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, nth, nullptr, [&](int task_id) {
        int ith = task_id;
        float * down_output_ptr_0 = s_down_output_ + ith * config_.stride;
        for(int j=0; j<config_.stride; ++j){
            down_output_ptr_0[j] = down_output_ptr_0[j] * weights[0];
        }
        for(int i=1; i<k; ++i){
            for(int j=0; j<config_.stride; ++j){
                float * down_output_ptr = down_output_ptr_0 + i * config_.hidden_size;
                down_output_ptr_0[j] += down_output_ptr[j] * weights[i];
            }
        }
        void * output_ptr = (uint8_t*)output + ith * config_.stride * hidden_type_size / hidden_blk_size;
        from_float(down_output_ptr_0, output_ptr, config_.stride, config_.hidden_type);
    }, nullptr);
}
void MOE::forward_many_m(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output) {
    size_t gate_input_em = config_.hidden_size / ggml_blck_size(config_.gate_type);
    size_t up_input_em = config_.hidden_size / ggml_blck_size(config_.up_type);
    size_t down_input_em = config_.intermediate_size / ggml_blck_size(config_.down_type);
    if(use_fp32_buffer_){ 
        gate_input_em = up_input_em = config_.hidden_size;
        down_input_em = config_.intermediate_size;
    }

    std::vector<int> expert_reorder_offset(config_.expert_num,0);  // [expert_id, offset_in_buffer]       
    std::vector<int> expert_selected_num(config_.expert_num,0);  // [expert_id, num_selected]
    std::vector<std::vector<std::pair<uint64_t, int>>> token_expert_mapping(qlen); // [token_id, expert_idx[expert_id, offset_in_buffer]]
   
   
    for (int i = 0; i < qlen; i++) {   
        token_expert_mapping [i].resize(k);
        for (int j = 0; j < k; j++) { 
            uint64_t expert_id =  expert_ids[i * k + j];
            token_expert_mapping [i][j] = std::make_pair(expert_id, expert_selected_num[expert_id]++);
        }
    }

    uint64_t reorder_offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        expert_reorder_offset[i] = reorder_offset;
        reorder_offset += expert_selected_num[i]; 
    }

    
    int nth = config_.hidden_size / config_.stride; 
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, qlen, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;  
        int token_id = task_id;  
        

        void* input_uint8_ptr = (uint8_t*)input + token_id * hidden_bytes;
        float* input_fp32_ptr = input_fp32_ + token_id * config_.hidden_size;
        void* gate_input_ptr = gate_input_ + token_id * gate_bytes;
        void* up_input_ptr = up_input_ + token_id * up_bytes;
        if(use_fp32_buffer_){
            to_float(input_uint8_ptr, input_fp32_ptr, config_.hidden_size, config_.hidden_type);
        }else{
            if (config_.hidden_type == gate_vec_type && config_.hidden_type == up_vec_type) {
                memcpy(gate_input_ptr, input_uint8_ptr, hidden_bytes);
                // up not need to copy
            } else {
                to_float(input_uint8_ptr, input_fp32_ptr, config_.hidden_size, config_.hidden_type);
                if (gate_vec_type == up_vec_type) {
                    from_float(input_fp32_ptr, gate_input_ptr, config_.hidden_size, gate_vec_type);
                    // up not need to copy
                } else {
                    if (config_.hidden_type != gate_vec_type) {
                        from_float(input_fp32_ptr, gate_input_ptr, config_.hidden_size, gate_vec_type);
                    } else {
                        memcpy(gate_input_ptr, input_uint8_ptr, hidden_bytes);
                    }
                    if (config_.hidden_type != up_vec_type) {
                        from_float(input_fp32_ptr, up_input_ptr, config_.hidden_size, up_vec_type); 
                    } else {
                        memcpy(up_input_ptr, input_uint8_ptr, hidden_bytes);
                    }
                }
            }
        }
    }, nullptr); 
 
    Backend_NUMA::getInstance().do_k_work_stealing_job(1, qlen*k, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_;  
        int token_id = task_id / k;
        int expert_idx =  task_id % k;
        uint64_t expert_id = expert_ids[token_id * k + expert_idx];  
 
        uint64_t expert_id_from_mapping = token_expert_mapping[token_id][expert_idx].first;
         
        if (expert_id_from_mapping != expert_id) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cerr << "MOE::forward_many_m --------------------" << std::endl
                    << "  expert_id : " << expert_id << std::endl 
                    << "  expert_id_from_mapping : " << expert_id_from_mapping << std::endl 
                    << "  task_id : " << task_id << std::endl 
                    << "  token_id : " << token_id << std::endl 
                    << "  expert_idx : " << expert_idx << std::endl
                    << "  qlen : " << qlen << std::endl
                    << "  k : " << k << std::endl
                    << "  token_expert_mapping.size : " << token_expert_mapping.size() << std::endl;
                    
            if (token_id < token_expert_mapping.size()) { 
                std::cerr << "  token_expert_mapping : " << std::endl;
                for (size_t j = 0; j < token_expert_mapping[token_id].size() && j < k; j++) { 
                    std::cerr << "    [" << j << "] : (" 
                            << token_expert_mapping[token_id][j].first 
                            << ", " 
                            << token_expert_mapping[token_id][j].second 
                            << ")"  << std::endl;
                }
            }
            std::abort();
        }
         
        assert(expert_id_from_mapping == expert_id);

        
        int base = expert_reorder_offset[expert_id];
        int pos = token_expert_mapping [token_id][expert_idx].second;
        void* gate_input_ptr;
        void* up_input_ptr;
        void* m_gate_input_ptr;
        void* m_up_input_ptr;
        if(use_fp32_buffer_){
            gate_input_ptr = input_fp32_ + token_id * config_.hidden_size; 
            m_gate_input_ptr = (float*)m_gate_input_ + (base+pos) * config_.hidden_size; 
            memcpy(m_gate_input_ptr, gate_input_ptr, config_.hidden_size*sizeof(float));
        }else{ 
            gate_input_ptr = (uint8_t*)gate_input_ + token_id * gate_bytes;
            up_input_ptr = (uint8_t*)up_input_ + token_id * up_bytes;
            m_gate_input_ptr = (uint8_t*)m_gate_input_ + (base+pos) * gate_bytes;
            m_up_input_ptr = (uint8_t*)m_up_input_ + (base+pos)  * up_bytes;
            memcpy(m_gate_input_ptr, gate_input_ptr, gate_bytes);
            if(gate_vec_type != up_vec_type){
                memcpy(m_up_input_ptr, up_input_ptr, up_bytes); 
            }
        } 
    }, nullptr);  
    
     
    nth = config_.intermediate_size / config_.stride; 
    Backend_NUMA::getInstance().do_k_work_stealing_job(config_.expert_num, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = gate_up_blocks_[nid].start_block;
        int num_blocks = gate_up_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;

        int x = task_id - start_block * config_.expert_num;
        int expert_id = x / num_blocks; 
        if(expert_selected_num[expert_id] == 0) return;

        int offset = x % num_blocks;
        int ith = start_block + offset;

        int expert_offsets = expert_reorder_offset[expert_id];
        int n = expert_selected_num[expert_id];
        size_t n_stride = config_.stride;
        void* gate_input_ptr;
        if(use_fp32_buffer_){
            gate_input_ptr = (float*)m_gate_input_ + expert_offsets * config_.hidden_size;
        }else{
            gate_input_ptr = (uint8_t*)m_gate_input_ + expert_offsets * gate_bytes;
        }

        
        float* gate_output_ptr = gate_output_ + expert_offsets * config_.intermediate_size + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__) 
        void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (expert_id * num_blocks + offset) * amx_stride_gate_bytes_;
        amx_gemm_compute(config_.gate_type, gate_proj_ptr, gate_input_ptr, gate_output_ptr, n, config_.stride, config_.hidden_size, config_.intermediate_size);
        #else 
        void* gate_proj_ptr = (uint8_t*)gate_numa_[nid] +  (expert_id * num_blocks + offset) * stride_gate_bytes_;
        bool sgemm_ok = llamafile_sgemm(n_stride, n, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, gate_input_em, gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, use_fp32_buffer_ ? GGML_TYPE_F32 : gate_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        if(!sgemm_ok){
            std::cout << "MOE::forward_many_m llamafile_sgemm failed" << std::endl;
        }
        #endif  
        void* up_input_ptr;
        if(use_fp32_buffer_){
            up_input_ptr = gate_input_ptr;
        }else{
            up_input_ptr = (gate_vec_type == up_vec_type) 
                    ? gate_input_ptr   
                    : (uint8_t*)m_up_input_ + expert_offsets * up_bytes;
        }  
         
        float* up_output_ptr = up_output_ + expert_offsets * config_.intermediate_size + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__) 
        void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (expert_id * num_blocks + offset) * amx_stride_up_bytes_;
        amx_gemm_compute(config_.up_type, up_proj_ptr, up_input_ptr, up_output_ptr, n, config_.stride, config_.hidden_size, config_.intermediate_size);
        #else
        void* up_proj_ptr = (uint8_t*)up_numa_[nid] +  (expert_id * num_blocks + offset) * stride_up_bytes_;
        llamafile_sgemm(n_stride, n, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, up_input_em, up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, use_fp32_buffer_ ? GGML_TYPE_F32 : up_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        #endif

        for(int i=0; i<n; i++){
            act_fn(up_output_ptr + i*config_.intermediate_size, gate_output_ptr+ i*config_.intermediate_size , n_stride); 
            if(!use_fp32_buffer_){
                if (config_.stride % down_blk_size == 0) { 
                    void* down_input_ptr = down_input_ + ((expert_offsets + i) * config_.intermediate_size + ith * config_.stride) * down_type_size / down_blk_size;
                    from_float(up_output_ptr + i * config_.intermediate_size, down_input_ptr, n_stride, down_vec_type);
                }
            }
        }

        
    }, nullptr);
    if(!use_fp32_buffer_){
        if (config_.stride % down_blk_size != 0) {
            Backend_NUMA::getInstance().do_k_work_stealing_job(1, config_.expert_num, nullptr, [&](int task_id) {
                int nid = Backend_NUMA::numa_node_;  
                int expert_id = task_id;   
                if(expert_selected_num[expert_id] == 0) return;  
                int expert_offsets = expert_reorder_offset[expert_id];
                int n = expert_selected_num[expert_id];
                float* up_output_ptr_ = up_output_ + expert_offsets * config_.intermediate_size;
                void* down_input_ptr = down_input_ + (expert_offsets * config_.intermediate_size) * down_type_size / down_blk_size;
                from_float(up_output_ptr_, down_input_ptr, n * config_.intermediate_size, down_vec_type);
            }, nullptr);
        }    
    }
    // { 
    //     std::lock_guard<std::mutex> lock(print_mutex);
    //     std::cout << "MOE::forward_many_m gate_output_ first 20 elements: " << std::endl;
    //     float* temp = new float[20];
    //     to_float(up_numa_[0], temp, 20, GGML_TYPE_BF16); 
    //     for (int i = 0; i < 20; ++i) {
    //         float value  = *((float*)temp + i) ; 
    //         std::cout  << value;
    //         if (i < 19) {
    //             std::cout << "  ";
    //         } else {
    //             std::cout << std::endl;
    //         }
    //     }
    //     delete[] temp;
    // }
    
    nth = config_.hidden_size / config_.stride;  
    Backend_NUMA::getInstance().do_k_work_stealing_job(config_.expert_num, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;
 
        int x = task_id - start_block * config_.expert_num;
        int expert_id = x / num_blocks; 
        if(expert_selected_num[expert_id] == 0) return;

        int offset = x % num_blocks;
        int ith = start_block + offset;

        int expert_offsets = expert_reorder_offset[expert_id];
        int n = expert_selected_num[expert_id];
        size_t n_stride = config_.stride;
        void* down_input_ptr;
        if(use_fp32_buffer_){
            down_input_ptr = up_output_ + expert_offsets * config_.intermediate_size;
        }else{
            down_input_ptr = down_input_ + expert_offsets * config_.intermediate_size * down_type_size / down_blk_size;
        }
        float* down_output_ptr = down_output_  + expert_offsets * config_.hidden_size + ith * config_.stride;
        #if defined(__AMX_INT8__) && defined(__AVX512VNNI__) 
        uint8_t* down_proj_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks + offset) * amx_stride_down_bytes_;
        amx_gemm_compute(config_.down_type, down_proj_ptr, down_input_ptr, down_output_ptr, n, n_stride, config_.intermediate_size, config_.hidden_size);
        #else
        void* down_proj_ptr = (uint8_t*)down_numa_[nid] + (expert_id * num_blocks + offset) * stride_down_bytes_;
        llamafile_sgemm(n_stride, n, config_.intermediate_size / down_blk_size, down_proj_ptr, config_.intermediate_size / down_blk_size, down_input_ptr, down_input_em, down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, use_fp32_buffer_ ? GGML_TYPE_F32 : down_vec_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        #endif    
    }, nullptr);
    
      Backend_NUMA::getInstance().do_k_work_stealing_job(qlen, nth, nullptr, [&](int task_id) {
        int nid = Backend_NUMA::numa_node_; 
        int start_block = down_blocks_[nid].start_block;
        int num_blocks = down_blocks_[nid].num_blocks;

        if (num_blocks == 0) return;

        int x = task_id - start_block * qlen;
        int token_id = x / num_blocks;  
        int offset = x % num_blocks; 

        int ith = start_block + offset;
        size_t n_stride = config_.stride; 

        int expert_id = token_expert_mapping [token_id][0].first;
        int pos = token_expert_mapping [token_id][0].second;
        int base = expert_reorder_offset[expert_id];
        float* down_output_ptr_0 = down_output_ + (base + pos)  * config_.hidden_size + ith * config_.stride;

        for(int j=0; j<n_stride; ++j){
            down_output_ptr_0[j] = down_output_ptr_0[j] * weights[token_id * k + 0];
        }
        for(int i=1; i<k; i++){
            expert_id = token_expert_mapping[token_id][i].first;
            pos = token_expert_mapping[token_id][i].second;
            base = expert_reorder_offset[expert_id]; 
            int expert_idx = token_id * k + i;
            float* down_output_ptr = down_output_  +  (base + pos) * config_.hidden_size + ith * config_.stride;
            for(int j=0; j<n_stride; ++j){
                    down_output_ptr_0[j] += down_output_ptr[j] * weights[expert_idx];
            }
        }
 
        void* output_ptr = (uint8_t*)output + (token_id * config_.hidden_size + ith * config_.stride) * hidden_type_size / hidden_blk_size;
        from_float(down_output_ptr_0, output_ptr, n_stride, config_.hidden_type);
    }, nullptr);
     

}



void MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, int* bsz_tensor) {
    int batch_size = bsz_tensor[0];
    int processed = 0;
    while (processed < batch_size) {
        int remaining = batch_size - processed;
        
        if (remaining < config_.group_min_len) { 
            for (int i = 0; i < remaining; i++) {
                int current_pos = processed + i;
                (this->*forward_one_impl)(
                    k, 
                    expert_ids + current_pos * k, 
                    weights + current_pos * k, 
                    (uint8_t*)input + current_pos * hidden_bytes, 
                    (uint8_t*)output + current_pos * hidden_bytes
                );
            }
            break;  
        }
         
        int forward_len = std::min(config_.group_max_len, remaining);
        (this->*forward_many_impl)(
            forward_len, 
            k, 
            expert_ids + processed * k, 
            weights + processed * k, 
            (uint8_t*)input + processed * hidden_bytes, 
            (uint8_t*)output + processed * hidden_bytes
        );
        
        processed += forward_len;
    }
    sync_flag.store(true, std::memory_order_seq_cst);
       
     
}


static void forward_wrapper(void* args) {
    ForwardParams* params = (ForwardParams*)args;
        params->moe_ptr->forward(
            params->qlen, 
            params->k, 
            params->expert_ids, 
            params->weights, 
            params->input, 
            params->output, 
            params->bsz_tensor
        );
        // delete params;  !don't delete params here
}

void MOE::submit_with_cuda_stream(intptr_t user_cuda_stream, int qlen, int k, const uint64_t* expert_ids, 
                                 const float* weights, const void* input, void* output, int* bsz_tensor) {
    sync_flag.store(false, std::memory_order_seq_cst); 
    ForwardParams* params = new ForwardParams();
    params->moe_ptr = this;
    params->qlen = qlen;
    params->k = k;
    params->expert_ids = expert_ids;
    params->weights = weights;
    params->input = input;
    params->output = output; 
    params->bsz_tensor = bsz_tensor;
     
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)forward_wrapper, params);
}

void MOE::sync() {
    while (!sync_flag.load(std::memory_order_seq_cst))
        ;
}
static void sync_(void * moe_ptr) { 
    MOE* moe = (MOE*)moe_ptr;
    moe->sync();
}

void MOE::sync_with_cuda_stream(intptr_t user_cuda_stream) { 
    cudaLaunchHostFunc((cudaStream_t)user_cuda_stream, (cudaHostFn_t)&sync_, (void*)this);
}

