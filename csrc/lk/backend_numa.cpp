/**
 * @Description  : 
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : guqiong96
 * @LastEditTime : 2025-08-12 10:33:34
 * @LastEditors  : guqiong96
 * @LastEditTime : 2025-08-12 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#include "backend_numa.h"
#include <fstream>
#include <unordered_set>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <set>
#include <algorithm> 
// #define __AMX_INT8__ 1
// #define __AVX512VNNI__ 1
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    #include "../operators/llamafile/amx_gemm.hpp"
#endif


thread_local int Backend_NUMA::numa_node_ = -1;
thread_local int Backend_NUMA::thread_local_id_ = -1;


int read_topology(const std::string& cpu_path, const char* file) {
    std::ifstream ifs(cpu_path + "/" + file);
    int value;
    return ifs >> value ? value : -1;
}
 

void Backend_NUMA::init_cpu_info() { 
    
    num_cpus_ = numa_num_configured_cpus();
    cpus_info_.clear();
    cpus_info_.reserve(num_cpus_);

    std::map<std::pair<int, int>, int> unique_cores; // (package_id, raw_core_id) -> continuous_id
    int next_phys_id = 0; 
    std::map<std::pair<int, int>, int> core_counters; // (package_id, raw_core_id) -> current logic_idx

    for (int i = 0; i < num_cpus_; ++i) {
        std::string cpu_dir = "/sys/devices/system/cpu/cpu" + std::to_string(i);
        int cid = read_topology(cpu_dir, "topology/core_id");
        int pid = read_topology(cpu_dir, "topology/physical_package_id");


        std::pair<int, int> core_key = std::make_pair(pid, cid);
   
        if (unique_cores.find(core_key) == unique_cores.end()) {
            unique_cores[core_key] = next_phys_id++;
        }
        
        core_counters[core_key] = 0;
    }

    num_cores_ = next_phys_id; 
 
    for (int i = 0; i < num_cpus_; ++i) {
        std::string cpu_dir = "/sys/devices/system/cpu/cpu" + std::to_string(i);
        
        int raw_cid = read_topology(cpu_dir, "topology/core_id");
        int pid = read_topology(cpu_dir, "topology/physical_package_id");
         
        int nid = numa_node_of_cpu(i);
   
        std::pair<int, int> core_key = std::make_pair(pid, raw_cid);
        
        CpuInfo info = {
            .cpuid_id = i,         
            .core_id = unique_cores[core_key],   
            .node_id = nid,                      
            .package_id = pid,               
            .logic_idx = core_counters[core_key]
        };

        core_counters[core_key]++;   
        
        cpus_info_.push_back(info);
    }

    
    cpus_per_node_ = num_cpus_ / numa_nodes_;
    hyper_threading_open_ = num_cpus_ > num_cores_;
     
}

Backend_NUMA::Backend_NUMA(int num_threads) {
    
    init_cpu_info();

    const char* env_power_saving = std::getenv("LK_POWER_SAVING");
     
    power_saving_mode_ = false;
    if (env_power_saving != nullptr) { 
        std::string val(env_power_saving);
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);
        
        if (val == "1" || val == "true" || val == "yes" || val == "on") {
            power_saving_mode_ = true;
            std::cout << "Using LK_POWER_SAVING from environment: " << val << std::endl;
        }
    }

    const char* env_threads = std::getenv("LK_THREADS");
    if (env_threads != nullptr) { 
        bool is_valid = true;
        for (const char* p = env_threads; *p != '\0'; ++p) {
            if (!std::isdigit(static_cast<unsigned char>(*p))) {
                is_valid = false;
                break;
            }
        }
        
        if (is_valid) {
            num_threads= std::atoi(env_threads);
            std::cout << "Using LK_THREADS from environment: " 
                      << num_threads << std::endl;
        }else{
            num_threads = num_cpus_ / 2;
        }

    }else{
        num_threads = num_cpus_ / 2;
    }
 
    max_threads_ = num_threads < numa_nodes_ ? numa_nodes_ : num_threads;  
    max_threads_ = max_threads_ > num_cpus_ - 2 ? num_cpus_ - 2 : max_threads_;

    node_threads_.resize(numa_nodes_);
    threads_info_.resize(max_threads_);
    cpu_to_thread_id_.resize(num_cpus_, -1);
    

    int base = max_threads_ / numa_nodes_;
    int remain = max_threads_ % numa_nodes_;
    int tid = 0; 
    for (int nid = 0; nid < numa_nodes_; ++nid) {
        int n = base + (nid < remain);
        node_threads_[nid].clear();
        node_threads_[nid].reserve(n); 
        int n_find = 0;
        for(int cid = 0; cid < num_cpus_; ++cid){
            if(n_find == n) break; 
            if(cpus_info_[cid].node_id == nid){
                auto& this_cpu = cpus_info_[cid];
                threads_info_[tid] = {
                    .thread_id = tid,                    
                    .cpu_id = cid,
                    .core_id = this_cpu.core_id,
                    .node_id = this_cpu.node_id,
                    .package_id = this_cpu.package_id,
                    .logic_idx = this_cpu.logic_idx
                };
                node_threads_[nid].push_back(tid);
                cpu_to_thread_id_[cid] = tid;
                

                auto& this_thread = threads_info_[tid];
                if(numa_node_of_cpu(cid) != nid){
                    std::cout << "numa_node_of_cpu(cid) != nid   cid:" << cid << " nid:" << nid;
                }
                std::cout << "Backend_NUMA init, thread_id: " << tid << " cpu_id:" << this_thread.cpu_id << " core_id:" << this_thread.core_id << "[" << this_thread.logic_idx <<"]"  << " node_id:" << this_thread.node_id<< std::endl;
                
                tid++;
                n_find++;
            }
        }
    }
     
    thread_state_.resize(max_threads_);
    for (int i = 0; i < max_threads_; i++) {
        A_ThreadState* state = new (std::align_val_t{64}) A_ThreadState(); 
        state->status.store(ThreadStatus::WAITING, std::memory_order_relaxed);
        state->curr.store(0, std::memory_order_relaxed);
        state->end = 0;
        thread_state_[i] = state;
    }
    std::atomic_thread_fence(std::memory_order_release);

    workers_.resize(max_threads_);
    for (int i = 0; i < max_threads_; i++) {
        workers_[i] = std::thread(&Backend_NUMA::worker_thread, this, i);
    }

    std::cout << "Backend_NUMA init suscess, numa nodes: " << numa_nodes_ << " cors:" << num_cores_ << " cpus:" << num_cpus_ << " max_threads: " << max_threads_ << std::endl;
}

Backend_NUMA::~Backend_NUMA() {
    for (int i = 0; i < max_threads_; i++) {
        thread_state_[i]->status.store(ThreadStatus::EXIT,
                                       std::memory_order_release);
    }
    for (int i = 0; i < max_threads_; i++) {
        if (workers_[i].joinable()) {
            workers_[i].join();
        }
    }

    for (A_ThreadState* state : thread_state_) {
        state->~A_ThreadState();
         
        #if __cpp_aligned_new >= 201606  // C++17  
            operator delete(state, std::align_val_t{64});
        #else
            AlignedFree(state);   
        #endif
    }
    thread_state_.clear();
}

int Backend_NUMA::get_num_threads(){
    return max_threads_;
}

void Backend_NUMA::do_work(int nth, std::function<void(int)> init_func,
                                   std::function<void(int)> compute_func,
                                   std::function<void(int)> finalize_func) {
 
    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func;
      
    int base = nth / max_threads_;
    int remain = nth % max_threads_;
  
    int begin = 0;
    int end = 0;     
    for (int i = 0; i < max_threads_; i++) {
        begin = end;
        end = begin + (base + (i < remain));    
        if (begin >= end) { 
            thread_state_[i]->status.store(ThreadStatus::WAITING, std::memory_order_release);
            thread_state_[i]->end = -1; 
            thread_state_[i]->curr.store(0, std::memory_order_relaxed);
        } else {
            thread_state_[i]->curr.store(begin, std::memory_order_relaxed);
            thread_state_[i]->end = end;
            thread_state_[i]->status.store(ThreadStatus::WORKING, std::memory_order_release);
        }
    } 

    for (int i = 0; i < max_threads_; i++) {
        while (thread_state_[i]->status.load(std::memory_order_acquire) == ThreadStatus::WORKING) {
            if(power_saving_mode_) std::this_thread::yield();
        }
    }
}

void Backend_NUMA::do_k_work_stealing_job(int k, int nth,
                                   std::function<void(int)> init_func,
                                   std::function<void(int)> compute_func,
                                   std::function<void(int)> finalize_func) {
    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func;
                                    
    int base = nth / numa_nodes_;
    int remain = nth % numa_nodes_;
     
    int begin = 0;
    int end =0;
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int n_tasks = (base + (nid < remain)) * k;  
        int n_threads = node_threads_[nid].size();
        if (n_threads == 0){
            throw std::runtime_error("Backend_NUMA::do_k_work_stealing_job: n_threads == 0");
        }
        if(n_tasks == 0) {
            continue;
        }
          
        int t_base= n_tasks / n_threads;
        int t_remain = n_tasks % n_threads;
        
        for (int j = 0; j < n_threads; j++) { 
            int tid = node_threads_[nid][j];
            begin = end;
            end = begin + t_base + (j < t_remain);
            if (begin >= end) { 
                thread_state_[tid]->status.store(ThreadStatus::WAITING, std::memory_order_release);
                thread_state_[tid]->end = -1; 
                thread_state_[tid]->curr.store(0, std::memory_order_relaxed);
            } else {
                thread_state_[tid]->curr.store(begin, std::memory_order_relaxed);
                thread_state_[tid]->end = end;
                thread_state_[tid]->status.store(ThreadStatus::WORKING, std::memory_order_release);
            }
        }
    }

    for (int i = 0; i < max_threads_; i++) { 
        while (thread_state_[i]->status.load(std::memory_order_acquire) == ThreadStatus::WORKING) {
            if(power_saving_mode_) std::this_thread::yield();
        }
    }
}

void Backend_NUMA::process_tasks(int thread_id) {

     
    if (init_func_ != nullptr) {
        init_func_(thread_id);
    }
    while (true) {
        if (thread_state_[thread_id]->status.load(std::memory_order_acquire) !=
            ThreadStatus::WORKING) {
            break;
        } 
        int task_id = thread_state_[thread_id]->curr.fetch_add(
            1, std::memory_order_acq_rel);
        if (task_id >= thread_state_[thread_id]->end) {
            break;
        }
        compute_func_(task_id);
    } 
    for(int i=0; i<max_threads_; i++){
        auto& this_ = threads_info_[thread_id];
        auto& other_ = threads_info_[i];
        if(this_.node_id == other_.node_id && this_.cpu_id != other_.cpu_id){
            if (thread_state_[i]->status.load(std::memory_order_acquire) != ThreadStatus::WORKING) {
                continue;
            } 
            while (true) {
                int task_id = thread_state_[i]->curr.fetch_add(
                    1, std::memory_order_acq_rel);
                if (task_id >= thread_state_[i]->end) {
                    break;
                }
                compute_func_(task_id);
            } 

        }
    }
 

    if (finalize_func_ != nullptr) {
        finalize_func_(thread_id);
    }
    thread_state_[thread_id]->status.store(ThreadStatus::WAITING,
                                           std::memory_order_release);
}

void Backend_NUMA::worker_thread(int thread_id) { 
    auto start = std::chrono::steady_clock::now(); 
 
    thread_local_id_ = thread_id;
    numa_node_ = threads_info_[thread_id].node_id; 
    int cpu_id = threads_info_[thread_id].cpu_id;

    assert(numa_node_ == numa_node_of_cpu(cpu_id));
    
    bind_to_cpu(cpu_id); 
    set_numa_mempolicy(numa_node_);

    #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
        ggml_tile_config_init();
    #endif
    
    while (true) {
        ThreadStatus status =
            thread_state_[thread_id]->status.load(std::memory_order_acquire);
        if (status == ThreadStatus::WORKING) {
            process_tasks(thread_id);
            start = std::chrono::steady_clock::now();
        } else if (status == ThreadStatus::WAITING) {
            auto now = std::chrono::steady_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                      start)
                    .count();
            if (duration > 50) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if(power_saving_mode_) std::this_thread::yield();
        } else if (status == ThreadStatus::EXIT) {
            return;
        }
        
    }
} 


void bind_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
        perror("sched_setaffinity failed");
        exit(EXIT_FAILURE);
    }
}

void bind_to_numa_node(int node_id) { 
    struct bitmask *node_cpus = numa_allocate_cpumask();
    if (numa_node_to_cpus(node_id, node_cpus) != 0) {
        perror("Failed to get NUMA node CPUs");
        numa_free_cpumask(node_cpus);
        std::abort();
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
 
    for (unsigned int i = 0; i < node_cpus->size; ++i) {
        if (numa_bitmask_isbitset(node_cpus, i)) {
            CPU_SET(i, &cpuset);
        }
    }
    numa_free_cpumask(node_cpus);
 
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        perror("sched_setaffinity failed"); 
    }
}
 
void set_numa_mempolicy(int node_id) {
    struct bitmask* mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, node_id);
 
    int policy = MPOL_BIND;

    if (set_mempolicy(policy, mask->maskp, mask->size) == -1) {
        std::cerr << "set_mempolicy failed for node " << node_id 
                  << ": " << errno << " (" << strerror(errno) << ")\n";
        std::abort();
    }
    numa_free_nodemask(mask);
}

 
void* allocate_aligned_numa(size_t size, int node) { 
    size_t alignment = 64;
    size_t total_size = size + alignment - 1;
    void* raw_ptr = numa_alloc_onnode(total_size, node);
    if (!raw_ptr) {
        std::cerr << "Failed to allocate " << size << " bytes on NUMA node " << node << std::endl;
        return nullptr;
    }
     
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned_addr);
}

void free_aligned_numa(void* aligned_ptr, size_t size) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(aligned_ptr);
    void* raw_ptr = reinterpret_cast<void*>(addr & ~(63));
    numa_free(raw_ptr, size);
}

void* allocate_aligned(size_t size) {
    const size_t alignment = 64; 
    size_t total_size = size + alignment + sizeof(void*);
    void* raw_ptr = malloc(total_size);
    if (!raw_ptr) return nullptr;
     
    uintptr_t aligned_addr = (reinterpret_cast<uintptr_t>(raw_ptr) + sizeof(void*) + alignment - 1) & ~(alignment - 1);
    void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
 
    void** prev_ptr = reinterpret_cast<void**>(aligned_ptr) - 1;
    *prev_ptr = raw_ptr;

    return aligned_ptr;
} 

void free_aligned(void* aligned_ptr, size_t size) {
    if (!aligned_ptr) return; 
    void** prev_ptr = reinterpret_cast<void**>(aligned_ptr) - 1;
    void* raw_ptr = *prev_ptr;
    free(raw_ptr);
}

