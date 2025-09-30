#ifndef CPUINFER_BACKEND_NUMA_H
#define CPUINFER_BACKEND_NUMA_H
#ifndef USE_NUMA
#define USE_NUMA
#endif

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <vector> 
#include <sched.h>  
#include <unistd.h> 
#include <numa.h>
#include <numaif.h>
#include <assert.h>

#include <iostream>

enum ThreadStatus {
    WORKING,
    WAITING,
    EXIT,
};

struct ThreadState {
    std::unique_ptr<std::atomic<ThreadStatus>> status;
    std::unique_ptr<std::atomic<int>> curr;
    int end;
};

static int numa_nodes_ = 0; 
static bool numa_initialized = [](){
    if (numa_available() < 0) {
        std::cerr << "NUMA not available! abort.\n";
        std::abort();
        return false;
    } 
    numa_nodes_ = numa_num_configured_nodes();
    return true;
}();

static std::atomic<int> numa_counter(0); 
inline int get_next_numa_node() {
    return  (numa_counter.fetch_add(1, std::memory_order_relaxed)) % numa_nodes_;
}
  
constexpr size_t CACHE_LINE_SIZE = 64;
 

struct A_ThreadState {
    alignas(CACHE_LINE_SIZE) std::atomic<ThreadStatus> status;
    char padding1[CACHE_LINE_SIZE - sizeof(std::atomic<ThreadStatus>)];
    alignas(CACHE_LINE_SIZE) std::atomic<int> curr;
    char padding2[CACHE_LINE_SIZE - sizeof(std::atomic<int>)]; 
    int end; 
};

class Backend_NUMA {
public: 
    Backend_NUMA(const Backend_NUMA&) = delete;
    Backend_NUMA& operator=(const Backend_NUMA&) = delete;
 
    static Backend_NUMA& getInstance() {
        static Backend_NUMA instance;
        return instance;
    }
    void init_cpu_info();

    int get_num_threads();
     
    void do_k_work_stealing_job(int, int,
                                   std::function<void(int)>,
                                   std::function<void(int)>,
                                   std::function<void(int)>);
     

    void do_work(int, std::function<void(int)>,
                              std::function<void(int)>,
                              std::function<void(int)>);
    
    #ifdef USE_NUMA
    static thread_local int numa_node_;
    #endif
    static thread_local int thread_local_id_;

    bool power_saving_mode_; 

private:
    Backend_NUMA(int num_threads = 32);   
    ~Backend_NUMA();
    int max_threads_; 
    int num_cpus_;
    int num_cores_; 
    int cpus_per_node_;
    int hyper_threading_open_ = false;
    std::vector<int> cpu_to_thread_id_;

    struct CpuInfo {
        int cpuid_id;
        int core_id;
        int node_id;
        int package_id;
        int logic_idx;
    };

    struct ThreadInfo
    {
        int thread_id;
        int cpu_id;
        int core_id;
        int node_id;
        int package_id;
        int logic_idx;
    }; 
    std::vector<std::vector<int>> node_threads_;
    std::vector<CpuInfo> cpus_info_; 
    std::vector<ThreadInfo> threads_info_; 
    std::vector<A_ThreadState *> thread_state_; // [thread_num]
    std::function<void(int)> init_func_;
    std::function<void(int)> compute_func_;
    std::function<void(int)> finalize_func_;
    std::vector<std::thread> workers_; 
    void process_tasks(int);
    void worker_thread(int);
};
void bind_to_cpu(int cpu_id);
void bind_to_numa_node(int node_id);
void set_numa_mempolicy(int node_id);
void* allocate_aligned_numa(size_t size, int node);
void* allocate_aligned(size_t size);
void free_aligned_numa(void* aligned_ptr, size_t size);
void free_aligned(void* aligned_ptr, size_t size); 



#endif