import torch
import time
import numpy as np
import ctypes

# 加载vllm._lk_C模块
try:
    import vllm._lk_C
    LK_MOE_AVAILABLE = True
    print("成功加载vllm._lk_C模块")
except ImportError:
    LK_MOE_AVAILABLE = False
    print("无法加载vllm._lk_C模块，测试无法运行")
    exit(1)

# 设置CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    print("此测试需要CUDA支持")
    exit(1)

# 定义BF16类型为30
BF16_TYPE = 30

class TestMOESubmitWithCUDAStream:
    def __init__(self):
        # 初始化参数
        self.num_experts = 8
        self.top_k = 2
        self.hidden_size = 512
        self.intermediate_size = 2048
        self.stride = 32
        self.group_min_len = 10
        self.group_max_len = 1024
        
        # 分配权重内存（使用CPU内存，因为LK::MOE需要CPU内存的权重）
        self._init_weights()
        
        # 创建MOE实例
        self.moe = self._create_moe_instance()
        
        # 初始化CUDA流
        self.stream = torch.cuda.Stream()
        self.stream_ptr = self._get_cuda_stream_ptr(self.stream)
        
        # 初始化用于图捕获的缓冲区
        self._init_graph_buffers()
        
        # 初始化CUDA图字典
        self.cuda_graphs = {}
        
    def _init_weights(self):
        """初始化权重参数"""
        # 为gate_proj和up_proj创建合并的权重张量
        # 在实际应用中，这些权重应该从模型加载
        w13_weight = torch.randn(
            (self.num_experts, 2 * self.intermediate_size, self.hidden_size),
            dtype=torch.bfloat16, device='cpu')
        
        # 分离gate_proj和up_proj
        self.gate_proj = w13_weight.narrow(1, 0, self.intermediate_size).contiguous()
        self.up_proj = w13_weight.narrow(1, self.intermediate_size, self.intermediate_size).contiguous()
        
        # 创建down_proj权重
        self.down_proj = torch.randn(
            (self.num_experts, self.hidden_size, self.intermediate_size),
            dtype=torch.bfloat16, device='cpu')
    
    def _create_moe_instance(self):
        """创建MOE实例"""
        # 创建MOEConfig
        moe_config = vllm._lk_C.MOEConfig(
            self.num_experts,        # expert_num
            self.top_k,              # routed_expert_num
            self.hidden_size,        # hidden_size
            self.intermediate_size,  # intermediate_size
            self.stride,             # stride
            self.group_min_len,      # group_min_len
            self.group_max_len,      # group_max_len
            self.gate_proj.data_ptr(),  # gate_proj
            self.up_proj.data_ptr(),    # up_proj
            self.down_proj.data_ptr(),  # down_proj
            BF16_TYPE,               # gate_type (BF16)
            BF16_TYPE,               # up_type (BF16)
            BF16_TYPE,               # down_type (BF16)
            BF16_TYPE                # hidden_type (BF16)
        )
        
        # 创建MOE实例
        return vllm._lk_C.MOE(moe_config)
    
    def _get_cuda_stream_ptr(self, stream):
        """获取CUDA流的指针"""
        # 使用Python的ctypes获取CUDA流指针
        if hasattr(torch.cuda, 'current_stream'):
            with stream:
                current_stream_ptr = torch.cuda.current_stream().cuda_stream
                return ctypes.cast(current_stream_ptr, ctypes.c_void_p).value
        else:
            # 回退方法
            return 0
    
    def _init_graph_buffers(self):
        """初始化用于CUDA图捕获的缓冲区"""
        # 准备不同批次大小的缓冲区（类似于vllm中的实现）
        self.graph_batch_sizes = []
        batch_size = 1
        while batch_size <= 4096:
            self.graph_batch_sizes.append(batch_size)
            batch_size *= 2
        
        # 创建CPU和GPU缓冲区
        self.input_tensor_cpu = []
        self.expert_ids_cpu = []
        self.weights_cpu = []
        self.output_cpu = []
        self.bsz_tensor_cpu = []
        self.output_gpu = []
        
        # 初始化CUDA输入缓冲区
        self.input_tensor_cuda = []
        self.expert_ids_cuda = []
        self.weights_cuda = []
        self.bsz_tensor_cuda = []
        
        for batch_size in self.graph_batch_sizes:
            # CPU缓冲区（使用pin_memory提高主机到设备的传输速度）
            self.input_tensor_cpu.append(torch.zeros(
                (batch_size, self.hidden_size), device='cpu', dtype=torch.bfloat16, pin_memory=True))
            self.expert_ids_cpu.append(torch.zeros(
                (batch_size, self.top_k), device='cpu', dtype=torch.long, pin_memory=True))
            self.weights_cpu.append(torch.zeros(
                (batch_size, self.top_k), device='cpu', dtype=torch.float32, pin_memory=True))
            self.output_cpu.append(torch.zeros(
                (batch_size, self.hidden_size), device='cpu', dtype=torch.bfloat16, pin_memory=True))
            self.bsz_tensor_cpu.append(torch.zeros(
                (1), device='cpu', dtype=torch.int32, pin_memory=True))
            
            # GPU缓冲区
            self.output_gpu.append(torch.zeros(
                (batch_size, self.hidden_size), device='cuda', dtype=torch.bfloat16))
            
            # CUDA输入缓冲区
            self.input_tensor_cuda.append(torch.zeros(
                (batch_size, self.hidden_size), device='cuda', dtype=torch.bfloat16))
            self.expert_ids_cuda.append(torch.zeros(
                (batch_size, self.top_k), device='cuda', dtype=torch.long))
            self.weights_cuda.append(torch.zeros(
                (batch_size, self.top_k), device='cuda', dtype=torch.float32))
            self.bsz_tensor_cuda.append(torch.zeros(
                (1), device='cuda', dtype=torch.int32))
    
    def _find_best_graph_index(self, total_tokens):
        """找到最适合给定token数量的图索引"""
        low, high = 0, len(self.graph_batch_sizes) - 1
        best_index = 0
        
        while low <= high:
            mid = (low + high) // 2
            if self.graph_batch_sizes[mid] >= total_tokens:
                best_index = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return best_index
    
    def run_warmup(self, batch_size, num_runs=5):
        """运行warmup阶段"""
        print(f"\n运行warmup阶段 (batch_size={batch_size}, runs={num_runs})")
        
        # 生成随机输入数据
        input_data = torch.randn(
            (batch_size, self.hidden_size), dtype=torch.bfloat16, device='cpu')
        expert_ids = torch.randint(0, self.num_experts, 
            (batch_size, self.top_k), dtype=torch.long, device='cpu')
        weights = torch.randn(
            (batch_size, self.top_k), dtype=torch.float32, device='cpu')
        bsz_tensor = torch.tensor([batch_size], device='cpu', dtype=torch.int32)
        
        # 执行多次warmup运行
        start_time = time.time()
        for i in range(num_runs):
            # 复制数据到预分配的缓冲区
            with self.stream:
                # 提交到CUDA流
                self.moe.submit_with_cuda_stream(
                    self.stream_ptr,
                    batch_size,           # qlen
                    self.top_k,           # k
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    self.output_cpu[0].data_ptr(),  # 使用第一个缓冲区进行warmup
                    bsz_tensor.data_ptr()
                )
                # 同步流
                self.moe.sync_with_cuda_stream(self.stream_ptr)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        print(f"Warmup完成，平均时间: {(end_time - start_time) / num_runs * 1000:.2f} ms/run")
    
    def run_graph_capture_and_replay(self, batch_size, capture_runs=1, replay_runs=10):
        """运行CUDA图捕获和回放阶段"""
        print(f"\n运行CUDA图捕获和回放阶段 (batch_size={batch_size})")
        
        # 找到最适合的图索引
        graph_index = self._find_best_graph_index(batch_size)
        graph_batch_size = self.graph_batch_sizes[graph_index]
        print(f"使用图索引: {graph_index}, 图批次大小: {graph_batch_size}")
        
        # 在CUDA上生成随机输入数据
        input_data = torch.randn(
            (batch_size, self.hidden_size), dtype=torch.bfloat16, device='cuda')
        expert_ids = torch.randint(0, self.num_experts, 
            (batch_size, self.top_k), dtype=torch.long, device='cuda')
        weights = torch.randn(
            (batch_size, self.top_k), dtype=torch.float32, device='cuda')
        bsz_tensor = torch.tensor([batch_size], device='cuda', dtype=torch.int32)
        
        # 检查是否已经为该批次大小捕获了图
        graph_key = f"{batch_size}_{self.top_k}"
        if graph_key not in self.cuda_graphs:
            # 捕获阶段
            print("开始CUDA图捕获...")
            start_time = time.time()
            
            # 创建CUDA图对象
            cuda_graph = torch.cuda.CUDAGraph()
            
            # 设置捕获的流
            with torch.cuda.stream(self.stream):
                # 开始捕获
                with torch.cuda.graph(cuda_graph, stream=self.stream):
                    # 将CUDA上的输入数据复制到CUDA缓冲区（在捕获范围内）
                    self.input_tensor_cuda[graph_index][:batch_size].copy_(input_data, non_blocking=True)
                    self.expert_ids_cuda[graph_index][:batch_size].copy_(expert_ids, non_blocking=True)
                    self.weights_cuda[graph_index][:batch_size].copy_(weights, non_blocking=True)
                    self.bsz_tensor_cuda[graph_index].copy_(bsz_tensor, non_blocking=True)
                    
                    # 从CUDA缓冲区复制到CPU缓冲区（在捕获范围内）
                    self.input_tensor_cpu[graph_index][:batch_size].copy_(self.input_tensor_cuda[graph_index][:batch_size], non_blocking=False)
                    self.expert_ids_cpu[graph_index][:batch_size].copy_(self.expert_ids_cuda[graph_index][:batch_size], non_blocking=False)
                    self.weights_cpu[graph_index][:batch_size].copy_(self.weights_cuda[graph_index][:batch_size], non_blocking=False)
                    self.bsz_tensor_cpu[graph_index].copy_(self.bsz_tensor_cuda[graph_index], non_blocking=False)
                    
                    # 提交到CUDA流（这将被捕获）
                    self.moe.submit_with_cuda_stream(
                        self.stream_ptr,
                        batch_size,                           # qlen
                        self.top_k,                           # k
                        self.expert_ids_cpu[graph_index].data_ptr(),
                        self.weights_cpu[graph_index].data_ptr(),
                        self.input_tensor_cpu[graph_index].data_ptr(),
                        self.output_cpu[graph_index].data_ptr(),
                        self.bsz_tensor_cpu[graph_index].data_ptr()
                    )
                    self.moe.sync_with_cuda_stream(self.stream_ptr)
                    
                    # 将结果复制回GPU
                    self.output_gpu[graph_index][:batch_size].copy_(self.output_cpu[graph_index][:batch_size], non_blocking=True)
            
            torch.cuda.synchronize()
            capture_time = time.time() - start_time
            print(f"CUDA图捕获完成，时间: {capture_time * 1000:.2f} ms")
            
            # 存储捕获的图
            self.cuda_graphs[graph_key] = cuda_graph
        else:
            print("使用已捕获的CUDA图")
        
        # 回放阶段
        print("开始CUDA图回放测试...")
        start_time = time.time()
        
        with torch.cuda.stream(self.stream):
            for _ in range(replay_runs):
                # 更新CUDA输入数据（在回放循环中更新）
                input_data = torch.randn_like(input_data)
                expert_ids = torch.randint(0, self.num_experts, size=expert_ids.size(), dtype=expert_ids.dtype, device=expert_ids.device)
                weights = torch.randn_like(weights)
                
                # 复制新数据到CUDA缓冲区
                self.input_tensor_cuda[graph_index][:batch_size].copy_(input_data, non_blocking=True)
                self.expert_ids_cuda[graph_index][:batch_size].copy_(expert_ids, non_blocking=True)
                self.weights_cuda[graph_index][:batch_size].copy_(weights, non_blocking=True)
                
                # 回放捕获的图
                self.cuda_graphs[graph_key].replay()
        
        torch.cuda.synchronize()
        replay_time = time.time() - start_time
        
        print(f"CUDA图回放完成，总时间: {replay_time * 1000:.2f} ms")
        print(f"平均回放时间: {(replay_time / replay_runs) * 1000:.2f} ms/run")
        
        # 验证结果
        self._verify_results(graph_index, batch_size)
        
    def _verify_results(self, graph_index, batch_size):
        """验证结果的有效性"""
        # 检查输出是否包含非零值（简单验证）
        output_mean = self.output_gpu[graph_index][:batch_size].mean().item()
        output_std = self.output_gpu[graph_index][:batch_size].std().item()
        
        print(f"输出验证: 均值={output_mean:.6f}, 标准差={output_std:.6f}")
        
        # 如果均值和标准差都为0，说明可能有问题
        if abs(output_mean) < 1e-6 and output_std < 1e-6:
            print("警告: 输出可能不正确（均值和标准差都接近0）")
        else:
            print("输出验证通过")
    
    def run_test(self):
        """运行完整测试"""
        print("开始MOE submit_with_cuda_stream测试")
        print(f"参数: num_experts={self.num_experts}, top_k={self.top_k}, hidden_size={self.hidden_size}")
        
        # 测试不同的批次大小
        batch_sizes = [16, 64, 256]
        
        for batch_size in batch_sizes:
            # 运行warmup
            self.run_warmup(batch_size)
            
            # 运行图捕获和回放测试
            self.run_graph_capture_and_replay(batch_size)
        
        print("\n测试完成！")

# 运行测试
if __name__ == "__main__":
    if LK_MOE_AVAILABLE:
        test = TestMOESubmitWithCUDAStream()
        test.run_test()
    else:
        print("无法运行测试，vllm._lk_C模块不可用")