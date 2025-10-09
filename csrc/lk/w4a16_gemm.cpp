#include "w4a16_gemm.hpp"

// BFloat16构造函数实现
BFloat16::BFloat16() : data(0) {}

BFloat16::BFloat16(float value) {
    // 将float转换为BFloat16表示
    // 这里是简化实现，实际应用中需要更精确的转换
    uint32_t f32 = *reinterpret_cast<uint32_t*>(&value);
    data = static_cast<uint16_t>(f32 >> 16);
}

// BFloat16转换运算符实现
BFloat16::operator float() const {
    // 将BFloat16转换回float
    uint32_t f32 = static_cast<uint32_t>(data) << 16;
    return *reinterpret_cast<float*>(&f32);
}

// quint4x2构造函数实现
quint4x2::quint4x2() = default;

quint4x2::quint4x2(uint8_t lo, uint8_t hi) {
    data[0] = lo & 0xF; // 低4位
    data[1] = hi & 0xF; // 高4位
}

#if defined(__AVX2__) && defined(__FMA__)

// 用于将int8转换为float32的实现
inline __m256 CVT_INT8_TO_FP32_AVX2(__m128i x) {
  // 先将int8扩展到int16
  __m256i epi16 = _mm256_cvtepi8_epi16(x);
  // 然后将int16扩展到int32
  // 注意：这里需要将__m256i转换为两个__m128i分别处理
  __m128i epi16_low = _mm256_castsi256_si128(epi16);
  __m128i epi16_high = _mm256_extracti128_si256(epi16, 1);
  
  // 将低8个int16扩展为int32
  __m256i epi32_low = _mm256_cvtepi16_epi32(epi16_low);
  // 将高8个int16扩展为int32
  __m256i epi32_high = _mm256_cvtepi16_epi32(epi16_high);
  
  // 将int32转换为float32
  __m256 f32_low = _mm256_cvtepi32_ps(epi32_low);
  __m256 f32_high = _mm256_cvtepi32_ps(epi32_high);
  
  // 合并结果
  return _mm256_permute2f128_ps(f32_low, f32_high, 0x20);
}
 
// AVX2版本的tinygemm_kernel_nn特化实现
template <bool has_bias>
void tinygemm_kernel_nn<BFloat16, has_bias, 4, 32>::apply(
    const BFloat16* __restrict__ A,
    const quint4x2* __restrict__ B,
    BFloat16* __restrict__ C,
    const uint8_t* __restrict__ Bz,
    const BFloat16* __restrict__ Bs,
    const float* __restrict__ bias,
    int64_t K,
    int group_size,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t strideBz,
    int64_t strideBs) {
  constexpr int ROWS = 4;
  constexpr int COLS = 2; // 32 / 16 = 2

  // 预取距离
  constexpr int PREFETCH_SIZE_K = 8 * 4;

  // 使用float32进行计算
  __m256 va[ROWS];
  __m256 vb[COLS];
  __m256 vc[ROWS * COLS];
  __m256 vc_master[ROWS * COLS];

  // 掩码和零值
  __m128i mask = _mm_set1_epi8(0xF);  // 低4位
  __m128i fifteen = _mm_set1_epi8(15);

  // 将查找表加载到向量寄存器
  __m256 bf16_lut_vec = _mm256_loadu_ps(bf16_lut_32);
  __m256 scales[COLS];
  __m128i zeros[COLS * 2];

  // 索引用于解包
  __m128i idx1 = _mm_set_epi8(
      15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);
  __m128i idx0 = _mm_set_epi8(
      7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);

  const int64_t K2 = K >> 1;
  const int64_t lda2 = lda >> 1;
  const int64_t ldb2 = ldb;
  const int64_t gs2 = group_size >> 1;
  
  // 将A转换为float32指针
  const float* a_ptr = reinterpret_cast<const float*>(A);
  
  // 加载偏置或初始化为零
  auto loadc = [&](auto i) {
    int col = i % COLS;
    if constexpr (has_bias) {
      vc_master[i] = _mm256_loadu_ps(bias + col * 16);
    } else {
      vc_master[i] = _mm256_set1_ps(0.f);
    }
  };
  // 手动展开循环，因为AVX2不支持Unroll宏的全部功能
  for (int i = 0; i < ROWS * COLS; ++i) {
    loadc(i);
  }

  // 主计算循环
  for (int64_t k = 0; k < K2; k += gs2) {
    // 预计算阶段
    for (int i = 0; i < ROWS * COLS; ++i) {
      const int row = i / COLS;
      const int col = i % COLS;
      vc[i] = _mm256_set1_ps(0.f);  // 重置累加器

      // 加载零点和缩放因子
      if (row == 0 && col % 2 == 0) {
        const int kgs = k / gs2;
        // 加载B的零点
        __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Bz + kgs * strideBz + col * 16));
        tmp = _mm_sub_epi8(tmp, fifteen);
        zeros[col] = _mm_shuffle_epi8(idx0, tmp);
        zeros[col + 1] = _mm_shuffle_epi8(idx1, tmp);

        // 加载B的缩放因子
        // 将BFloat16转换为float32
        float b_scale_f32[16];
        for (int j = 0; j < 16; j++) {
          b_scale_f32[j] = static_cast<float>(Bs[kgs * strideBs + col * 16 + j]);
        }
        scales[col] = _mm256_loadu_ps(b_scale_f32);
        scales[col + 1] = _mm256_loadu_ps(b_scale_f32 + 8);
      }
    }

    // 计算阶段
    for (int64_t k_offset = 0; k_offset < gs2; ++k_offset) {
      for (int i = 0; i < ROWS * COLS; ++i) {
        const int row = i / COLS;
        const int col = i % COLS;

        if (col == 0) {
          // 加载A的值到va寄存器
          va[row] = _mm256_set1_ps(a_ptr[row * lda2 + k + k_offset]);
        }
        
        if (row == 0 && col % 2 == 0) {
          // 加载并解包B的值
          __m128i vb_u4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(B + (k + k_offset) * ldb + col * 16));

          // 解包并减去零点
          __m128i vb_i8_lo = _mm_and_si128(vb_u4, mask);
          __m128i vb_i8_hi = _mm_and_si128(_mm_srli_epi16(vb_u4, 4), mask);
          vb_i8_lo = _mm_sub_epi8(vb_i8_lo, zeros[col]);
          vb_i8_hi = _mm_sub_epi8(vb_i8_hi, zeros[col + 1]);

          // 转换为float32并查找值
          // 这里使用简单的方法，因为AVX2没有直接的BFloat16支持
         float vb_f32_lo[8], vb_f32_hi[8]; 
          
          // 手动展开循环，使用编译时常量索引
          // j=0
          int8_t val_lo0 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_lo, 0));
          int8_t val_hi0 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_hi, 0));
          int idx_lo0 = val_lo0 + 15;
          int idx_hi0 = val_hi0 + 15;
          idx_lo0 = std::max(0, std::min(31, idx_lo0));
          idx_hi0 = std::max(0, std::min(31, idx_hi0));
          vb_f32_lo[0] = bf16_lut_32[idx_lo0];
          vb_f32_hi[0] = bf16_lut_32[idx_hi0];
          
          // j=1
          int8_t val_lo1 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_lo, 1));
          int8_t val_hi1 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_hi, 1));
          int idx_lo1 = val_lo1 + 15;
          int idx_hi1 = val_hi1 + 15;
          idx_lo1 = std::max(0, std::min(31, idx_lo1));
          idx_hi1 = std::max(0, std::min(31, idx_hi1));
          vb_f32_lo[1] = bf16_lut_32[idx_lo1];
          vb_f32_hi[1] = bf16_lut_32[idx_hi1];
          
          // j=2
          int8_t val_lo2 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_lo, 2));
          int8_t val_hi2 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_hi, 2));
          int idx_lo2 = val_lo2 + 15;
          int idx_hi2 = val_hi2 + 15;
          idx_lo2 = std::max(0, std::min(31, idx_lo2));
          idx_hi2 = std::max(0, std::min(31, idx_hi2));
          vb_f32_lo[2] = bf16_lut_32[idx_lo2];
          vb_f32_hi[2] = bf16_lut_32[idx_hi2];
          
          // j=3
          int8_t val_lo3 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_lo, 3));
          int8_t val_hi3 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_hi, 3));
          int idx_lo3 = val_lo3 + 15;
          int idx_hi3 = val_hi3 + 15;
          idx_lo3 = std::max(0, std::min(31, idx_lo3));
          idx_hi3 = std::max(0, std::min(31, idx_hi3));
          vb_f32_lo[3] = bf16_lut_32[idx_lo3];
          vb_f32_hi[3] = bf16_lut_32[idx_hi3];
          
          // j=4
          int8_t val_lo4 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_lo, 4));
          int8_t val_hi4 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_hi, 4));
          int idx_lo4 = val_lo4 + 15;
          int idx_hi4 = val_hi4 + 15;
          idx_lo4 = std::max(0, std::min(31, idx_lo4));
          idx_hi4 = std::max(0, std::min(31, idx_hi4));
          vb_f32_lo[4] = bf16_lut_32[idx_lo4];
          vb_f32_hi[4] = bf16_lut_32[idx_hi4];
          
          // j=5
          int8_t val_lo5 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_lo, 5));
          int8_t val_hi5 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_hi, 5));
          int idx_lo5 = val_lo5 + 15;
          int idx_hi5 = val_hi5 + 15;
          idx_lo5 = std::max(0, std::min(31, idx_lo5));
          idx_hi5 = std::max(0, std::min(31, idx_hi5));
          vb_f32_lo[5] = bf16_lut_32[idx_lo5];
          vb_f32_hi[5] = bf16_lut_32[idx_hi5];
          
          // j=6
          int8_t val_lo6 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_lo, 6));
          int8_t val_hi6 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_hi, 6));
          int idx_lo6 = val_lo6 + 15;
          int idx_hi6 = val_hi6 + 15;
          idx_lo6 = std::max(0, std::min(31, idx_lo6));
          idx_hi6 = std::max(0, std::min(31, idx_hi6));
          vb_f32_lo[6] = bf16_lut_32[idx_lo6];
          vb_f32_hi[6] = bf16_lut_32[idx_hi6];
          
          // j=7
          int8_t val_lo7 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_lo, 7));
          int8_t val_hi7 = static_cast<int8_t>(_mm_extract_epi8(vb_i8_hi, 7));
          int idx_lo7 = val_lo7 + 15;
          int idx_hi7 = val_hi7 + 15;
          idx_lo7 = std::max(0, std::min(31, idx_lo7));
          idx_hi7 = std::max(0, std::min(31, idx_hi7));
          vb_f32_lo[7] = bf16_lut_32[idx_lo7];
          vb_f32_hi[7] = bf16_lut_32[idx_hi7];
          
          // 加载到向量寄存器
          vb[col] = _mm256_loadu_ps(vb_f32_lo);
          vb[col + 1] = _mm256_loadu_ps(vb_f32_hi); 

          if (PREFETCH_SIZE_K > 0) {
            _mm_prefetch(reinterpret_cast<const char*>(B + (k + k_offset + PREFETCH_SIZE_K) * ldb2 + col * 16), _MM_HINT_T0);
          }
        }
        
        // 执行矩阵乘法计算 (va[row] * vb[col] 累加)
        vc[i] = _mm256_fmadd_ps(va[row], vb[col], vc[i]);
      }
    }

    // 后计算阶段 - 应用缩放因子
    for (int i = 0; i < ROWS * COLS; ++i) {
      const int col = i % COLS;
      vc_master[i] = _mm256_fmadd_ps(vc[i], scales[col], vc_master[i]);
    }
  }

  // 存储结果
  for (int i = 0; i < ROWS * COLS; ++i) {
    const int row = i / COLS;
    const int col = i % COLS;
    if (col % 2 == 0) {
      // 将结果从float32转换回BFloat16
      float result_f32[32];
      _mm256_storeu_ps(result_f32, vc_master[i]);
      _mm256_storeu_ps(result_f32 + 8, vc_master[i + 1]);
      
      // 转换并存储
      for (int j = 0; j < 16; j++) {
        C[row * ldc + col * 16 + j] = static_cast<BFloat16>(result_f32[j]);
      }
    }
  }
}

// 显式实例化模板，以便其他文件可以链接到这些实现
template struct tinygemm_kernel_nn<BFloat16, true, 4, 32>;
template struct tinygemm_kernel_nn<BFloat16, false, 4, 32>;

// AVX2版本的unpack_B函数
inline void unpack_B(
    BFloat16* __restrict__ Btmp,
    const quint4x2* __restrict__ packed_B,
    const uint8_t* __restrict__ Bz,
    const BFloat16* __restrict__ Bs,
    int64_t N,
    int64_t K,
    int group_size,
    int64_t ldb,
    int64_t ldb_tmp,
    int64_t strideBz,
    int64_t strideBs) {
  const int64_t K2 = K >> 1;
  const int64_t gs2 = group_size >> 1;
  const int64_t ldb2 = ldb;
  const int64_t ldb_tmp2 = ldb_tmp;
  
  // 使用float32进行中间计算
  float* btmp_ptr = reinterpret_cast<float*>(malloc(N * K * sizeof(float)));

  __m128i mask = _mm_set1_epi8(0xF);  // 低4位
  __m128i zeros[2];

  // 索引
  __m128i z_idx1 = _mm_set_epi8(
      15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);
  __m128i z_idx0 = _mm_set_epi8(
      7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);

  for (int n = 0; n < N; n += 16) {  // AVX2每次处理16个元素
    for (int k = 0; k < K2; ++k) {
      if (k % gs2 == 0) {
        const int kgs = k / gs2;

        // 加载B的零点
        __m128i tmp = _mm_loadu_si128(reinterpret_cast<const __m128i*>(Bz + kgs * strideBz + n));
        zeros[0] = _mm_shuffle_epi8(z_idx0, tmp);
        zeros[1] = _mm_shuffle_epi8(z_idx1, tmp);
      }

      // 加载并解包B
      __m128i vb_u4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed_B + k * ldb2 + n));

      // 解包并减去零点
      __m128i vb_i8_lo = _mm_and_si128(vb_u4, mask);
      __m128i vb_i8_hi = _mm_and_si128(_mm_srli_epi16(vb_u4, 4), mask);
      vb_i8_lo = _mm_sub_epi8(vb_i8_lo, zeros[0]);
      vb_i8_hi = _mm_sub_epi8(vb_i8_hi, zeros[1]);

      // 转换为float32并应用缩放因子
      // 由于AVX2限制，这里使用标量方法
      float* b_row_ptr = btmp_ptr + k * ldb_tmp2 + n;
      for (int j = 0; j < 8; j++) {
        int8_t val_lo = static_cast<int8_t>(_mm_extract_epi8(vb_i8_lo, j));
        int8_t val_hi = static_cast<int8_t>(_mm_extract_epi8(vb_i8_hi, j));
        
        // 加载缩放因子（从BFloat16转换为float32）
        const int kgs = k / gs2;
        float scale_lo = static_cast<float>(Bs[kgs * strideBs + n + j]);
        float scale_hi = static_cast<float>(Bs[kgs * strideBs + n + j + 8]);
        
        // 应用缩放
        b_row_ptr[j] = static_cast<float>(val_lo) * scale_lo;
        b_row_ptr[j + 8] = static_cast<float>(val_hi) * scale_hi;
      }
    }
  }

  // 将结果从float32转换回BFloat16
  for (int i = 0; i < N * K; i++) {
    Btmp[i] = static_cast<BFloat16>(btmp_ptr[i]);
  }
  
  free(btmp_ptr);
}

void w4a16_gemm(
    int M, 
    int N, 
    int K, 
    const uint8_t* __restrict__ B, 
    const float* __restrict__ A, 
    const float* __restrict__ scale, 
    float* __restrict__ C, 
    int ldc,
    int group_size) {
    // 转换为BFloat16格式的临时数组
    std::vector<BFloat16> A_bf16(M * K);
    std::vector<BFloat16> C_bf16(M * N);
    
    // 将float转换为BFloat16
    for (int i = 0; i < M * K; ++i) {
        A_bf16[i] = BFloat16(A[i]);
    }
    
    // 转换权重格式（从uint8到quint4x2）
    std::vector<quint4x2> B_quint4x2(K * N / 2);
    for (int i = 0; i < K * N; i += 2) {
        // 假设输入的uint8是低4位和高4位分别存储两个4位量化值
        uint8_t lo = B[i / 2] & 0xF;
        uint8_t hi = (B[i / 2] >> 4) & 0xF;
        B_quint4x2[i / 2] = quint4x2(lo, hi);
    }
    
    // 转换scale格式（从float到BFloat16）
    std::vector<BFloat16> scale_bf16(K * N / 2);
    for (int i = 0; i < K * N / 2; ++i) {
        scale_bf16[i] = BFloat16(scale[i]);
    }
    
    // 调用tinygemm_kernel_nn进行计算
    tinygemm_kernel_nn<BFloat16, false, 4, 32>::apply(
        A_bf16.data(),
        B_quint4x2.data(),
        C_bf16.data(),
        nullptr,  // 假设没有零点
        scale_bf16.data(),
        nullptr,  // 假设没有偏置
        K,        // 输入特征维度
        group_size,      // 假设组大小 
        K,        // lda
        N,        // ldb
        N,        // ldc
        0,        // 假设strideBz为0
        N         // 假设strideBs为N
    );
    
    // 将结果从BFloat16转换回float
    for (int i = 0; i < M * N; ++i) {
        C[i] = static_cast<float>(C_bf16[i]);
    }
}


#endif