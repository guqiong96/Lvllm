#ifndef W4A16_GEMM_HPP
#define W4A16_GEMM_HPP

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <immintrin.h>

#define __AVX2__ 1
#define __FMA__ 1

// 添加w4a16_gemm函数声明
void w4a16_gemm(
    int M, 
    int N, 
    int K, 
    const uint8_t* __restrict__ B, 
    const float* __restrict__ A, 
    const float* __restrict__ scale, 
    float* __restrict__ C, 
    int ldc,
    int group_size = 32);


// 自定义的BFloat16类型定义
struct BFloat16 {
    uint16_t data; // 16位存储
    
    // 构造函数
    BFloat16();
    BFloat16(float value);
    
    // 转换运算符
    operator float() const;
};

// 自定义的quint4x2类型定义（4位量化，每16字节存储32个值）
struct quint4x2 {
    uint8_t data[2]; // 存储两个4位量化值
    
    quint4x2();
    quint4x2(uint8_t lo, uint8_t hi);
};

// 用于向量操作的模板结构体
template <typename T, bool has_bias, int ROWS, int COLS>
struct tinygemm_kernel_nn;

#if defined(__AVX2__) && defined(__FMA__)

// 模拟BFloat16到Float32的转换
#define CVT_BF16_TO_FP32_AVX2(a) _mm256_cvtph_ps(_mm256_castsi256_ph(a))

// 用于将int8转换为float32
inline __m256 CVT_INT8_TO_FP32_AVX2(__m128i x);

// BFloat16常量查找表
constexpr float bf16_lut_32[32] = {
    0.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f,
    1.0f, 0.75f, 0.5f, 0.25f, 0.0f, -1.0f, -2.0f, -3.0f,
    0.0f, -4.0f, -8.0f, -16.0f, -32.0f, -64.0f, -128.0f, -256.0f,
    -512.0f, -1024.0f, -2048.0f, -4096.0f, -8192.0f, -16384.0f, -32768.0f, -65536.0f
};

// AVX2版本的tinygemm_kernel_nn特化实现
template <bool has_bias>
struct tinygemm_kernel_nn<BFloat16, has_bias, 4, 32> {
  static inline void apply(
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
      int64_t strideBs);
};

// 可以继续添加其他块大小的特化实现，如4x64, 2x32等

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
    int64_t strideBs);
#endif 

#endif // W4A16_GEMM_HPP