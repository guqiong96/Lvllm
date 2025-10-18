#include <stdio.h>
#include "bestla_wrapper.h"
#include "bestla_ut.h"
#include "sycl_ut.h"
#include "sycl/sycl_wrapper.h"

namespace bestla {
using namespace ut;
using namespace utils;
using namespace sycl_utils;
using namespace sycl_gemm;
namespace sycl_ut {
int constexpr TestMs = 1000;
class Benchmark_Fp32Fp32 {
 public:
  Benchmark_Fp32Fp32() {
    UT_START();
    benchmark_all(1024, 4096, 4096);
    benchmark_all(4096, 4096, 4096);
  }

  using AType = float;
  using BType = float;
  using CType = float;
  using SGemmT = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemmT>;

  template <typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems) {
    LOG_T log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = A;
    auto B_d = B;
    auto C_d = C;
    auto psize = (size_t)m * n * k * 2;
    sycl::range<2> group{SGemmT::WgM, SGemmT::WgN};
    sycl::range<2> problem{static_cast<size_t>(m) / SGemmT::TileM, static_cast<size_t>(n) / SGemmT::TileN};
    utils::GemmProblem gp(1, m, n, k);
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        auto ev = KernelLauncher::compute<false>(q, m, n, k, {{A, k}, {B, n}, {C, n}});
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f\n", log.get_log_str(), flops);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, BTLA_DTYPE::F32, BTLA_DTYPE::F32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32),
           bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(BTLA_DTYPE::F32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch, 0);
    fill_buffer_randn(A.data(), m * k, -0.5f, 0.5f);
    fill_buffer_randn(B.data(), n * k, -0.5f, 0.5f);
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    sycl_vector<float> dA(A.size(), q), dB(B.size(), q), dC(C.size(), q);
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    q->memcpy(dB.data(), B.data(), B.size() * 4).wait();

    benchmark<LOG>(m, n, k, batch, dA.data(), dB.data(), dC.data(), testtime);
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_Fp32Fp32 sBenchmark_Fp32Fp32;
#endif

class Benchmark_Fp16Fp16 {
 public:
  Benchmark_Fp16Fp16() {
    UT_START();
    benchmark_all(1024, 4096, 4096);
    benchmark_all(4096, 4096, 4096);
    benchmark_all(4096, 4096 * 3, 4096);
  }

  using AType = sycl::half;
  using BType = sycl::half;
  using CType = sycl::half;
  using SGemmT = xve::DefaultHGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, sycl::half>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, sycl::half>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, sycl::half>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemmT>;

  template <typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems) {
    LOG_T log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = A;
    auto B_d = B;
    auto C_d = C;
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        auto ev = KernelLauncher::compute<false>(q, m, n, k, {{A, k}, {B, n}, {C, n}});
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f\n", log.get_log_str(), flops);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, BTLA_DTYPE::F32, BTLA_DTYPE::F32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32),
           bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(BTLA_DTYPE::F32));
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    sycl_vector<AType> dA(size_t(m) * k * batch, q);
    sycl_vector<BType> dB(size_t(k) * n * batch, q);
    sycl_vector<CType> dC(size_t(m) * n * batch, q);

    benchmark<LOG>(m, n, k, batch, dA.data(), dB.data(), dC.data(), testtime);
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_Fp16Fp16 sBenchmark_Fp16Fp16;
#endif

class Benchmark_S4Fp32Fp32 {
 public:
  Benchmark_S4Fp32Fp32() {
    UT_START();
    benchmark_all(1, 4096, 4096);
    benchmark_all(1, 4096, 11008);
    benchmark_all(1, 4096, 4096 * 3);
    benchmark_all(1, 4096 * 3, 4096);
    benchmark_all(1024, 4096, 4096);
    benchmark_all(4096, 4096, 4096);
  }

  using AType = float;
  using BType = float;
  using CType = float;
  using SGemmT = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightS4<GCT, float>;
  template <class GCT>
  using ProBTransT = sycl_prologue_b::WeightS4Trans<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBT, EpiT, SGemmT>;
  using KernelLauncherT = sycl_wrapper::LauncherWOQ<ProAT, ProBTransT, EpiT, SGemmT>;

  template <typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, uint8_t* B, float* B_scale, CType* C, float timems) {
    LOG_T log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = A;
    auto B_d = B;
    auto C_d = C;
    auto psize = (size_t)m * n * k * 2;
    sycl::range<2> group{SGemmT::WgM, SGemmT::WgN};
    sycl::range<2> problem{static_cast<size_t>(m) / SGemmT::TileM, static_cast<size_t>(n) / SGemmT::TileN};
    utils::GemmProblem gp(1, m, n, k);
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        auto ev = KernelLauncher::compute(q, m, n, k, 128, {{A, k}, {B, B_scale, n}, {C, n}});
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f\n", log.get_log_str(), flops);
  }

  template <typename LOG_T>
  void benchmarkT(int m, int n, int k, int batch, AType* A, uint8_t* B, float* B_scale, CType* C, float timems) {
    LOG_T log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = A;
    auto B_d = B;
    auto C_d = C;
    auto psize = (size_t)m * n * k * 2;
    int blks = k / 32;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        auto ev = KernelLauncherT::compute(q, m, n, k, 128, {{A, k}, {B, B_scale, blks}, {C, n}});
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f\n", log.get_log_str(), flops);
  }

  template <typename LOG_T>
  void benchmark_gemv_T2(int m, int n, int k, int batch, AType* A, uint8_t* B, float* B_scale, CType* C, float timems) {
    LOG_T log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = (const AType*)A;
    auto B_d = B;
    auto C_d = C;
    auto S_d = B_scale;
    auto psize = (size_t)m * n * k * 2;
    int blocksize = 128;
    int blks = k / blocksize;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        int constexpr SgSize = 16;
        int constexpr TileK = 32;
        int constexpr GroupK = SgSize * TileK;
        sycl::range<1> group{SgSize};
        sycl::range<1> problem{static_cast<size_t>(n) * SgSize};
        auto ev = ProBTransT<SGemmT>::gemv(A_d, {B_d, S_d, blks}, C_d, n, k, blocksize, q);
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f\n", log.get_log_str(), flops);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, BTLA_DTYPE::F32, BTLA_DTYPE::F32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32),
           bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(BTLA_DTYPE::F32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<float> B_scale(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch, 0);
    fill_buffer_randn(A.data(), m * k, -0.5f, 0.5f);
    fill_buffer_randn(B.data(), n * k, -0.5f, 0.5f);
    fill_buffer_randn(B_scale.data(), n * k, -0.5f, 0.5f);
    avector<uint8_t> B_s8(k * n * batch / 2);
    fill_buffer_randn(B_s8.data(), B_s8.size(), uint8_t(0), uint8_t(255));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
      memcpy(B_s8.data() + i * n * k / 2, B_s8.data(), n * k * sizeof(uint8_t) / 2);
      memcpy(B_scale.data() + i * n * k, B_scale.data(), n * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    sycl_vector<float> dA(A.size(), q), dB(B.size(), q), dC(C.size(), q), dB_scale(B_scale.size(), q);
    sycl_vector<uint8_t> dBs8(B_s8.size(), q);
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    q->memcpy(dB.data(), B.data(), B.size() * 4).wait();
    if (m == 1) {
      benchmark_gemv_T2<LOG>(m, n, k, batch, dA.data(), dBs8.data(), dB_scale.data(), dC.data(), testtime);
    } else {
      benchmarkT<LOG>(m, n, k, batch, dA.data(), dBs8.data(), dB_scale.data(), dC.data(), testtime);
    }
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_S4Fp32Fp32 sBenchmark_S4Fp32Fp32;
#endif

class Benchmark_S4Fp16Fp16 {
 public:
  Benchmark_S4Fp16Fp16() {
    UT_START();
    benchmark_all(1, 4096, 4096, 128);
    benchmark_all(1, 4096, 11008, 128);
    benchmark_all(1, 4096, 4096 * 4, 128);
    benchmark_all(1, 4096 * 3, 4096, 128);
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096 * 3, 4096, 32);
  }

  using AType = sycl::half;
  using BType = sycl::half;
  using CType = sycl::half;
  using GemmT = xve::DefaultHGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, sycl::half>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightS4<GCT, sycl::half>;
  template <class GCT>
  using ProBTransT = sycl_prologue_b::WeightS4Trans<GCT, sycl::half>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, sycl::half>;
  using KernelLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBT, EpiT, GemmT>;
  using KernelTLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBTransT, EpiT, GemmT>;

  template <typename LOG_T>
  void benchmark_gemm(int m, int n, int k, int blocksize, int batch, AType* A, uint8_t* B, BType* B_scale, CType* C,
                      float timems) {
    LOG_T log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = (const AType*)A;
    auto B_d = B;
    auto C_d = C;
    auto S_d = B_scale;
    auto psize = (size_t)m * n * k * 2;
    int blks = k / blocksize;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        auto ev = KernelLauncher::compute(q, m, n, k, blocksize, {{A_d, k}, {B_d, S_d, n}, {C_d, n}});
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f\n", log.get_log_str(), flops);
  }
#if 0
  template <typename LOG_T>
  void benchmark_gemmT(int m, int n, int k, int blocksize, int batch, AType* A, uint8_t* B, BType* B_scale, CType* C,
                       float timems) {
    LOG_T log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = (const AType*)A;
    auto B_d = B;
    auto C_d = C;
    auto S_d = B_scale;
    auto psize = (size_t)m * n * k * 2;
    int blks = k / blocksize;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        auto ev = KernelTLauncher::compute({m, n, k, blocksize, {A_d, k}, {B_d, S_d, blks}, {C_d, n}}, q);
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f\n", log.get_log_str(), flops);
  }

  template <typename LOG_T>
  void benchmark_gemmT_DQ(int m, int n, int k, int blocksize, int batch, AType* A, uint8_t* B, BType* B_scale,
                          BType* DQB, CType* C, float timems) {
    LOG_T log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = (const AType*)A;
    auto B_d = B;
    auto C_d = C;
    auto S_d = B_scale;
    auto psize = (size_t)m * n * k * 2;
    int blks = k / blocksize;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        auto ev = KernelTLauncher::compute({m, n, k, blocksize, {A_d, k}, {B_d, S_d, blks}, {C_d, n}}, q);
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f\n", log.get_log_str(), flops);
  }
#endif
  template <typename LOG_T>
  void benchmark_gemv_T2(int m, int n, int k, int blocksize, int batch, AType* A, uint8_t* B, BType* B_scale, CType* C,
                         float timems) {
    LOG_T log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = (const AType*)A;
    auto B_d = B;
    auto C_d = C;
    auto S_d = B_scale;
    auto psize = (size_t)m * n * k * 2;
    int blks = k / blocksize;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        auto ev = ProBTransT<GemmT>::gemv(A_d, {B_d, S_d, blks}, C_d, n, k, blocksize, q);
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f\n", log.get_log_str(), flops);
  }

  void benchmark_all(int m, int n, int k, int blocksize) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F16, BTLA_DTYPE::F16, BTLA_DTYPE::F16);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F16),
           bestla_dtype_str(BTLA_DTYPE::F16), bestla_dtype_str(BTLA_DTYPE::F16));

    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    int blks = k / blocksize;
    sycl_vector<AType> dA(size_t(m) * k * batch, q);
    sycl_vector<CType> dC(size_t(m) * n * batch, q);
    sycl_vector<BType> dB_scale(blks * n * batch, q);
    sycl_vector<uint8_t> dBs8(size_t(n) * k * batch / 2, q);
    if (m == 1) {
      benchmark_gemv_T2<LOG>(m, n, k, blocksize, batch, dA.data(), dBs8.data(), dB_scale.data(), dC.data(), testtime);
    } else {
      benchmark_gemm<LOG>(m, n, k, blocksize, batch, dA.data(), dBs8.data(), dB_scale.data(), dC.data(), testtime);
    }
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_S4Fp16Fp16 sBenchmark_S4Fp16Fp16;
#endif

class Benchmark_DequantS4 {
 public:
  Benchmark_DequantS4() {
    UT_START();
    benchmark_all_reorder_back(4096, 4096, 32);
    // benchmark_all_reorder_back_half(4096, 4096, 32);
    benchmark_all_reorder(4096, 4096, 32);
    benchmark_all_reorder(16384, 4096, 32);
    benchmark_all(4096, 4096, 32);
    benchmark_all(16384, 4096, 32);
    benchmark_all(16384, 16384, 32);
    benchmark_memcpy(2480, 4096, 32);
    benchmark_memcpy(16384, 16384, 32);
  }
  void benchmark_all_reorder_back(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 2) {
        auto tmp = srcptr[i / 2 + j * k / 2];
        auto noffset = i / blocksize + j * blks;
        ref[i + j * k] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset];
        ref[i + 1 + j * k] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset];
      }
    }
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();

    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    using ProB = sycl_prologue_b::WeightS4Trans<sycl_gemm::xve::DefaultSGemmCore, float>;
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < TestMs) {
      for (size_t i = 0; i < 1; i++) {
        auto ev =
            ProB::dequant_s4_trans<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= TestMs) {
          break;
        }
      }
    }
    avector<float> refNT(k * n);
    kernel::wrapper::Transpose2D<float>::forward<BTLA_ISA::NoSIMD>(ref.data(), refNT.data(), n, k, k, n);
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(refNT.data(), dequant.data(), dequant.size(), 0.001f);
    log.record();
    auto psize = (size_t)n * k * 4 + n * k / 2 + n * k / blocksize * 4;
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }

  void benchmark_all_reorder_back_half(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<utils::fp16> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), utils::fp16(0.01f), utils::fp16(0.03f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 2) {
        auto tmp = srcptr[i / 2 + j * k / 2];
        auto noffset = i / blocksize + j * blks;
        auto s = float(scale[noffset]);
        ref[i + j * k] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * s;
        ref[i + 1 + j * k] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * s;
      }
    }
    sycl_vector<sycl::half> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 2).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();

    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    using ProB = sycl_prologue_b::WeightS4Trans<sycl_gemm::xve::DefaultHGemmCore, sycl::half>;
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < TestMs) {
      for (size_t i = 0; i < 1; i++) {
        auto ev =
            ProB::dequant_s4_trans<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= TestMs) {
          break;
        }
      }
    }
    avector<utils::fp16> refNT(k * n);
    kernel::wrapper::Transpose2D<utils::fp16>::forward<BTLA_ISA::NoSIMD>(ref.data(), refNT.data(), n, k, k, n);
    q->memcpy(dequant.data(), DB_d, dequant.size() * 2).wait();
    buffer_error(refNT.data(), dequant.data(), dequant.size(), utils::fp16(0.001f));
    log.record();
    auto psize = (size_t)n * k * 2 + n * k / 2 + n * k / blocksize * 2;
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }

  void benchmark_all_reorder(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int j = 0; j < n; j += 1) {
      for (int i = 0; i < k; i += 2) {
        auto tmp = srcptr[i / 2 + j * k / 2];
        auto noffset = i / blocksize + j * blks;
        ref[i + j * k] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset];
        ref[i + 1 + j * k] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset];
      }
    }
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();

    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    using ProB = sycl_prologue_b::WeightS4Trans<sycl_gemm::xve::DefaultSGemmCore, float>;
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < TestMs) {
      for (size_t i = 0; i < 1; i++) {
        auto ev = ProB::dequant_s4<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= TestMs) {
          break;
        }
      }
    }

    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);
    log.record();
    auto psize = (size_t)n * k * 4 + n * k / 2 + n * k / blocksize * 4;
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }

  void benchmark_all(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j += 2) {
        auto tmp = srcptr[i * n / 2 + j / 2];
        auto noffset = i / blocksize * n + j;
        ref[i * n + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset + 0];
        ref[i * n + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset + 1];
      }
    }
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();

    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    utils::timer<std::chrono::milliseconds> tm;
    using ProB = sycl_prologue_b::WeightS4<sycl_gemm::xve::DefaultSGemmCore, float>;
    tm.start();
    while (tm.stop() < TestMs) {
      for (size_t i = 0; i < 1; i++) {
        auto ev = ProB::dequant_s4<sycl_prologue_b::KernelConfigBase>(n, k, blocksize, {dB.data(), dS.data(), n},
                                                                      dequantB.data(), q);
        ev.wait();
        log.add(event_helper::execute_time(ev) * 1000);
        if (tm.stop() >= TestMs) {
          break;
        }
      }
    }

    q->memcpy(dequant.data(), dequantB.data(), dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);
    log.record();
    auto psize = (size_t)n * k * 4 + n * k / 2 + n * k / blocksize * 4;
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }

  void benchmark_memcpy(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<float> dequant(n * k);
    fill_buffer_randn(dequant.data(), dequant.size(), 0.01f, 0.03f);
    sycl_vector<float> dequantB0(n * k, q);
    sycl_vector<float> dequantB1(n * k, q);
    q->memcpy(dequantB0.data(), dequant.data(), dequant.size() * 4).wait();

    using LOG = timer_statistics_logger<TestMs * 2>;
    LOG log;
    utils::timer<std::chrono::milliseconds> tm;
    tm.start();
    while (tm.stop() < TestMs) {
      for (size_t i = 0; i < 1; i++) {
        auto e = q->memcpy(dequantB1.data(), dequantB0.data(), dequantB0.size() * 4);
        e.wait();
        log.add(event_helper::execute_time(e) * 1000);
        if (tm.stop() >= TestMs) {
          break;
        }
      }
    }

    log.record();
    auto psize = (size_t)n * k * 4 * 2;
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Memory Bandwidth:%.3f\n", log.get_log_str(), flops);
  }
};
#ifdef BTLA_UT_SYCL
static Benchmark_DequantS4 sBenchmark_DequantS4;
#endif
}  // namespace sycl_ut
}  // namespace bestla
