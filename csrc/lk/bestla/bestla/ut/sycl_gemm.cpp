#include "bestla_ut.h"
#include "sycl_ut.h"
#include "../sycl/sycl_wrapper.h"
#include "bestla_prologue_b.h"
#undef BTLA_UT_SYCL
namespace bestla {
using namespace ut;
using namespace utils;
using namespace sycl_utils;
using namespace sycl_gemm;
namespace sycl_ut {
class UT_SyclSGemm {
 public:
  UT_SyclSGemm() {
    UT_START();
    ut(1, 1024, 1024);
    ut(300, 1024, 1024);
    ut(1024, 1024, 1024);
  }
  using SGemmT = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemmT>;

  void ut(int m, int n, int k) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<float> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), -0.5f, 0.5f);
    fill_buffer_randn(matB.data(), matB.size(), -0.5f, 0.5f);
    gemmref_fp32fp32fp32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    sycl_vector<float> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 4).wait();
    q->memcpy(dB.data(), matB.data(), matB.size() * 4).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto ev = KernelLauncher::compute(q, m, n, k, {{A_d, k}, {B_d, n}, {C_d, n}});
    ev.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclSGemm sUT_SyclSGemm;
#endif

class UT_SyclHGemm {
 public:
  UT_SyclHGemm() {
    UT_START();
    ut(1, 1024, 1024);
    ut(300, 1024, 1024);
    ut(1024, 1024, 1024);
    ut(1033, 1024, 1024);
  }
  using SGemmT = xve::DefaultHGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, sycl::half>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, sycl::half>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, sycl::half>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemmT>;

  void ut(int m, int n, int k) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::fp16> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matB.data(), matB.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    gemmref_fp16fp16fp16(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);

    sycl_vector<sycl::half> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dB.data(), matB.data(), matB.size() * 2).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto ev = KernelLauncher::compute(q, m, n, k, {{A_d, k}, {B_d, n}, {C_d, n}});
    ev.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), utils::fp16(0.2f));
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclHGemm sUT_SyclHGemm;
#endif

class UT_SyclS4SGemm {
 public:
  UT_SyclS4SGemm() {
    UT_START();
    utT(6, 4096, 11008, 128);
    ut(6, 32000, 4096, 128);
    utT(6, 32000, 4096, 128);
    ut(300, 1024, 1024, 32);
    ut(1024, 1024, 1024, 32);
    utT(1024, 1024, 1024, 32);
  }
  using SGemm_t = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightS4<GCT, float>;
  template <class GCT>
  using ProBTransT = sycl_prologue_b::WeightS4Trans<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBT, EpiT, SGemm_t>;
  using KernelTLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBTransT, EpiT, SGemm_t>;

  void ut(int m, int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<float> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), -0.5f, 0.5f);
    int blks = k / blocksize;
    avector<float> B_scale(size_t(blks) * n);
    avector<uint8_t> B_s8(k * n / 2);
    fill_buffer_randn(B_s8.data(), B_s8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(B_scale.data(), B_scale.size(), 0.001f, 0.005f);
    auto srcptr = (utils::int4x2*)B_s8.data();
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j += 2) {
        auto tmp = srcptr[i * n / 2 + j / 2];
        auto noffset = i / blocksize * n + j;
        matB[i * n + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * B_scale[noffset + 0];
        matB[i * n + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * B_scale[noffset + 1];
      }
    }
    gemmref_fp32fp32fp32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    sycl_vector<float> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q), dB_scale(B_scale.size(), q);
    sycl_vector<uint8_t> dBs8(B_s8.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 4).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 4).wait();
    sycl::range<2> group{SGemm_t::WgM, SGemm_t::WgN};
    sycl::range<2> problem{m / SGemm_t::TileM, n / SGemm_t::TileN};
    auto A_d = dA.data();
    auto Bs8_d = dBs8.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();
    utils::GemmProblem gp(1, m, n, k);
    auto ev = KernelLauncher::compute(q, m, n, k, blocksize, {{A_d, k}, {Bs8_d, B_scale_d, n}, {C_d, n}});
    ev.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }

  void utT(int m, int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<float> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), -0.5f, 0.5f);
    int blks = k / blocksize;
    avector<float> B_scale(size_t(blks) * n);
    avector<uint8_t> B_s8(k * n / 2);
    fill_buffer_randn(B_s8.data(), B_s8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(B_scale.data(), B_scale.size(), 0.001f, 0.005f);
    auto srcptr = (utils::int4x2*)B_s8.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        matB[i * k + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * B_scale[noffset];
        matB[i * k + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * B_scale[noffset];
      }
    }
    avector<float> matBNT(k * n);
    kernel::wrapper::Transpose2D<float>::forward<BTLA_ISA::NoSIMD>(matB.data(), matBNT.data(), n, k, k, n);
    gemmref_fp32fp32fp32(m, n, k, matA.data(), matBNT.data(), ref.data(), k, n, n);
    sycl_vector<float> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q), dB_scale(B_scale.size(), q);
    sycl_vector<uint8_t> dBs8(B_s8.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 4).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 4).wait();
    sycl::range<2> group{SGemm_t::WgM, SGemm_t::WgN};
    sycl::range<2> problem{m / SGemm_t::TileM, n / SGemm_t::TileN};
    auto A_d = dA.data();
    auto Bs8_d = dBs8.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();
    utils::GemmProblem gp(1, m, n, k);
    auto ev = KernelTLauncher::compute(q, m, n, k, blocksize, {{A_d, k}, {Bs8_d, B_scale_d, blks}, {C_d, n}});
    ev.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclS4SGemm sUT_SyclS4SGemm;
#endif

class UT_SyclS4HGemm {
 public:
  UT_SyclS4HGemm() {
    UT_START();
    ut(300, 1024, 1024, 32);
    ut(1024, 1024, 1024, 32);
    utT(1024, 1024, 1024, 32);
  }
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

  void ut(int m, int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::fp16> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    int blks = k / blocksize;
    avector<utils::fp16> B_scale(size_t(blks) * n);
    avector<uint8_t> B_s8(k * n / 2);
    fill_buffer_randn(B_s8.data(), B_s8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(B_scale.data(), B_scale.size(), utils::fp16(0.001f), utils::fp16(0.005f));
    auto srcptr = (utils::int4x2*)B_s8.data();
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j += 2) {
        auto tmp = srcptr[i * n / 2 + j / 2];
        auto noffset = i / blocksize * n + j;
        matB[i * n + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * float(B_scale[noffset + 0]);
        matB[i * n + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * float(B_scale[noffset + 1]);
      }
    }
    gemmref_fp16fp16fp16(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    sycl_vector<sycl::half> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q), dB_scale(B_scale.size(), q);
    sycl_vector<uint8_t> dBs8(B_s8.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 2).wait();
    auto A_d = dA.data();
    auto Bs8_d = dBs8.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();
    auto ev = KernelLauncher::compute(q, m, n, k, blocksize, {{A_d, k}, {Bs8_d, B_scale_d, n}, {C_d, n}});
    ev.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), utils::fp16(0.2f));
  }

  void utT(int m, int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::fp16> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    int blks = k / blocksize;
    avector<utils::fp16> B_scale(size_t(blks) * n);
    avector<uint8_t> B_s8(k * n / 2);
    fill_buffer_randn(B_s8.data(), B_s8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(B_scale.data(), B_scale.size(), utils::fp16(0.001f), utils::fp16(0.005f));
    auto srcptr = (utils::int4x2*)B_s8.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        matB[i * k + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * float(B_scale[noffset]);
        matB[i * k + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * float(B_scale[noffset]);
      }
    }
    avector<utils::fp16> matBNT(k * n);
    kernel::wrapper::Transpose2D<utils::fp16>::forward<BTLA_ISA::NoSIMD>(matB.data(), matBNT.data(), n, k, k, n);
    gemmref_fp16fp16fp16(m, n, k, matA.data(), matBNT.data(), ref.data(), k, n, n);
    sycl_vector<sycl::half> dA(matA.size(), q), dC(matC.size(), q), dB_scale(B_scale.size(), q);
    sycl_vector<uint8_t> dBs8(B_s8.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 2).wait();
    auto A_d = dA.data();
    auto Bs8_d = dBs8.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();
    auto ev = KernelTLauncher::compute(q, m, n, k, blocksize, {{A_d, k}, {Bs8_d, B_scale_d, blks}, {C_d, n}});
    ev.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), utils::fp16(0.2f));
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclS4HGemm sUT_SyclS4HGemm;
#endif

class UT_SyclInt4Dequant {
 public:
  UT_SyclInt4Dequant() {
    UT_START();
    ut_fp32(1024, 1024, 32);
    ut_fp32_T(1024, 1024, 32);
  }

  void ut_fp32(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", n, k, blocksize, dev->getName().c_str());
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
    using ProB = sycl_prologue_b::WeightS4<sycl_gemm::xve::DefaultSGemmCore, float>;
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto ev = ProB::dequant_s4<sycl_prologue_b::KernelConfigBase>(n, k, blocksize, {B_d, S_d, n}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);
  }

  void ut_fp32_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        ref[i * k + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset];
        ref[i * k + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset];
      }
    }
    using ProB = sycl_prologue_b::WeightS4Trans<sycl_gemm::xve::DefaultSGemmCore, float>;
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto ev = ProB::dequant_s4<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);

    avector<float> refNT(k * n);
    kernel::wrapper::Transpose2D<float>::forward<BTLA_ISA::NoSIMD>(ref.data(), refNT.data(), n, k, k, n);
    ev = ProB::dequant_s4_trans<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
    ev.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(refNT.data(), dequant.data(), dequant.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclInt4Dequant sUT_SyclInt4Dequant;
#endif

class UT_SyclS4Gemv {
 public:
  UT_SyclS4Gemv() {
    UT_START();
    ut_T(1024, 11008, 32);
    ut_T(1024, 1024, 32);
    ut_half(1024, 11008, 32);
    ut_half(1024, 1024, 32);
  }
  using SGemm_t = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, float>;
  template <class GCT>
  using ProBTransT = sycl_prologue_b::WeightS4Trans<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemm_t>;

  void ut_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(A.data(), A.size(), -0.1f, 0.3f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset];
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset];
      }
    }
    gemmref_fp32fp32fp32(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<float> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    int constexpr SgSize = 16;
    int constexpr TileK = 2;
    int constexpr GroupK = SgSize * TileK;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{n * SgSize};
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto ev = ProBTransT<SGemm_t>::gemv(A_d, {B_d, S_d, blks}, C_d, n, k, blocksize, q);
    ev.wait();
    q->memcpy(C.data(), C_d, C.size() * 4).wait();
    buffer_error(refC.data(), C.data(), C.size(), 0.001f);
  }

  void ut_half(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<utils::fp16> scale(blks * n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), utils::fp16(0.01f), utils::fp16(0.03f));
    fill_buffer_randn(A.data(), A.size(), utils::fp16(-0.1f), utils::fp16(0.3f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        float fscale = float(scale[noffset]);
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * fscale;
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * fscale;
      }
    }
    gemmref_fp16fp16fp16(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<sycl::half> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 2).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 2).wait();
    int constexpr SgSize = 16;
    int constexpr TileK = 32;
    int constexpr GroupK = SgSize * TileK;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{n * SgSize};
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto ev = sycl_prologue_b::WeightS4Trans<xve::DefaultHGemmCore, sycl::half>::gemv(A_d, {B_d, S_d, blks}, C_d, n, k,
                                                                                      blocksize, q);
    ev.wait();
    q->memcpy(C.data(), C_d, C.size() * 2).wait();
    buffer_error(refC.data(), C.data(), C.size(), utils::fp16(0.1f));
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclS4Gemv sUT_SyclS4Gemv;
#endif

void mha_sref(float* Q, float* K, float* V, float* S, float* O, int batch, int seq, int seqA, int hnum, int hsize) {
  avector<float> tmps(seqA);
  int nf = hnum * hsize;
  const float attn_scale = 1.0f / sqrtf(static_cast<float>(hsize));
  int n_past = seqA - seq;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < seq; j++) {
      for (int ii = 0; ii < hnum; ii++) {
        float maxs = 0.f;
        for (int jj = 0; jj < seqA; jj++) {
          float tmp = 0.f;
          if (jj <= j + n_past) {
            for (int kk = 0; kk < hsize; kk++) {
              tmp +=
                  Q[i * seq * nf + j * nf + ii * hsize + kk] * K[i * nf * seqA + ii * seqA * hsize + jj * hsize + kk];
            }
            tmp *= attn_scale;
          } else {
            tmp = -INFINITY;
          }

          tmps[jj] = tmp;
          maxs = std::max(maxs, tmp);
        }
        float sums = 0.f;
        for (int jj = 0; jj < seqA; jj++) {
          tmps[jj] = std::expf(tmps[jj] - maxs);
          sums += tmps[jj];
        }
        sums = 1.f / sums;
        for (int jj = 0; jj < seqA; jj++) {
          tmps[jj] *= sums;
          S[i * seq * hnum * seqA + j * hnum * seqA + ii * seqA + jj] = tmps[jj];
        }
        for (int kk = 0; kk < hsize; kk++) {
          float tmp = 0.f;
          for (int jj = 0; jj < seqA; jj++) {
            tmp += tmps[jj] * V[i * nf * seqA + ii * hsize * seqA + kk * seqA + jj];
          }
          O[i * seq * nf + j * nf + ii * hsize + kk] = tmp;
        }
      }
    }
  }
}

class UT_MHASgemm {
 public:
  UT_MHASgemm() {
    UT_START();
    ut_T(1, 1, 1, 32, 128);
    ut_T(1, 1, 64, 32, 128);
    ut_T(4, 1, 64, 32, 128);
    ut_T(4, 64, 64, 32, 128);
  }
  template <typename T, typename T_DST>
  class MHA {
   public:
    template <bool Mask>
    static sycl::event forward(int batch, int seq, int seq_acc, int hnum, int hsize, const T* Q, const T* K, const T* V,
                               T_DST* O, sycl::queue* q) {
      const float attn_scale = 1.0f / sqrtf(static_cast<float>(hsize));
      int constexpr SgSize = 16;
      assert(hsize % SgSize == 0);
      int n_past = seq_acc - seq;
      if constexpr (Mask) {
        assert(seq > 1);
      }
      int WgSize = SgSize;
      int seq_acc_pad = utils::padto_le(seq_acc, WgSize * 2);
      int nf = hnum * hsize;
      auto ev = q->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<T, 1> slm(sycl::range(std::max(seq_acc, 1024)), cgh);
        cgh.parallel_for(sycl::nd_range<1>(WgSize * batch * seq * hnum, WgSize),
                         [=](auto it) [[intel::reqd_sub_group_size(SgSize)]] {
                           auto sg = it.get_sub_group();
                           auto sg_idx = sg.get_group_id()[0];
                           auto wg_idx = it.get_group(0);
                           auto wg_loc_id = it.get_local_id();
                           auto lane_id = sg.get_local_id()[0];

                           int i = wg_idx;
                           int ih = i % hnum;
                           i /= hnum;
                           int is = i % seq;
                           i /= seq;
                           int ib = i % batch;
                           size_t Q_off = ib * seq * nf + is * nf + ih * hsize;
                           size_t K_off = ib * seq_acc * nf + ih * hsize * seq_acc;
                           size_t V_off = ib * seq_acc * nf + ih * hsize * seq_acc;
                           size_t O_off = ib * seq * nf + is * nf + ih * hsize;
                           typedef sycl::vec<T, 2> TC;
                           T maxs = -INFINITY;
                           for (int jj = 0; jj < seq_acc; jj++) {
                             TC tmp = {0, 0};
                             if constexpr (Mask) {
                               if (jj <= is + n_past) {
                                 for (int ik = wg_loc_id * 2; ik < hsize; ik += WgSize * 2) {
                                   tmp += *(TC*)&Q[Q_off + ik] * *(TC*)&K[K_off + jj * hsize + ik];
                                 }
                                 tmp *= attn_scale;
                               } else {
                                 tmp = {-INFINITY, -INFINITY};
                               }
                             } else {
                               for (int ik = wg_loc_id * 2; ik < hsize; ik += WgSize * 2) {
                                 tmp += *(TC*)&Q[Q_off + ik] * *(TC*)&K[K_off + jj * hsize + ik];
                               }
                               tmp *= attn_scale;
                             }
                             T tmp_sum = tmp[0] + tmp[1];
                             T sum = 0;
                             for (int i = 0; i < SgSize; i += 1) {
                               sum += sg.shuffle(tmp_sum, i);
                             }
                             slm[jj] = sum;
                             maxs = std::max(maxs, sum);
                           }
                           float fsums = 0.f;
                           float fmax = float(maxs);
                           int jj = wg_loc_id * 2;
                           for (; jj < seq_acc_pad; jj += WgSize * 2) {
                             auto s2 = *(TC*)&slm[jj];
                             s2[0] = std::expf(s2[0] - fmax);
                             s2[1] = std::expf(s2[1] - fmax);
                             fsums += s2[0];
                             fsums += s2[1];
                             *(TC*)&slm[jj] = s2;
                           }
                           if (jj < seq_acc) {
                             slm[jj] = std::expf(float(slm[jj]) - fmax);
                             fsums += slm[jj];
                             if (jj + 1 < seq_acc) {
                               slm[jj + 1] = std::expf(float(slm[jj + 1]) - fmax);
                               fsums += slm[jj + 1];
                             }
                           }
                           float gsum = 0;
                           for (int i = 0; i < SgSize; i += 1) {
                             gsum += sg.shuffle(fsums, i);
                           }
                           T scale = 1.f / gsum;
                           jj = wg_loc_id * 2;
                           for (; jj < seq_acc_pad; jj += WgSize * 2) {
                             auto s2 = *(TC*)&slm[jj];
                             s2 *= scale;
                             *(TC*)&slm[jj] = s2;
                           }
                           if (jj < seq_acc) {
                             slm[jj] *= scale;
                             if (jj + 1 < seq_acc) {
                               slm[jj + 1] *= scale;
                             }
                           }

                           for (int kk = 0; kk < hsize; kk++) {
                             TC tmp = {0, 0};
                             jj = wg_loc_id * 2;
                             for (; jj < seq_acc_pad; jj += WgSize * 2) {
                               auto s2 = *(TC*)&slm[jj];
                               auto v2 = *(TC*)&V[V_off + kk * seq_acc + jj];
                               tmp += s2 * v2;
                             }
                             if (jj < seq_acc) {
                               tmp[0] += slm[jj] * V[V_off + kk * seq_acc + jj];
                               if (jj + 1 < seq_acc) {
                                 tmp[1] += slm[jj + 1] * V[V_off + kk * seq_acc + jj + 1];
                               }
                             }
                             T tmp_sum = tmp[0] + tmp[1];
                             T sum = 0;
                             for (int i = 0; i < SgSize; i += 1) {
                               sum += sg.shuffle(tmp_sum, i);
                             }
                             O[O_off + kk] = sum;
                           }
                         });
      });
      return ev;
    }
  };

  void ut_T(int batch, int seq, int seqA, int hnum, int hsize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    assert(seqA >= seq);
    printf("Test Case %s: %d %d %d %d %d Device:%s\n", __FUNCTION__, batch, seq, seqA, hnum, hsize,
           dev->getName().c_str());
    avector<float> Q(batch * seq * hnum * hsize), K(batch * seqA * hnum * hsize), V(batch * seqA * hnum * hsize);
    fill_buffer_randn(Q.data(), Q.size(), -0.5f, 0.5f);
    fill_buffer_randn(K.data(), K.size(), -0.5f, 0.5f);
    fill_buffer_randn(V.data(), V.size(), -0.5f, 0.5f);
    avector<float> S(batch * seq * hnum * seqA), O(batch * seq * hnum * hsize);
    mha_sref(Q.data(), K.data(), V.data(), S.data(), O.data(), batch, seq, seqA, hnum, hsize);
    sycl_vector<float> dQ(batch * seq * hnum * hsize, q), dK(batch * seqA * hnum * hsize, q),
        dV(batch * seqA * hnum * hsize, q);
    sycl_vector<float> dS(batch * seq * hnum * seqA, q), dO(batch * seq * hnum * hsize, q);
    q->memcpy(dQ.data(), Q.data(), Q.size() * sizeof(Q[0]));
    q->memcpy(dK.data(), K.data(), K.size() * sizeof(K[0]));
    q->memcpy(dV.data(), V.data(), V.size() * sizeof(V[0]));
    q->wait();
    auto Qptr = dQ.data();
    auto Kptr = dK.data();
    auto Vptr = dV.data();
    auto Sptr = dS.data();
    auto Optr = dO.data();
    int nf = hnum * hsize;
    sycl::range<1> num_items{batch * seq * hnum};
    int n_past = seqA - seq;
    const float attn_scale = 1.0f / sqrtf(static_cast<float>(hsize));
    if (seq > 1) {
      MHA<float, float>::forward<true>(batch, seq, seqA, hnum, hsize, Qptr, Kptr, Vptr, Optr, q).wait();
    } else {
      MHA<float, float>::forward<false>(batch, seq, seqA, hnum, hsize, Qptr, Kptr, Vptr, Optr, q).wait();
    }
    // auto ev = q->submit([&](sycl::handler& cgh) {
    //   cgh.parallel_for(num_items, [=](auto it) {
    //     int i = it;
    //     int ih = i % hnum;
    //     i /= hnum;
    //     int is = i % seq;
    //     i /= seq;
    //     int ib = i % batch;
    //     float maxs = 0.f;
    //     float tmps[64];
    //     for (int jj = 0; jj < seqA; jj++) {
    //       float tmp = 0.f;
    //       if (jj <= is + n_past) {
    //         for (int kk = 0; kk < hsize; kk++) {
    //           tmp += Qptr[ib * seq * nf + is * nf + ih * hsize + kk] *
    //                  Kptr[ib * nf * seqA + kk + ih * seqA * hsize + jj * hsize];
    //         }
    //         tmp *= attn_scale;
    //       } else {
    //         tmp = -INFINITY;
    //       }

    //      tmps[jj] = tmp;
    //      maxs = std::max(maxs, tmp);
    //    }
    //    float sums = 0.f;
    //    for (int jj = 0; jj < seqA; jj++) {
    //      tmps[jj] = std::expf(tmps[jj] - maxs);
    //      sums += tmps[jj];
    //    }
    //    sums = 1.f / sums;
    //    for (int jj = 0; jj < seqA; jj++) {
    //      tmps[jj] *= sums;
    //      Sptr[ib * seq * hnum * seqA + is * hnum * seqA + ih * seqA + jj] = tmps[jj];
    //    }
    //    for (int kk = 0; kk < hsize; kk++) {
    //      float tmp = 0.f;
    //      for (int jj = 0; jj < seqA; jj++) {
    //        tmp += tmps[jj] * Vptr[ib * seqA * nf + jj + ih * hsize * seqA + kk * seqA];
    //      }
    //      Optr[ib * seq * nf + is * nf + ih * hsize + kk] = tmp;
    //    }
    //  });
    //});
    q->wait();
    avector<float> STar(batch * seq * hnum * seqA), OTar(batch * seq * hnum * hsize);
    q->memcpy(STar.data(), Sptr, STar.size() * sizeof(STar[0]));
    q->memcpy(OTar.data(), Optr, OTar.size() * sizeof(OTar[0]));
    q->wait();
    // buffer_error(S.data(), STar.data(), S.size(), 0.001f);
    buffer_error(O.data(), OTar.data(), O.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
#endif
static UT_MHASgemm sUT_MHASgemm;
}  // namespace sycl_ut
}  // namespace bestla
