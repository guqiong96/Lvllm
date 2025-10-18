#include "bestla_ut.h"
#include "bestla_prologue_b.h"
#include "sycl_ut.h"
#include "sycl/sycl_device.h"
#include "sycl/sycl_utils.h"
#include "sycl/sycl_storage.h"
#include "sycl/sycl_gemm.h"
#include "sycl/sycl_prologue_b.h"
namespace bestla {
using namespace ut;
using namespace utils;
namespace sycl_ut {
class UT_SyclDevice {
 public:
  UT_SyclDevice() {
    UT_START();
    auto dev = UT_Device::get();
    dev->print();
  }
};
static UT_SyclDevice sUT_SyclDevice;

class UT_SyclVector {
 public:
  UT_SyclVector() {
    UT_START();
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    auto svec = sycl_utils::sycl_vector<float>(1000, q);
    utils::avector<float> hsrc(1000);
    ut::fill_buffer_randn(hsrc.data(), hsrc.size(), -0.5f, 0.5f);
    q->memcpy(svec.data(), hsrc.data(), hsrc.size() * 4).wait();
    auto hdst = sycl_utils::sycl2host(svec.data(), svec.size(), q);
    ut::buffer_error(hsrc.data(), hdst.data(), hsrc.size(), 0.f);
  }
};
// static UT_SyclVector sUT_SyclVector;

class UT_StorageMemCheck {
 public:
  UT_StorageMemCheck() {
    UT_START();
    ut_s4(4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
  }

  void ut_s4(int n, int k, int blocksize, BTLA_DTYPE qtype, bool asym = false) {
    printf("Test C type Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    using GemmCore = sycl_gemm::xve::DefaultSGemmCore;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<GemmCore>;

    auto packedW =
        PrologueB::createStorage(n, k, blocksize, qtype, bestla_dtype<float>, bestla_dtype<utils::bf16>, asym);
    avector<int8_t> buf0(packedW.mSize), buf1(packedW.mSize);
    packedW.assign(buf0.data());
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    sycl_storage::StorageWeightKBlockNInteger sycl_stor(packedW);
    sycl_utils::sycl_vector<int8_t> dbuf(sycl_stor.getDeviceSize(), q);
    sycl_stor.assign(dbuf.data());
    sycl_stor.fromHost(packedW, q);
    storage::gemm::StorageWeightKBlockNInteger tmp = packedW;
    tmp.assign(buf1.data());
    sycl_stor.toHost(tmp, q);
    buffer_error(buf0.data(), buf1.data(), buf0.size());
  }
};
static UT_StorageMemCheck sUT_StorageMemCheck;

class UT_BlockQunatize_S3S4 {
 public:
  UT_BlockQunatize_S3S4() {
    UT_START();
    using GemmCore = sycl_gemm::xve::DefaultSGemmCore;

    ut<GemmCore>(4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
  }
  template <class GemmCore>
  void ut(int n, int k, int blocksize, BTLA_DTYPE QUANT_T, bool isAsym = false) {
    auto constexpr RuntimeISA = BTLA_ISA::AVX2;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("%s DType %s %d: %d %d %d Asym:%d\n", __FUNCTION__, utils::bestla_dtype_str(QUANT_T), int(RuntimeISA), n, k,
           blocksize, isAsym);
    utils::aligned_vector<float> raw(n * k);
    ut::fill_buffer_randn(raw.data(), raw.size(), -0.5f, 0.5f);
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<GemmCore>;
    auto ptr = PrologueB::createStorage(n, k, blocksize, QUANT_T, BTLA_DTYPE::F32, BTLA_DTYPE::F32, isAsym);
    avector<int8_t> buffer(ptr.mSize);
    ptr.assign(buffer.data());
    PrologueB::packTransposeWeight(n, k, raw.data(), k, &ptr, UT_Threading::get());
    auto transtor = ptr.toTrans();
    avector<int8_t> buffer1(transtor.mSize);
    transtor.assign(buffer1.data());
    PrologueB::convertTransStorage(ptr, transtor, UT_Threading::get());
    sycl_storage::StorageWeightKBlockNInteger sycl_stor(transtor);
    sycl_utils::sycl_vector<int8_t> dbuf(sycl_stor.getDeviceSize(), q);
    sycl_stor.assign(dbuf.data());
    sycl_stor.fromHost(transtor, q);
    avector<float> dequant(n * k, 0);
    using ProB = sycl_prologue_b::WeightS4Trans<GemmCore, float>;
    sycl_utils::sycl_vector<float> dequantB(n * k, q);
    int blks = updiv(k, blocksize);
    auto evt = ProB::dequant_s4<sycl_prologue_b::KernelConfigTrans>(
        n, k, blocksize, {(uint8_t*)sycl_stor.mQBuf, (float*)sycl_stor.mSBuf, blks}, dequantB.data(), q);
    evt.wait();
    q->memcpy(dequant.data(), dequantB.data(), dequantB.size() * 4).wait();
    ut::buffer_error(raw.data(), dequant.data(), dequant.size(), 0.01f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
// no proper threshold for this UT
//
#endif
static UT_BlockQunatize_S3S4 sUT_BlockQunatize_S3S4;

class UT_CompFp32 {
 public:
  UT_CompFp32() {
    UT_START();
    ut_s4();
  }

  void ut_s4() {
    using GemmCore = sycl_gemm::xve::DefaultSGemmCore;
    ut<GemmCore>(1, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F32, false);
    ut<GemmCore>(1, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP, BTLA_DTYPE::F32, false);
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k, int blocksize, BTLA_DTYPE qtype, BTLA_DTYPE stype, bool isAsym) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d-%d type:%s core:%s scaletype:%s Asym:%d\n", __FUNCTION__, m, n, k, blocksize,
           bestla_dtype_str(qtype), gemm::CoreAttr::to_str(GemmCore_T::ID), bestla_dtype_str(stype), isAsym);
    auto constexpr RuntimeISA = BTLA_ISA::AVX2;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<GemmCore_T>;
    blocksize = blocksize == -1 ? k : blocksize;
    auto packedw = PrologueB::createStorage(n, k, blocksize, qtype, stype, bestla_dtype<float>, isAsym);
    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n);
    fill_buffer_randn(matBf32.data(), matBf32.size(), -0.5f, 0.5f);
    fill_buffer_randn(matAf32.data(), matAf32.size(), -0.5f, 0.5f);
    PrologueB::packWeight(n, k, matBf32.data(), n, &packedw, UT_Threading::get());
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    sycl_utils::sycl_vector<float> dC(n, q), dA(k * m, q);
    q->memcpy(dA.data(), matAf32.data(), matAf32.size() * 4).wait();
    using ProBTransT = sycl_prologue_b::WeightS4Trans<GemmCore_T, float>;
    auto transtor = packedw.toTrans();
    avector<int8_t> buffer1(transtor.mSize);
    transtor.assign(buffer1.data());
    PrologueB::convertTransStorage(packedw, transtor, UT_Threading::get());
    sycl_storage::StorageWeightKBlockNInteger sycl_stor(transtor);
    sycl_utils::sycl_vector<int8_t> dbuf(sycl_stor.getDeviceSize(), q);
    sycl_stor.assign(dbuf.data());
    sycl_stor.fromHost(transtor, q);
    int blks = updiv(k, blocksize);
    auto ev = ProBTransT::gemv(dA.data(), {(uint8_t*)sycl_stor.mQBuf, (float*)sycl_stor.mSBuf, blks}, dC.data(), n, k,
                               blocksize, q);
    ev.wait();
    q->memcpy(matC.data(), dC.data(), matC.size() * 4).wait();

    auto err = get_ut_err(qtype);
    auto dbits = bestla_dtype_bits(qtype);
    auto type = bestla_dtype_type(qtype);
    auto constexpr dtype_int = bestla_dtype_type(BTLA_DTYPE::TypeInt);
    buffer_error(refC.data(), matC.data(), refC.size(), err);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
#endif
static UT_CompFp32 sUT_CompFp32;
}  // namespace sycl_ut
}  // namespace bestla
