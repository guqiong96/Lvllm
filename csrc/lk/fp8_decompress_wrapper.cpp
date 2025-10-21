#include "bestla/bestla.h"
#include "bestla/bestla_utils.h"
#include "bestla/kernel_ref.h"
#include "bestla/kernel_wrapper.h"
#include <cstdint>
 
#ifdef __cplusplus
extern "C" {
#endif
 
void decompress_kblock_f8_fp_wrapper(
    void* srcptr, void* dstptr, int row, int col, int ld_src, 
    int ld_dst, void* scales, int k_offset, int kblock, int NPad, int src_f8_type) {
    using namespace bestla;
    using namespace bestla::kernel;
    using namespace bestla::kernel::wrapper;
    using namespace bestla::utils;
     
    DecompressKBlockF8FP<8>::template forward<
        BTLA_ISA::AVX2, float, float>(
            static_cast<f8*>(srcptr),
            static_cast<float*>(dstptr),
            row, col, ld_src, ld_dst,
            static_cast<float*>(scales),
            k_offset, kblock, NPad,
            static_cast<BTLA_DTYPE>(src_f8_type)
        );
}
 
void decompress_kblock_f8_fp_noscale_wrapper(
    void* srcptr, void* dstptr, int row, int col, int ld_src, 
    int ld_dst, int src_f8_type) {
    using namespace bestla;
    using namespace bestla::kernel;
    using namespace bestla::kernel::wrapper;
    using namespace bestla::utils;
     
    int8_t tmp[1024];  
     
    DecompressKBlockF8FpNoScale<float>::template forward<
        BTLA_ISA::AVX2>(
            static_cast<f8*>(srcptr),
            static_cast<float*>(dstptr),
            row, col, ld_src, ld_dst,
            tmp, sizeof(tmp),
            static_cast<BTLA_DTYPE>(src_f8_type)
        );
}

#ifdef __cplusplus
}
#endif