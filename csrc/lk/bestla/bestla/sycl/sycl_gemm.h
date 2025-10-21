//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once

#ifdef BTLA_SYCL
#include <array>

#include "bestla/bestla_utils.h"
#include <sycl/sycl.hpp>

namespace bestla {
namespace sycl_gemm {
namespace xve {
class Config_Fp32Fp32Fp32 {
 public:
  static int constexpr sg_size = 16;
  static int constexpr sg_m = 16;
  static int constexpr sg_n = 2;
  static int constexpr sg_k = 32;
  static int constexpr unroll_k = 4;
  static int constexpr wg_m = 8;
  static int constexpr wg_n = 32;

  using data_type_a = float;
  using data_type_b = float;
  using data_type_c = float;
  using data_type_acc = float;
};

template <class ConfigT>
class SGemmCoreSharedB {
 public:
  static int constexpr SgSize = ConfigT::sg_size;
  static int constexpr WgM = ConfigT::wg_m;
  static int constexpr WgN = ConfigT::wg_n;
  static int constexpr SgNStride = WgN / SgSize;
  static int constexpr WgWorkers = WgM * WgN;
  static int constexpr SgCount = WgWorkers / SgSize;
  static int constexpr TileM = ConfigT::sg_m;
  static int constexpr TileN = ConfigT::sg_n;
  static int constexpr TileK = ConfigT::sg_k;
  static int constexpr UnrollK = ConfigT::unroll_k;
  static int constexpr WgNEle = WgN * TileN;
  static int constexpr WgMEle = WgM * TileM;
  static int constexpr SgNEle = SgSize * TileN;
  static int constexpr SLM_B_Size = WgNEle * TileK;
  static int constexpr SLM_A_Size = 0;

  using TA = typename ConfigT::data_type_a;
  using TB = typename ConfigT::data_type_b;
  using TC = typename ConfigT::data_type_c;
  using TACC = typename ConfigT::data_type_acc;

  using SLM_B_Acc = sycl::local_accessor<TB, 1>;

  using AType = TA;
  using BType = TB;
  using CType = TC;
  static auto constexpr NTILE = WgNEle;
  static auto constexpr MTILE = WgMEle;
  static auto constexpr KTILE = TileK;
  static auto constexpr PACK_ROW = 1;
  static int constexpr PREFERRED_N = NTILE;
  static auto constexpr ISA = BTLA_ISA::ISA_COUNT;
  static auto constexpr ID = 0;

  static inline void compute(const TA* aptr, int lda, const SLM_B_Acc& bacc, TACC* accptr,
                             const sycl_utils::nd_item_helper<SGemmCoreSharedB<ConfigT>>& helper) {
#pragma unroll(1)
    for (int ik = 0; ik < TileK; ik += UnrollK) {
      int constexpr MReg = TileM / SgSize;
      TA regA[UnrollK * MReg];
      for (int im = 0; im < MReg; im++) {
        *(sycl::vec<TA, UnrollK>*)&regA[im * UnrollK] =
            *(sycl::vec<TA, UnrollK>*)&aptr[(helper.sg_id() + im * SgSize) * lda + ik];
      }

#pragma unroll
      for (int ikk = 0; ikk < UnrollK; ikk++) {
        TB tmpB[TileN];
#pragma unroll
        for (int in = 0; in < TileN; in++) {
          tmpB[in] = bacc[helper.sg_idx_n() * SgNEle + helper.sg_id() * TileN + in + (ik + ikk) * WgNEle];
        }
#pragma unroll
        for (size_t im = 0; im < TileM; im++) {
          auto tmpA = helper.sg.shuffle(regA[ikk + im / SgSize * UnrollK], im % SgSize);
#pragma unroll
          for (size_t in = 0; in < TileN; in++) {
            accptr[im * TileN + in] += tmpA * tmpB[in];
          }
        }
      }
    }
  }

  static inline void compute_mtail(const TA* aptr, int lda, const SLM_B_Acc& bacc, TACC* accptr,
                                   const sycl_utils::nd_item_helper<SGemmCoreSharedB<ConfigT>>& helper, int& m_tail) {
    if (m_tail > 0) {
#pragma unroll(1)
      for (int ik = 0; ik < TileK; ik += UnrollK) {
        for (int ikk = 0; ikk < UnrollK; ikk++) {
          TB tmpB[TileN];
#pragma unroll
          for (int in = 0; in < TileN; in++) {
            tmpB[in] = bacc[helper.sg_idx_n() * SgNEle + helper.sg_id() * TileN + in + (ik + ikk) * WgNEle];
          }
          for (size_t im = 0; im < m_tail; im++) {
            auto tmpA = aptr[im * lda + ik + ikk];
#pragma unroll
            for (size_t in = 0; in < TileN; in++) {
              accptr[im * TileN + in] += tmpA * tmpB[in];
            }
          }
        }
      }
    }
  }
};

using DefaultSGemmCore = SGemmCoreSharedB<Config_Fp32Fp32Fp32>;

class Config_Fp16Fp16Fp16 {
 public:
  static int constexpr sg_size = 16;
  static int constexpr sg_m = 16;
  static int constexpr sg_n = 4;
  static int constexpr sg_k = 32;
  static int constexpr unroll_k = 4;
  static int constexpr wg_m = 16;
  static int constexpr wg_n = 32;

  using data_type_a = sycl::half;
  using data_type_b = sycl::half;
  using data_type_c = sycl::half;
  using data_type_acc = sycl::half;
};

template <class ConfigT>
class HGemmCoreSharedB {
 public:
  static int constexpr SgSize = ConfigT::sg_size;
  static int constexpr WgM = ConfigT::wg_m;
  static int constexpr WgN = ConfigT::wg_n;
  static int constexpr SgNStride = WgN / SgSize;
  static int constexpr WgWorkers = WgM * WgN;
  static int constexpr SgCount = WgWorkers / SgSize;
  static int constexpr TileM = ConfigT::sg_m;
  static int constexpr TileN = ConfigT::sg_n;
  static int constexpr TileK = ConfigT::sg_k;
  static int constexpr UnrollK = ConfigT::unroll_k;
  static int constexpr WgNEle = WgN * TileN;
  static int constexpr WgMEle = WgM * TileM;
  static int constexpr SgNEle = SgSize * TileN;
  static int constexpr SLM_B_Size = WgNEle * TileK;
  static int constexpr SLM_A_Size = 0;

  using TA = typename ConfigT::data_type_a;
  using TB = typename ConfigT::data_type_b;
  using TC = typename ConfigT::data_type_c;
  using TACC = typename ConfigT::data_type_acc;

  using SLM_B_Acc = sycl::local_accessor<TB, 1>;

  static inline void compute(const TA* aptr, int lda, const SLM_B_Acc& bacc, TACC* accptr,
                             const sycl_utils::nd_item_helper<HGemmCoreSharedB<ConfigT>>& helper) {
#pragma unroll(1)
    for (int ik = 0; ik < TileK; ik += UnrollK) {
      static_assert((UnrollK * sizeof(TA)) % sizeof(float) == 0);
      int constexpr MReg = TileM / SgSize;
      static_assert(MReg == 1);
      TA regA[UnrollK * MReg];
      for (int im = 0; im < MReg; im++) {
        *(sycl::vec<TA, UnrollK>*)&regA[im * UnrollK] =
            *(sycl::vec<TA, UnrollK>*)&aptr[(helper.sg_id() + im * SgSize) * lda + ik];
      }
#pragma unroll
      for (int ikk = 0; ikk < UnrollK; ikk++) {
        TB tmpB[TileN];
#pragma unroll
        for (int in = 0; in < TileN; in++) {
          tmpB[in] = bacc[helper.sg_idx_n() * SgNEle + helper.sg_id() * TileN + in + (ik + ikk) * WgNEle];
        }
#pragma unroll
        for (size_t im = 0; im < TileM; im++) {
          auto tmpA = helper.sg.shuffle(regA[ikk + im / SgSize * UnrollK], im % SgSize);
#pragma unroll
          for (size_t in = 0; in < TileN; in++) {
            accptr[im * TileN + in] += tmpA * tmpB[in];
          }
        }
      }
    }
  }

  static inline void compute_mtail(const TA* aptr, int lda, const SLM_B_Acc& bacc, TACC* accptr,
                                   const sycl_utils::nd_item_helper<HGemmCoreSharedB<ConfigT>>& helper,
                                   const int& m_tail) {
    if (m_tail > 0) {
#pragma unroll(1)
      for (int ik = 0; ik < TileK; ik += UnrollK) {
#pragma unroll
        for (int ikk = 0; ikk < UnrollK; ikk++) {
          TB tmpB[TileN];
#pragma unroll
          for (int in = 0; in < TileN; in++) {
            tmpB[in] = bacc[helper.sg_idx_n() * SgNEle + helper.sg_id() * TileN + in + (ik + ikk) * WgNEle];
          }
          for (size_t im = 0; im < m_tail; im++) {
            auto tmpA = aptr[im * lda + ik + ikk];
#pragma unroll
            for (size_t in = 0; in < TileN; in++) {
              accptr[im * TileN + in] += tmpA * tmpB[in];
            }
          }
        }
      }
    }
  }
};

using DefaultHGemmCore = HGemmCoreSharedB<Config_Fp16Fp16Fp16>;
}  // namespace xve

}  // namespace sycl_gemm
}  // namespace bestla
#endif
