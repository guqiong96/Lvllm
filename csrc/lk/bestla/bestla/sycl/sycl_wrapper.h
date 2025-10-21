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
#include <sycl/sycl.hpp>

#include "bestla/bestla_utils.h"
#include "sycl_utils.h"
#include "sycl_device.h"
#include "sycl_gemm.h"
#include "sycl_epilogue.h"
#include "sycl_prologue_a.h"
#include "sycl_prologue_b.h"

namespace bestla {
namespace sycl_wrapper {
template <template <class GCT> class ProAT, template <class GCT> class ProBT, template <class GCT> class EpiT,
          class GemmCoreT>
class Launcher {
 public:
  using GemmCore = GemmCoreT;
  using PrologueA = ProAT<GemmCore>;
  using PrologueB = ProBT<GemmCore>;
  using Epilogue = EpiT<GemmCore>;
  using AType = typename GemmCore::TA;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::TB;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::TC;
  using ACCType = typename GemmCore::TACC;
  using EpiParam = typename Epilogue::Param;
  struct Param {
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };
  template <bool debug = false>
  static inline sycl::event compute(sycl::queue* q, int m, int n, int k, const Param& _param) {
    sycl::range<2> group{GemmCore::WgM, GemmCore::WgN};
    auto A = _param.paramA.A;
    auto B = _param.paramB.B;
    auto C = _param.paramC.C;
    int lda = _param.paramA.lda;
    int ldb = _param.paramB.ldb;
    int ldc = _param.paramC.ldc;
    int m_pad = utils::padto(utils::updiv(m, GemmCore::TileM), GemmCore::WgM);
    sycl::range<2> problem{static_cast<size_t>(m_pad), static_cast<size_t>(n) / GemmCore::TileN};
    auto ev = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<BType, 1> slm_b(sycl::range(GemmCore::SLM_B_Size), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>(problem, group),
          [=](sycl::nd_item<2> it) [[sycl::reqd_work_group_size(
              1, GemmCore::WgM,
              GemmCore::WgN)]] [[intel::kernel_args_restrict]] [[intel::reqd_sub_group_size(GemmCore::SgSize)]] {
            sycl_utils::nd_item_helper<GemmCore> helper(it);
            if constexpr (debug) {
              compute_tile(k, B, ldb, slm_b, A, lda, C, ldc, it);
            } else {
              int m_tail = m - helper.sg_g_m();
              m_tail = m_tail > GemmCore::TileM ? GemmCore::TileM : m_tail;
              if (m_tail == GemmCore::TileM) {
                compute_tile(k, B, ldb, slm_b, A, lda, C, ldc, it);
              } else {
                compute_tail(k, B, ldb, slm_b, A, lda, C, ldc, m_tail, it);
              }
            }
          });
    });
    return ev;
  }

  static void compute_tile(int k, const BType* B, int ldb, const sycl::local_accessor<BType, 1>& slm_b, const AType* A,
                           int lda, CType* C, int ldc, sycl::nd_item<2>& it) {
    sycl_utils::nd_item_helper<GemmCore> helper(it);
    ACCType tmp[GemmCore::TileM * GemmCore::TileN];
    for (size_t im = 0; im < GemmCore::TileM; im++)
      for (size_t in = 0; in < GemmCore::TileN; in++) tmp[im * GemmCore::TileN + in] = ACCType(0.f);
#pragma forceinline recursive
    for (int i = 0; i < k; i += GemmCore::TileK) {
      PrologueB::getWeight({B, ldb}, slm_b, i, helper);
      it.barrier(sycl::access::fence_space::local_space);
      GemmCore::compute(&A[helper.item_g_m() * lda + i], lda, slm_b, tmp, helper);
      it.barrier(sycl::access::fence_space::local_space);
    }
#pragma forceinline recursive
    Epilogue::store({C, ldc}, tmp, helper);
  }

  static void compute_tail(int k, const BType* B, int ldb, const sycl::local_accessor<BType, 1>& slm_b, const AType* A,
                           int lda, CType* C, int ldc, int m_tail, sycl::nd_item<2>& it) {
    sycl_utils::nd_item_helper<GemmCore> helper(it);
    ACCType tmp[GemmCore::TileM * GemmCore::TileN];
    for (size_t im = 0; im < GemmCore::TileM; im++)
      for (size_t in = 0; in < GemmCore::TileN; in++) tmp[im * GemmCore::TileN + in] = ACCType(0.f);
#pragma forceinline recursive
    for (int i = 0; i < k; i += GemmCore::TileK) {
      PrologueB::getWeight({B, ldb}, slm_b, i, helper);
      it.barrier(sycl::access::fence_space::local_space);
      GemmCore::compute_mtail(&A[helper.item_g_m() * lda + i], lda, slm_b, tmp, helper, m_tail);
      it.barrier(sycl::access::fence_space::local_space);
    }
#pragma forceinline recursive
    Epilogue::store_tail({C, ldc}, tmp, helper, m_tail);
  }
};

template <template <class GCT> class ProAT, template <class GCT> class ProBT, template <class GCT> class EpiT,
          class GemmCoreT>
class LauncherWOQ {
 public:
  using GemmCore = GemmCoreT;
  using PrologueA = ProAT<GemmCore>;
  using PrologueB = ProBT<GemmCore>;
  using Epilogue = EpiT<GemmCore>;
  using AType = typename GemmCore::TA;
  using AParam = typename PrologueA::Param;
  using BType = typename GemmCore::TB;
  using BParam = typename PrologueB::Param;
  using CType = typename GemmCore::TC;
  using ACCType = typename GemmCore::TACC;
  using EpiParam = typename Epilogue::Param;
  struct Param {
    const AParam paramA;
    const BParam paramB;
    const EpiParam paramC;
  };

  template <bool debug = false>
  static inline sycl::event compute(sycl::queue* q, int m, int n, int k, int blocksize, const Param& _param) {
    sycl::range<2> group{GemmCore::WgM, GemmCore::WgN};
    auto A = _param.paramA.A;
    auto B = _param.paramB.B;
    auto B_scale = _param.paramB.scale;
    auto C = _param.paramC.C;
    int lda = _param.paramA.lda;
    int ldb = _param.paramB.ldb;
    int ldc = _param.paramC.ldc;
    int m_pad = utils::padto(utils::updiv(m, GemmCore::TileM), GemmCore::WgM);
    sycl::range<2> problem{static_cast<size_t>(m_pad), static_cast<size_t>(n) / GemmCore::TileN};
    auto ev = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<BType, 1> slm_b(sycl::range(GemmCore::SLM_B_Size), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>(problem, group),
          [=](sycl::nd_item<2> it) [[sycl::reqd_work_group_size(
              1, GemmCore::WgM,
              GemmCore::WgN)]] [[intel::kernel_args_restrict]] [[intel::reqd_sub_group_size(GemmCore::SgSize)]] {
            sycl_utils::nd_item_helper<GemmCore> helper(it);
            if constexpr (debug) {
              compute_tile(k, blocksize, B, B_scale, ldb, slm_b, A, lda, C, ldc, it);
            } else {
              int m_tail = m - helper.sg_g_m();
              m_tail = m_tail > GemmCore::TileM ? GemmCore::TileM : m_tail;
              if (m_tail == GemmCore::TileM) {
                compute_tile(k, blocksize, B, B_scale, ldb, slm_b, A, lda, C, ldc, it);
              } else {
                compute_tail(k, blocksize, m_tail, B, B_scale, ldb, slm_b, A, lda, C, ldc, it);
              }
            }
          });
    });
    return ev;
  }

  template <typename ScaleT>
  static void compute_tile(int k, int blocksize, const uint8_t* B, const ScaleT* B_scale, int ldb,
                           const sycl::local_accessor<BType, 1>& slm_b, const AType* A, int lda, CType* C, int ldc,
                           sycl::nd_item<2>& it) {
    sycl_utils::nd_item_helper<GemmCore> helper(it);
    ACCType tmp[GemmCore::TileM * GemmCore::TileN];
    for (size_t im = 0; im < GemmCore::TileM; im++)
      for (size_t in = 0; in < GemmCore::TileN; in++) tmp[im * GemmCore::TileN + in] = ACCType(0.f);
#pragma forceinline recursive
    for (int i = 0; i < k; i += GemmCore::TileK) {
      PrologueB::getWeight({B, B_scale, ldb}, slm_b, i, blocksize, helper);
      it.barrier(sycl::access::fence_space::local_space);
      GemmCore::compute(&A[helper.item_g_m() * k + i], k, slm_b, tmp, helper);
      it.barrier(sycl::access::fence_space::local_space);
    }
#pragma forceinline recursive
    Epilogue::store({C, ldc}, tmp, helper);
  }

  template <typename ScaleT>
  static void compute_tail(int k, int blocksize, int m_tail, const uint8_t* B, const ScaleT* B_scale, int ldb,
                           const sycl::local_accessor<BType, 1>& slm_b, const AType* A, int lda, CType* C, int ldc,
                           sycl::nd_item<2>& it) {
    sycl_utils::nd_item_helper<GemmCore> helper(it);
    ACCType tmp[GemmCore::TileM * GemmCore::TileN];
    for (size_t im = 0; im < GemmCore::TileM; im++)
      for (size_t in = 0; in < GemmCore::TileN; in++) tmp[im * GemmCore::TileN + in] = ACCType(0.f);
#pragma forceinline recursive
    for (int i = 0; i < k; i += GemmCore::TileK) {
      PrologueB::getWeight({B, B_scale, ldb}, slm_b, i, blocksize, helper);
      it.barrier(sycl::access::fence_space::local_space);
      GemmCore::compute_mtail(&A[helper.item_g_m() * k + i], k, slm_b, tmp, helper, m_tail);
      it.barrier(sycl::access::fence_space::local_space);
    }
#pragma forceinline recursive
    Epilogue::store_tail({C, ldc}, tmp, helper, m_tail);
  }
};
}  // namespace sycl_wrapper
}  // namespace bestla
#endif
