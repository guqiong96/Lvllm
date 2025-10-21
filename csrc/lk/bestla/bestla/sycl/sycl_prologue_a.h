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
namespace sycl_prologue_a {

template <typename SrcT>
struct ParamActivationBase {
  const SrcT* A;
  int lda;
};
template <class GemmCoreT, typename SrcT>
class ActivationBase {
 public:
  using AType = typename GemmCoreT::TA;
  using SrcType = SrcT;
  using Param = ParamActivationBase<SrcType>;
  static inline void getActivation(const Param& _param, AType* aptr, sycl_utils::nd_item_helper<GemmCoreT>& helper) {}
};

}  // namespace sycl_prologue_a
}  // namespace bestla
#endif
