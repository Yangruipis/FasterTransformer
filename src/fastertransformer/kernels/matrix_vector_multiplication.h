/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "int8_utils.cuh"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

template<typename T>
void int8WeightPerChannelLdkMultiplicationLauncher(const int8_t* weight,
                                                   const T*      input,
                                                   const T*      scale_list,
                                                   T*            output,
                                                   const int     m,
                                                   const int     n,
                                                   const int     k,
                                                   cudaStream_t  stream);

template<typename T>
void int4WeightPerChannelLdkMultiplicationLauncher(const int8_t* weight,
                                                   const T*      input,
                                                   const T*      scale_list,
                                                   T*            output,
                                                   const int     m,
                                                   const int     n,
                                                   const int     k,
                                                   cudaStream_t  stream);

template<typename T>
void invokeInt4WeightExtractionTrans(
    const int8_t* weight, const T* scale_list, T* output, const int n, const int k, cudaStream_t stream);

template<typename T>
void invokeInt4WeightExtractionNoTrans(
    const int8_t* weight, const T* scale_list, T* output, const int n, const int k, cudaStream_t stream);

template<typename T>
void invokeInt4WeightExtractionNoTrans2(
    const int8_t* weight, const T* scale_list, T* output, const int n, const int k, cudaStream_t stream);

template<typename T>
void invokeInt8WeightExtractionTrans(
    const int8_t* weight, const T* scale_list, T* output, const int n, const int k, cudaStream_t stream);

template<typename T>
void invokeInt8WeightExtractionNoTrans(
    const int8_t* weight, const T* scale_list, T* output, const int n, const int k, cudaStream_t stream);

template<typename T>
void invokeInt8diffNoTrans(
    const int8_t* weight, const T* scale_list, const T* output, const int n, const int k, cudaStream_t stream);

}  // namespace fastertransformer
