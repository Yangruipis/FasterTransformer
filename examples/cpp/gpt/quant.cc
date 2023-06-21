/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "src/fastertransformer/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "3rdparty/INIReader.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include "src/fastertransformer/utils/nvtx_utils.h"
#include <cuda_profiler_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <vector>

using namespace fastertransformer;

template<typename T>
std::vector<T> loadWeightFromBinHelper(std::vector<size_t> shape, std::string filename)
{
    if (shape.size() > 2) {
        printf("[ERROR] shape should have less than two dims \n");
        return std::vector<T>();
    }
    size_t dim0 = shape[0], dim1 = 1;
    if (shape.size() == 2) {
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;
    if (size == 0) {
        return std::vector<T>();
    }

    std::vector<T> host_array(size);
    std::ifstream  in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        return std::vector<T>();
    }

    size_t loaded_data_size = sizeof(T) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    in.read((char*)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        return std::vector<T>();
    }
    in.close();
    // If we succeed, return an array with values.
    return host_array;
}

int main(int argc, char* argv[])
{
    std::string filename1 =
        "/data/git_repos/models/bloom-560m/c-model-int4/1-gpu/model.layers.0.attention.query_key_value.weight.0.bin";

    std::vector<size_t> shape1{6144, 1024};

    std::vector<int8_t>   host_array1 = loadWeightFromBinHelper<int8_t>(shape1, filename1);

    printf("%d: %d\n", 0, host_array1[0]);
    printf("%d: %d\n", 1000, host_array1[1000]);
    printf("%d: %d\n", 2000, host_array1[2000]);


    std::string filename2 =
      "/data/git_repos/models/bloom-560m/c-model-int4/1-gpu/model.layers.0.attention.query_key_value.scale.0.bin";

    std::vector<size_t> shape2{6144, 1};

    std::vector<half>   host_array2 = loadWeightFromBinHelper<half>(shape2, filename2);

    printf("%d: %5f\n", 0, float(host_array2[0]));


    std::string filename =
        "/data/git_repos/models/bloom-560m/c-model-fp16/1-gpu/model.layers.0.attention.query_key_value.weight.0.bin";

    std::vector<size_t> shape{2048, 6144};

    std::vector<half>   host_array = loadWeightFromBinHelper<half>(shape, filename);



    const size_t        num_elts = shape[0] * shape[1];
    std::vector<int8_t> host_quantized_weight_buf(num_elts);
    std::vector<int8_t> host_quantized_weight_int4_buf(num_elts / 2);
    std::vector<float>      host_scales_buf(shape[1]);

    std::vector<half> host_array_trans(num_elts);

    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        host_array_trans[j * shape[0] + i] = host_array[i * shape[1] + j];
      }
    }

    const half*         current_weight           = host_array_trans.data();
    int8_t*             current_quantized_weight = host_quantized_weight_int4_buf.data();
    const float         quant_range_scale        = 1.f / float(1 << (4 - 1));

    const size_t num_rows = shape[1];
    const size_t num_cols = shape[0];

    const int bytes_per_out_col = num_cols / 2;

    std::vector<float> per_col_max(num_rows);
    for (int jj = 0; jj < num_rows; ++jj) {
      per_col_max[jj] = 0.f;
    }

    for (int ii = 0; ii < num_rows; ++ii) {
      const half* current_weight_row = current_weight + ii * num_cols;
      for (int jj = 0; jj < num_cols; ++jj) {
        per_col_max[ii] = std::max(per_col_max[ii], std::abs(float(current_weight_row[jj])));
      }
    }


    for (int jj = 0; jj < num_rows; ++jj) {
      per_col_max[jj] *= quant_range_scale;
      host_scales_buf[jj] = per_col_max[jj];
    }

    for (int ii = 0; ii < num_rows; ++ii) {
      int8_t*     current_quantized_weight_row = current_quantized_weight + ii * bytes_per_out_col;
      const half* current_weight_row           = current_weight + ii * num_cols;
      for (int jj = 0; jj < bytes_per_out_col; ++jj) {

        // We will pack two int4 elements per iteration of the inner loop.
        int8_t packed_int4s = 0;
        for (int packed_idx = 0; packed_idx < 2; ++packed_idx) {
          const int input_idx = 2 * jj + packed_idx;
          if (input_idx < num_cols) {
            const float  col_scale      = per_col_max[ii];
            const float  weight_elt     = float(current_weight_row[input_idx]);
            const float  scaled_weight  = round(weight_elt / col_scale);
            int          int_weight     = int(scaled_weight);
            const int8_t clipped_weight = std::max(-8, std::min(7, int_weight));
            packed_int4s |= ((clipped_weight & 0x0F) << (4 * packed_idx));
            if (ii == 0 && jj == 0)
              printf(" %d, %5f\n", clipped_weight, col_scale);
          }
        }
        current_quantized_weight_row[jj] = packed_int4s;
      }
    }
    printf("%d: %d\n", 0, host_quantized_weight_int4_buf[0]);
    printf("%d: %d\n", 1000, host_quantized_weight_int4_buf[1000]);
    printf("%d: %d\n", 2000, host_quantized_weight_int4_buf[2000]);

    for (int i = 0; i < 100000; i++) {
      if (host_array1[i] != host_quantized_weight_int4_buf[i]) {
        printf("%d: %d vs %d\n", i, host_array1[i], host_quantized_weight_int4_buf[i]);
      }
    }


    // const size_t        num_elts = shape[0] * shape[1];
    // std::vector<int8_t> host_quantized_weight_buf(num_elts);
    // std::vector<float>      host_scales_buf(shape[1]);

    // symmetric_quantize<float, float>(
    //     host_quantized_weight_buf.data(), host_scales_buf.data(), host_array.data(), shape, QuantType::PACKED_INT4_WEIGHT_ONLY);

    // for (int i = 0; i < 70; ++i) {
    //     int8_t original = host_quantized_weight_buf[i];
    //     int8_t high     = original >> 4;
    //     int8_t low      = original << 4;
    //     low             = low >> 4;

    //   printf("%d %d * %5f -> %5f vs %5f\n", i, low,
    //          host_scales_buf[(2*i)%shape[1]], host_array[2*i],
    //          host_scales_buf[(2*i)%shape[1]] * float(low));
    //   printf("%d %d * %5f -> %5f vs %5f\n", i, high,
    //          host_scales_buf[(2*i+1)%shape[1]], host_array[2*i+1],
    //          host_scales_buf[(2*i+1)%shape[1]] * float(high));

    // }


    // const size_t        num_elts   = shape[0] * shape[1];
    // std::vector<int8_t> host_quantized_weight_buf(num_elts);
    // std::vector<float>   host_scales_buf(shape[1]);

    // for (int i = 0; i < num_elts; i++) {
    //     printf("raw %5f\n", double(host_array[i]));
    // }

    // const float*         current_weight           = host_array.data();
    // int8_t*             current_quantized_weight = host_quantized_weight_buf.data();
    // const float         quant_range_scale        = 1.f / float(1 << (8 - 1));

    // const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    // const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    // const int bytes_per_out_col = num_cols / 1;

    // std::vector<float> per_col_max(num_cols);
    // for (int jj = 0; jj < num_cols; ++jj) {
    //     per_col_max[jj] = 0.f;
    // }
    // for (int ii = 0; ii < num_rows; ++ii) {
    //     const float* current_weight_row = current_weight + ii * num_cols;
    //     for (int jj = 0; jj < num_cols; ++jj) {
    //         per_col_max[jj] = std::max(per_col_max[jj], std::abs(float(current_weight_row[jj])));
    //     }
    // }

    // for (int jj = 0; jj < num_cols; ++jj) {
    //     per_col_max[jj] *= quant_range_scale;
    //     host_scales_buf[jj] = float(per_col_max[jj]);
    //     // printf("scale %5f\n", host_scales_buf[jj]);
    // }

    // for (int ii = 0; ii < num_rows; ++ii) {
    //     int8_t*     current_quantized_weight_row = current_quantized_weight + ii * bytes_per_out_col;
    //     const float* current_weight_row           = current_weight + ii * num_cols;
    //     for (int jj = 0; jj < bytes_per_out_col; ++jj) {

    //       const float  col_scale           = per_col_max[jj];
    //       const float  weight_elt          = float(current_weight_row[jj]);
    //       const float  scaled_weight       = round(weight_elt / col_scale);
    //       const int8_t clipped_weight      = int8_t(std::max(-128.f, std::min(127.f, scaled_weight)));
    //       current_quantized_weight_row[jj] = clipped_weight;


    //         // // We will pack two int4 elements per iteration of the inner loop.
    //         // int8_t packed_int4s = 0;
    //         // for (int packed_idx = 0; packed_idx < 2; ++packed_idx) {
    //         //     const int input_idx = 2 * jj + packed_idx;
    //         //     if (input_idx < num_cols) {
    //         //         const float  col_scale      = per_col_max[input_idx];
    //         //         const float  weight_elt     = float(current_weight_row[input_idx]);
    //         //         const float  scaled_weight  = round(weight_elt / col_scale);
    //         //         int          int_weight     = int(scaled_weight);
    //         //         const int8_t clipped_weight = std::max(-8, std::min(7, int_weight));

    //         //         // Kill the sign extension bits (hence 0x0F mask) then shift to upper bits
    //         //         // if packing the second int4 and or the bits into the final result.
    //         //         // if (packed_idx == 0) {
    //         //         //     packed_int4s = clipped_weight << 4;
    //         //         // }
    //         //         // else {
    //         //         //     packed_int4s = packed_int4s | (clipped_weight & 0b00001111);
    //         //         // }
    //         //         packed_int4s |= ((clipped_weight & 0x0F) << (4 * packed_idx));
    //         //         printf("int4 %d\n", clipped_weight);
    //         //     }
    //         // }
    //         // current_quantized_weight_row[jj] = packed_int4s;
    //         // // printf("i=%d, v=%d\n", jj, packed_int4s);
    //         // printf("int8 %d\n", packed_int4s);
    //     }
    // }

    // for (int i = 0; i < 70; ++i) {
    //   printf("%d %d * %5f -> %5f vs %5f\n", i, host_quantized_weight_buf[i], host_scales_buf[i%shape[1]], host_array[i], host_scales_buf[i%shape[1]] * float(host_quantized_weight_buf[i]));
    // }

    // // for (int jj = 0; jj < host_scales_buf.size(); ++jj) {
    // //     printf("scale %5f\n", host_scales_buf[jj]);
    // // }

    // // for (int jj = 0; jj < host_quantized_weight_int4_buf.size(); ++jj) {
    // //     printf("int4 %d\n", host_quantized_weight_int4_buf[jj]);
    // // }

    // // for (int jj = 0; jj < host_quantized_weight_int4_buf.size(); ++jj) {

    // //     int8_t original = host_quantized_weight_int4_buf[jj];
    // //     int8_t high     = original >> 4;
    // //     int8_t low      = original << 4;
    // //     low             = low >> 4;
    // //     printf("ori=%d, low=%d, high=%d, scale=%5f\n", original, low, high, host_scales_buf[jj % shape[1]]);
    // //     printf("low=%5f, high=%5f\n", low * host_scales_buf[jj*2 % shape[1]], high * host_scales_buf[(jj*2+1) % shape[1] ]);
    // // }

    return 0;
}
