
#include <cuda_fp16.h>
#include <math.h>

template<typename T>
__device__ void
int4WeightExtractionDevice(const int8_t* weight,
                                const T* scale_list,
                                T* output,
                                const int n,
                                const int k)
{
    for(int i = blockIdx.x * k + threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x){
        int8_t original = weight[i];
        int8_t high = original >> 4;
        int8_t low = original << 4; low = low >> 4;
        output[i * 2] = T(high) * scale_list[blockIdx.x];
        output[i * 2 + 1] = T(low) * scale_list[blockIdx.x];
    }
}

__device__ void
int4WeightCompressionDevice(const int8_t* input,
                                int8_t* output,
                                const int n,
                                const int k)
{
    for(int i = blockIdx.x * k + threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x){
      int8_t packed_int4s = 0;
      for (int packed_idx = 0; packed_idx < 2; ++packed_idx) {
        const int input_idx = 2 * i + packed_idx;
        int int_weight = int(input[input_idx]);
        const int8_t clipped_weight = max(-8, min(7, int_weight));
        packed_int4s |= ((clipped_weight & 0x0F) << (4 * packed_idx));
      }
      output[i] = packed_int4s;
      // output[i] = (input[i * 2+1] << 4) | (input[i * 2] & 0b00001111);
    }
}

template<typename T>
__device__ void
int8WeightExtractionDevice(const int8_t* weight,
                                const T* scale_list,
                                T* output,
                                const int n,
                                const int k)
{
    for(int i = blockIdx.x * k + threadIdx.x; i < blockIdx.x * k + k; i += blockDim.x){
        output[i] = T(weight[i]) * scale_list[blockIdx.x];
    }
}

extern "C" __global__ void int4WeightExtractionHalf(const int8_t* weight,
                                const half* scale_list,
                                half* output,
                                const int n,
                                const int k){
                                    int4WeightExtractionDevice<half>(weight, scale_list, output, n, k);
                                }

extern "C" __global__ void int4WeightExtractionFloat(const int8_t* weight,
                                const float* scale_list,
                                float* output,
                                const int n,
                                const int k){
                                    int4WeightExtractionDevice<float>(weight, scale_list, output, n, k);
                                }

extern "C" __global__ void int8WeightExtractionHalf(const int8_t* weight,
                                const half* scale_list,
                                half* output,
                                const int n,
                                const int k){
                                    int8WeightExtractionDevice<half>(weight, scale_list, output, n, k);
                                }

extern "C" __global__ void int8WeightExtractionFloat(const int8_t* weight,
                                const float* scale_list,
                                float* output,
                                const int n,
                                const int k){
                                    int8WeightExtractionDevice<float>(weight, scale_list, output, n, k);
                                }

extern "C" __global__ void int4WeightCompression(const int8_t* input,
                                int8_t* output,
                                const int n,
                                const int k){
                                    int4WeightCompressionDevice(input, output, n, k);
                                }
