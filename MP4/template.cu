#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define d 4
#define t 4
#define k 3

//@@ Define constant memory for device kernel here
// __constant__ float *deviceKernel;
__constant__ float deviceKernel[k * k * k];

// naive version
// __global__ void conv3d(float *input, float *output, const int z_size,
//                        const int y_size, const int x_size) {
//   //@@ Insert kernel code here
//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;
//   int z = blockIdx.z * blockDim.z + threadIdx.z;
//   if (x >= x_size || y >= y_size || z >= z_size) {
//     return;
//   }
//   int i = x + y * x_size + z * (y_size * x_size);

//   float v = 0;
//   int x_start = x - (k / 2);
//   int y_start = y - (k / 2);
//   int z_start = z - (k / 2);
  
//   for (int p = 0; p < k; p++) {
//     for (int q = 0; q < k; q++) {
//       for (int r = 0; r < k; r++) {
//         int x1 = x_start + p;
//         int y1 = y_start + q;
//         int z1 = z_start + r;
//         if (((x1 >= 0) && (x1 < x_size)) && \
//             ((y1 >= 0) && (y1 < y_size)) && \
//             ((z1 >= 0) && (z1 < z_size))) {
//               int s = x1 + y1 * x_size + z1 * (y_size * x_size);
//               int t = p + q * k + r * (k * k);
//               v += input[s] * deviceKernel[t];
//         }
//       }
//     }
//   }

//   output[i] = v;
// }

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float N[t][t][t];
  // shared memory index
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  // global index
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int i = x + y * x_size + z * (y_size * x_size);

  // if (x >= x_size || y >= y_size || z >= z_size || \
  //     tx >= t || ty >= t || tz >= t) {
  //   return;
  // }

  if (((x >= 0) && (x < x_size)) && \
      ((y >= 0) && (y < y_size)) && \
      ((z >= 0) && (z < z_size))) {
    N[tx][ty][tz] = input[i];
  } else {
    return;
  }

  __syncthreads();

  // // 输出索引
  // int ox = blockIdx.x * t + tx;
  // int oy = blockIdx.y * t + ty;
  // int oz = blockIdx.z * t + tz;
  
  // 范围
  int x_start = blockIdx.x * blockDim.x;
  int x_end = x_start + blockDim.x;
  int y_start = blockIdx.y * blockDim.y;
  int y_end = y_start + blockDim.y;
  int z_start = blockIdx.z * blockDim.z;
  int z_end = z_start + blockDim.z;

  // 输入起始索引
  int ix = x - (k / 2);
  int iy = y - (k / 2);
  int iz = z - (k / 2);

  // if (ox >= x_size || oy >= y_size || oz >= z_size || \
  //     tx >= t || ty >= t || tz >= t) {
  //   return;
  // }
  // int i = ox + oy * x_size + oz * (y_size * x_size);

  float v = 0;
  
  for (int p = 0; p < k; p++) {
    for (int q = 0; q < k; q++) {
      for (int r = 0; r < k; r++) {
        int x1 = ix + p;
        int y1 = iy + q;
        int z1 = iz + r;
        if (((x1 >= 0) && (x1 < x_size)) && \
            ((y1 >= 0) && (y1 < y_size)) && \
            ((z1 >= 0) && (z1 < z_size))) {
            
            int j1 = p + q * k + r * (k * k);
            if (((x1 >= x_start) && (x1 < x_end)) && \
                ((y1 >= y_start) && (y1 < y_end)) && \
                ((z1 >= z_start) && (z1 < z_end))) {
              // tx -> x, 所以需要减去偏置项
              v += N[tx + p - (k / 2)][ty + q - (k / 2)][tz + r - (k / 2)] * deviceKernel[j1];
            } else {
              int i1 = x1 + y1 * x_size + z1 * (y_size * x_size);
              v += input[i1] * deviceKernel[j1];
            }
        }
      }
    }
  }

  output[i] = v;
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  // int k = 3;
  // int outputLength = (z_size - k + 1) * \
  //                    (y_size - k + 1) * \
  //                    (x_size - k + 1);
  cudaMalloc((void **)& deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)& deviceOutput, (inputLength - 3) * sizeof(float));
  // cudaMalloc((void **)& deviceKernel, kernelLength * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  // const memory
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid((x_size / d) + 1, (y_size / d) + 1, (z_size / d) + 1);
  dim3 DimBlock(d, d, d);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>> (deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
