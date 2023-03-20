// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 768 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// // method 2
// __global__ void scan(float *input, float *output, int len, float *S) {
//   //@@ Modify the body of this function to complete the functionality of
//   //@@ the scan on the device
//   //@@ You may need multiple kernel calls; write your kernels before this
//   //@@ function and call them from the host
//   __shared__ float XY[2 * BLOCK_SIZE];
//   int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
//   int tx = threadIdx.x;
  
//   if (i < len) {
//     XY[tx] = input[i];
//   } else {
//     XY[tx] = 0;
//   }
//   if (i + blockDim.x < len) {
//     XY[tx + blockDim.x] = input[i + blockDim.x];
//   } else {
//     XY[tx + blockDim.x] = 0;
//   }

//   int stride = 1;
//   while (stride < 2 * BLOCK_SIZE) {
//     __syncthreads();
//     int index = (tx + 1) * stride * 2 - 1;
//     if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
//       XY[index] += XY[index - stride];
//     }
//     stride *= 2;
//   }

//   stride = BLOCK_SIZE / 2;
//   while (stride > 0) {
//     __syncthreads();
//     int index = (tx + 1) * stride * 2 - 1;
//     if ((index + stride) < 2 * BLOCK_SIZE) {
//       XY[index + stride] += XY[index];
//     }
//     stride /= 2;
//   }

//   __syncthreads();
//   if (i < len) {
//     output[i] = XY[tx];
//   }
//   if (i + blockDim.x < len) {
//     output[i + blockDim.x] = XY[tx + blockDim.x];
//   }

//   __syncthreads();
//   if (S != NULL && threadIdx.x == blockDim.x - 1) {
//     S[blockIdx.x] = XY[2 * blockDim.x - 1];
//   }
// }

// __global__ void add(float *input, float *S, int len) {
//   //@@ Modify the body of this function to complete the functionality of
//   //@@ the scan on the device
//   //@@ You may need multiple kernel calls; write your kernels before this
//   //@@ function and call them from the host
//   if (blockIdx.x == 0) {
//     return;
//   }
//   int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < len) {
//     input[i] += S[blockIdx.x - 1];
//   }
//   if (i + blockDim.x < len) {
//     input[i + blockDim.x] += S[blockIdx.x - 1];
//   }
// }
// // method 2

__global__
void scan(float *input, float *output, int len, float *S){
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[BLOCK_SIZE];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x;

  if (i < len) {
    XY[tx] = input[i];
  } else {
    XY[tx] = 0;
  }

  for (int stride = 1; stride < blockDim.x; stride <<= 1){
      __syncthreads();
      if (tx >= stride) {
        XY[tx] += XY[tx - stride];
      }
  }

  __syncthreads();
  if (i < len) {
    output[i] = XY[tx];
  }

  __syncthreads();
  if (S != NULL && tx == blockDim.x - 1) {
    S[blockIdx.x] = XY[tx];
  }
}

__global__ void add(float *input, float *S, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  int i = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
  if (i < len) {
    input[i] += S[blockIdx.x];
  }
}



int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  // float *BlockSum;
  // float *BlockScanSum;
  float *S;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  // add
  // wbCheck(cudaMalloc((void **)&BlockSum, numElements * sizeof(float)));
  // int d = (numElements / BLOCK_SIZE) + 1;
  // printf("%d\n", d);
  // wbCheck(cudaMalloc((void **)&BlockScanSum, d * sizeof(float)));
  int blocks = (numElements / BLOCK_SIZE) + 1;
  // int blocks = (numElements / (2 * BLOCK_SIZE)) + 1;
  printf("%d\n", blocks);
  wbCheck(cudaMalloc((void **)&S, blocks * sizeof(float)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(blocks, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  // dim3 DimGrid2(1, 1, 1);
  // dim3 DimBlock2(numElements, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>> (deviceInput, deviceOutput, numElements, S);
  cudaDeviceSynchronize();
  scan<<<1, DimGrid>>> (S, S, blocks, NULL);
  cudaDeviceSynchronize();
  // if (d > 0) {
  //   printf("%lf %lf\n", BlockScanSum[0], BlockScanSum[1]);
  // }
  // for (int i = 0; i < blocks; i++) {
  //   wbLog(TRACE, 1, " ");
  // }
  // wbLog(TRACE, "\n");
  
  
  // scan<<<1, DimBlock>>> (auxiliaryBlock, deviceOutput, numElements);
  // add<<<DimGrid, DimBlock>>> (BlockSum, BlockScanSum, deviceOutput, numElements);
  add<<<DimGrid, DimBlock>>> (deviceOutput, S, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  // for (int i = 0; i < numElements; i++) {
  //   printf("%lf\n", hostOutput[i]);
  // }
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(S);
  // cudaFree(BlockSum);
  // cudaFree(BlockScanSum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
