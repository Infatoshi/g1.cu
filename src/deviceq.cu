#include <cstdio>
#include <cuda_runtime.h>
int main(){
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("device            : %s  (sm_%d%d)\n", p.name, p.major, p.minor);
  printf("multiProcessorCount: %d SMs\n", p.multiProcessorCount);
  printf("regsPerMultiprocessor: %d\n", p.regsPerMultiprocessor);
  printf("regsPerBlock      : %d\n", p.regsPerBlock);
  printf("maxThreadsPerMultiProcessor: %d  (= %d warps)\n", p.maxThreadsPerMultiProcessor, p.maxThreadsPerMultiProcessor/p.warpSize);
  printf("maxBlocksPerMultiProcessor : %d\n", p.maxBlocksPerMultiProcessor);
  printf("maxThreadsPerBlock: %d\n", p.maxThreadsPerBlock);
  printf("sharedMemPerMultiprocessor : %zu B\n", p.sharedMemPerMultiprocessor);
  printf("sharedMemPerBlock : %zu B\n", p.sharedMemPerBlock);
  printf("warpSize          : %d\n", p.warpSize);
  // hand-math check: regs/SM divided by (255 regs * 64 threads), with 256-granularity
  int gran=256; int regs_per_thread=255; int rounded=((regs_per_thread+gran-1)/gran)*gran;
  printf("\n255 regs rounded to granularity %d = %d; blocks/SM by regs = %d / (%d*64) = %d\n",
         gran, rounded, p.regsPerMultiprocessor, rounded, p.regsPerMultiprocessor/(rounded*64));
  return 0;
}
