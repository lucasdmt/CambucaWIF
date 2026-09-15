#include "GPUSecp.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPUMath.h"
#include "GPUHash.h"

using namespace std;

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
  if (cudaSuccess != err)
  {
    printf("cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

GPUSecp::GPUSecp(
    const uint8_t *gTableXCPU,
    const uint8_t *gTableYCPU,
    const uint64_t *inputHashBufferCPU,
    const char* WIFprivKeyCPU, //lucas
    const int* posicoesCPU, //lucas
    int totalPosicoesCPUtemp  //lucas
    )
{
  printf("GPUSecp Starting\n");

  int gpuId = 0; // FOR MULTIPLE GPUS EDIT THIS
  CudaSafeCall(cudaSetDevice(gpuId));

  cudaDeviceProp deviceProp;
  CudaSafeCall(cudaGetDeviceProperties(&deviceProp, gpuId));

  printf("GPU.gpuId: #%d ", gpuId);
  printf(" %s ", deviceProp.name);
  printf("GPU.multiProcessorCount:%d ", deviceProp.multiProcessorCount);
  printf("GPU.BLOCKS_PER_GRID:%d ", BLOCKS_PER_GRID);
  printf("GPU.THREADS_PER_BLOCK:%d ", THREADS_PER_BLOCK);
  printf("GPU.CUDA_THREAD_COUNT:%d ", COUNT_CUDA_THREADS);
  printf("GPU.countHash160:%d \n", COUNT_INPUT_HASH);


  CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  CudaSafeCall(cudaDeviceSetLimit(cudaLimitStackSize, SIZE_CUDA_STACK));

  size_t limit = 0;
  cudaDeviceGetLimit(&limit, cudaLimitStackSize);
  printf("cudaLimitStackSize:%u ", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
  printf("cudaLimitPrintfFifoSize:%u ", (unsigned)limit);
  cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
  printf("cudaLimitMallocHeapSize:%u \n", (unsigned)limit);
  
  printf("Allocating: ");
  printf("inputHashBuffer ");
  CudaSafeCall(cudaMalloc((void **)&inputHashBufferGPU, COUNT_INPUT_HASH * SIZE_LONG));
  CudaSafeCall(cudaMemcpy(inputHashBufferGPU, inputHashBufferCPU, COUNT_INPUT_HASH * SIZE_LONG, cudaMemcpyHostToDevice));

  printf("gTableX ");
  CudaSafeCall(cudaMalloc((void **)&gTableXGPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemset(gTableXGPU, 0, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemcpy(gTableXGPU, gTableXCPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT, cudaMemcpyHostToDevice));

  printf("gTableY ");
  CudaSafeCall(cudaMalloc((void **)&gTableYGPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemset(gTableYGPU, 0, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT));
  CudaSafeCall(cudaMemcpy(gTableYGPU, gTableYCPU, COUNT_GTABLE_POINTS * SIZE_GTABLE_POINT, cudaMemcpyHostToDevice));

  printf("outputBuffer ");
  CudaSafeCall(cudaMalloc((void **)&outputBufferGPU, COUNT_CUDA_THREADS));
  CudaSafeCall(cudaHostAlloc(&outputBufferCPU, COUNT_CUDA_THREADS, cudaHostAllocWriteCombined | cudaHostAllocMapped));

  printf("outputHashes ");
  CudaSafeCall(cudaMalloc((void **)&outputHashesGPU, COUNT_CUDA_THREADS * SIZE_HASH160));
  CudaSafeCall(cudaHostAlloc(&outputHashesCPU, COUNT_CUDA_THREADS * SIZE_HASH160, cudaHostAllocWriteCombined | cudaHostAllocMapped));

  printf("outputPrivKeys ");
  CudaSafeCall(cudaMalloc((void **)&outputPrivKeysGPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY));
  CudaSafeCall(cudaHostAlloc(&outputPrivKeysCPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY, cudaHostAllocWriteCombined | cudaHostAllocMapped));


   printf("privKey WIF");
   CudaSafeCall(cudaMalloc((void**)&WIFprivKeyGPU, 53 * sizeof(char)));
   CudaSafeCall(cudaMemcpy(WIFprivKeyGPU, WIFprivKeyCPU, 53 * sizeof(char), cudaMemcpyHostToDevice));

    printf("posicoesGPU ");
    CudaSafeCall(cudaMalloc((void**)&posicoesGPU, 65 * sizeof(int)));
    CudaSafeCall(cudaMemcpy(posicoesGPU, posicoesCPU, 65 * sizeof(int), cudaMemcpyHostToDevice));

    d_totalPosicoesCPUtemp = totalPosicoesCPUtemp;

    //printf("Total posiçõesna gpu: %d\n",d_totalPosicoesCPUtemp);


  printf("\nAllocation Complete \n");
  CudaSafeCall(cudaGetLastError());
}

//Cuda Secp256k1 Point Multiplication
//Takes 32-byte privKey + gTable and outputs 64-byte public key [qx,qy]
__device__ void _PointMultiSecp256k1(uint64_t *qx, uint64_t *qy, uint16_t *privKey, uint8_t *gTableX, uint8_t *gTableY) {
//6123ae95438e22e11b4a116b4c0c3d514ecf6cfede99370cabebf4f282b4228f deve resultar em 228f 82b4 f4f2 abeb 370c de99 6cfe 4ecf 3d51 4c0c 116b 1b4a 22e1 438e ae95 6123
/*printf("funcfinal: ");
for (int i = 0; i < 16; i++) {
    printf("%04x ", privKey[i]);  // %04x = 4 dígitos hex, preenchidos com zero à esquerda
}
printf("\n");*/
    int chunk = 0;
    uint64_t qz[5] = {1, 0, 0, 0, 0};

    //Find the first non-zero point [qx,qy]
    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
      if (privKey[chunk] > 0) {
        int index = (CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;
        memcpy(qx, gTableX + index, SIZE_GTABLE_POINT);
        memcpy(qy, gTableY + index, SIZE_GTABLE_POINT);
        chunk++;
        break;
      }
    }

    //Add the remaining chunks together
    for (; chunk < NUM_GTABLE_CHUNK; chunk++) {
      if (privKey[chunk] > 0) {
        uint64_t gx[4];
        uint64_t gy[4];

        int index = (CHUNK_FIRST_ELEMENT[chunk] + (privKey[chunk] - 1)) * SIZE_GTABLE_POINT;
        
        memcpy(gx, gTableX + index, SIZE_GTABLE_POINT);
        memcpy(gy, gTableY + index, SIZE_GTABLE_POINT);

        _PointAddSecp256k1(qx, qy, qz, gx, gy);
      }
    }

    //Performing modular inverse on qz to obtain the public key [qx,qy]
    _ModInv(qz);
    _ModMult(qx, qz);
    _ModMult(qy, qz);
}

__constant__ char CHARSET_GPU[59] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

//GPU kernel function for computing Secp256k1 public key from input books
__global__ void
CudaRunSecp256k1Books(
    int iteration,
    uint8_t * gTableXGPU,
    uint8_t * gTableYGPU,
    uint64_t *inputHashBufferGPU,
    uint8_t *outputBufferGPU,
    uint8_t *outputHashesGPU,
    uint8_t *outputPrivKeysGPU,
    const char* privKey,          //lucas
    const int* posicoes,      //lucas
    int totalPosicoes    //lucas
){
  uint8_t privKeyLocal[32];
  memcpy(privKeyLocal, privKey, 32);

  int thread_id = IDX_CUDA_THREAD;//blockIdx.x * blockDim.x) + threadIdx.x
  //long long result2 = iteration * COUNT_CUDA_THREADS * THREAD_MULT;
  long long result2 = (long long)iteration * COUNT_CUDA_THREADS * THREAD_MULT;
  
  //printf("thread_id %d\n", thread_id);//ULTIMA é COUNT_CUDA_THREADS-1 e a primeira 0 errrrrr
  
  int start = thread_id * THREAD_MULT;
  int end = start + THREAD_MULT;
     long long result;

   for (int j = start; j < end; ++j){
      result=result2+j;

   //printf("privkey in %s",privKey);//debug
   //printf(" \n %lld result ",result);//debug


    int indicesCharset[52] = {0};  // Array inicializado com zeros
    int base = 58;

    // Preenchendo o array de trás para frente
    for (int i = totalPosicoes -1 ; i >= 0; i--) {
        indicesCharset[i] = result % base;  // Obtém o dígito menos significativo
        result /= base;  // Divide pelo valor da base para obter o próximo dígito
    }
   
   /*printf(" indices ");//debug
   for(int z=0; z<totalPosicoes; z++){
      printf("%d ",indicesCharset[z]);
   }*/

    char localstr[53];
    memcpy(localstr, privKey, 53);  // Copia a privkeyglobal para o buffer da thread
    for (int i = 0; i < totalPosicoes; i++) {
       localstr[posicoes[i]] = CHARSET_GPU[indicesCharset[i]];
    } 

   /*if (j == end - 1 && threadIdx.x == blockDim.x - 1 && blockIdx.x == gridDim.x - 1){  
      printf("privkey in %s \n",localstr);//debug
   }*/

        uint8_t privKeyBytes[32];
    if (_WIFToPrivKey(localstr, privKeyBytes)) {

       /*printf("[DEBUG] privKeyBytes: ");
       for (int i = 0; i < 32; i++) printf("%02x", privKeyBytes[i]);
       printf("\n");*/

        // Monta o uint16_t[16] invertido que _PointMultiSecp256k1 espera.
        // privKeyBytes[0] é o byte mais significativo da chave (big-endian),
        // e privKey[15] deve ser o chunk menos significativo — por isso o índice invertido.
        uint16_t privKey[16];
        for (int i = 0; i < 16; i++) {
            // Cada uint16_t agrupa 2 bytes consecutivos
            uint8_t hi = privKeyBytes[i * 2];
            uint8_t lo = privKeyBytes[i * 2 + 1];
            privKey[15 - i] = ((uint16_t)hi << 8) | lo;
        }

        uint64_t qx[4];
        uint64_t qy[4];
        _PointMultiSecp256k1(qx, qy, (uint16_t *)privKey, gTableXGPU, gTableYGPU);

        uint8_t hash160[SIZE_HASH160];
        uint64_t hash160Last8Bytes;
        _GetHash160Comp(qx, (uint8_t)(qy[0] & 1), hash160);
        GET_HASH_LAST_8_BYTES(hash160Last8Bytes, hash160);

        if (_BinarySearch(inputHashBufferGPU, COUNT_INPUT_HASH, hash160Last8Bytes) >= 0) {

            printf("possivel chave encontrada!: %s\n", localstr);

            int idxCudaThread = IDX_CUDA_THREAD;
            outputBufferGPU[idxCudaThread] += 1;
            for (int i = 0; i < SIZE_HASH160; i++) {
                outputHashesGPU[(idxCudaThread * SIZE_HASH160) + i] = hash160[i];
            }
            for (int i = 0; i < SIZE_PRIV_KEY; i++) {
                outputPrivKeysGPU[(idxCudaThread * SIZE_PRIV_KEY) + i] = privKey[i];
            }
        }

    } else {
        // WIF inválido (checksum errado, prefixo errado, etc.)
        // _WIFToPrivKey já imprime o motivo via printf de debug
    }

   }
}



void GPUSecp::doIterationSecp256k1Books(int iteration) {
  CudaSafeCall(cudaMemset(outputBufferGPU, 0, COUNT_CUDA_THREADS));
  CudaSafeCall(cudaMemset(outputHashesGPU, 0, COUNT_CUDA_THREADS * SIZE_HASH160));
  CudaSafeCall(cudaMemset(outputPrivKeysGPU, 0, COUNT_CUDA_THREADS * SIZE_PRIV_KEY));

  CudaRunSecp256k1Books<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
    iteration,
    gTableXGPU,
    gTableYGPU,
    inputHashBufferGPU,
    outputBufferGPU,
    outputHashesGPU,
    outputPrivKeysGPU,
    WIFprivKeyGPU,
    posicoesGPU,
    d_totalPosicoesCPUtemp //lucas
    );

  CudaSafeCall(cudaMemcpy(outputBufferCPU, outputBufferGPU, COUNT_CUDA_THREADS, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(outputHashesCPU, outputHashesGPU, COUNT_CUDA_THREADS * SIZE_HASH160, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaMemcpy(outputPrivKeysCPU, outputPrivKeysGPU, COUNT_CUDA_THREADS * SIZE_PRIV_KEY, cudaMemcpyDeviceToHost));
  CudaSafeCall(cudaGetLastError());
}

void GPUSecp::doPrintOutput() {
  for (int idxThread = 0; idxThread < COUNT_CUDA_THREADS; idxThread++) {
    if (outputBufferCPU[idxThread] > 0) {
      printf("HASH: ");
      for (int h = 0; h < SIZE_HASH160; h++) {
        printf("%02X", outputHashesCPU[(idxThread * SIZE_HASH160) + h]);
      }
      printf(" PRIV: ");
      for (int k = SIZE_PRIV_KEY -1 ; k >= 0; k--) {
        printf("%02X", outputPrivKeysCPU[(idxThread * SIZE_PRIV_KEY) + k]);
      }
      printf("\n");

      FILE *file = stdout;
      file = fopen(NAME_FILE_OUTPUT, "a");
      if (file != NULL) {
        fprintf(file, "HASH: ");
        for (int h = 0; h < SIZE_HASH160; h++) {
          fprintf(file, "%02X", outputHashesCPU[(idxThread * SIZE_HASH160) + h]);
        }
        fprintf(file, " PRIV: ");
        for (int k = SIZE_PRIV_KEY -1 ; k >= 0; k--) {
          fprintf(file, "%02X", outputPrivKeysCPU[(idxThread * SIZE_PRIV_KEY) + k]);
        }
        fprintf(file, "\n");
        fclose(file);
      }
    }
  }
}

void GPUSecp::updatemodohex(
    const int* posicoesCPU,
    int totalPosicoesCPU
)
{
    // se tamanho mudou, realoca
    if (totalPosicoesCPU != d_totalPosicoesCPUtemp)
    {
        if (posicoesGPU != nullptr)
            CudaSafeCall(cudaFree(posicoesGPU));

        CudaSafeCall(cudaMalloc(
            (void**)&posicoesGPU,
            totalPosicoesCPU * sizeof(int)
        ));

        d_totalPosicoesCPUtemp = totalPosicoesCPU;
    }

    // copia CPU → GPU
    CudaSafeCall(cudaMemcpy(
        posicoesGPU,
        posicoesCPU,
        totalPosicoesCPU * sizeof(int),
        cudaMemcpyHostToDevice
    ));
}

void GPUSecp::doFreeMemory() {
  printf("\nGPUSecp Freeing memory... ");

  CudaSafeCall(cudaFree(inputHashBufferGPU));

  CudaSafeCall(cudaFree(gTableXGPU));
  CudaSafeCall(cudaFree(gTableYGPU));

  CudaSafeCall(cudaFreeHost(outputBufferCPU));
  CudaSafeCall(cudaFree(outputBufferGPU));

  CudaSafeCall(cudaFreeHost(outputHashesCPU));
  CudaSafeCall(cudaFree(outputHashesGPU));

  CudaSafeCall(cudaFreeHost(outputPrivKeysCPU));
  CudaSafeCall(cudaFree(outputPrivKeysGPU));

  CudaSafeCall(cudaFree(WIFprivKeyGPU));
  CudaSafeCall(cudaFree(posicoesGPU));

  printf("Acabou \n");
}
