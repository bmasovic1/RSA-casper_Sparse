#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>
#include <numeric>
#include <windows.h>

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(call) \
do { cudaError_t err = call; if(err != cudaSuccess){ \
  printf("CUDA greška %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); exit(1); }} while(0)

#define CUSPARSE_CHECK(call) \
do { cusparseStatus_t status = call; if(status != CUSPARSE_STATUS_SUCCESS){ \
  printf("cuSPARSE greška %s:%d: %d\n",__FILE__,__LINE__,(int)status); exit(1); }} while(0)

static void enable_utf8_console() {
  std::setlocale(LC_ALL, ".UTF8");
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
  setvbuf(stdout, nullptr, _IOFBF, 1 << 20);
}

static std::string bytes_to_human(size_t b) {
  const char* suf[] = {"B","KB","MB","GB","TB"};
  double x = (double)b;
  int i = 0;
  while (x >= 1024.0 && i < 4) { x /= 1024.0; i++; }
  std::ostringstream os;
  os << std::fixed << std::setprecision(2) << x << " " << suf[i];
  return os.str();
}

struct GpuTimer {
  cudaEvent_t s{}, e{};
  GpuTimer() { CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e)); }
  ~GpuTimer(){ cudaEventDestroy(s); cudaEventDestroy(e); }
  void start(cudaStream_t st = 0) { CUDA_CHECK(cudaEventRecord(s, st)); }
  float stop(cudaStream_t st = 0) {
    CUDA_CHECK(cudaEventRecord(e, st));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    return ms;
  }
};

__device__ inline unsigned int lcg_random(unsigned int seed) {
    return seed * 1664525u + 1013904223u;
}

__device__ inline float lcg_random_float(unsigned int seed) {
    return (float)(lcg_random(seed) & 0x00FFFFFF) / (float)0x01000000;
}

// generator NNZ Kernel (Generiše random broj NNZ po redu)
__global__ void generator_matrica_row_ptr_random_kernel(
    int* row_ptr, 
    int M, 
    int K,
    float sparsity_percent,
    unsigned int seed) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0) {
        row_ptr[0] = 0;
    }
    
    if (idx > 0 && idx <= M) {
        unsigned int rng_state = seed + idx;
        rng_state = lcg_random(rng_state);
        
        int max_nnz = (int)(K * sparsity_percent / 100.0f);
        int min_nnz = max(1, (int)(max_nnz * 0.5f));
        
        int nnz_for_row = min_nnz + (int)(lcg_random_float(rng_state) * (max_nnz - min_nnz + 1));
        nnz_for_row = min(nnz_for_row, K);
        
        row_ptr[idx] = nnz_for_row;
    }
}

// pretvara random NNZ u CSR format Kernel
__global__ void prefix_sum_kernel(int* row_ptr, int M) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 1; i <= M; i++) {
            row_ptr[i] += row_ptr[i-1];
        }
    }
}

//Generiše sparse matricu A (CSR) kernel
__global__ void generator_matrica_sparse_kernel(
    int* col_idx, 
    float* val, 
    const int* row_ptr,
    int M, 
    int K, 
    int NNZ,
    unsigned int seed)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    
    int start = row_ptr[row];
    int end = row_ptr[row + 1];
    int nnz_in_row = end - start;
    
    unsigned int rng_state = seed + row * 12345;
    
    for (int i = 0; i < nnz_in_row; i++) {
        rng_state = lcg_random(rng_state);
        int col = (int)(lcg_random_float(rng_state) * K) % K;
        
        bool unique = true;
        for (int j = 0; j < i; j++) {
            if (col_idx[start + j] == col) {
                unique = false;
                break;
            }
        }
        
        if (!unique) {
            i--;
            continue;
        }
        
        col_idx[start + i] = col;
        
        rng_state = lcg_random(rng_state);
        val[start + i] = lcg_random_float(rng_state);
    }
}
//generator dense kernel
__global__ void generator_matrica_dense_kernel(
    float* B, 
    size_t total_elements, 
    unsigned int seed)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    
    unsigned int rng_state = seed + idx;
    B[idx] = lcg_random_float(rng_state);
}

/*Mnozenje Sparse x Dense
    Za svaki NENULTI A[i][k]: 
    Uzmi CIJELI red B[k][:]
    Ažuriraj CIJELI red: C[i][:] += A[i][k] × B[k][:]
*/

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
casperSPARSE_kernel(
    int M,
    int N_kroz_4,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ val,
    const float4* __restrict__ B,  //matrica B je u float 4 zapisu
    float4* __restrict__ C)
{
    int warp = (blockIdx.x * (blockDim.x / WARP_SIZE)) + threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & 31;  
    
    if (warp >= M) return;
    
    
    int start = row_ptr[warp];
    int end   = row_ptr[warp + 1];
    
    __shared__ int  sh_col[THREADS_PER_BLOCK];
    __shared__ float sh_val[THREADS_PER_BLOCK];
    
    int*   scol = &sh_col[(threadIdx.x / 32) * 32];
    float* sval = &sh_val[(threadIdx.x / 32) * 32];
    
    // Uzima po 32 kolone matrice B
    for (int col_offset = 0; col_offset < N_kroz_4; col_offset += 32) {
        
        float4 acc = make_float4(0, 0, 0, 0);
        
        int current_col_lane = col_offset + lane;  

        bool active_lane = (current_col_lane < N_kroz_4);

        for (int i = start; i < end; i += 32) {
            
            int batch_size = min(32, end - i);
            
            if (lane < batch_size) {
                scol[lane] = col_idx[i + lane];
                sval[lane] = val[i + lane];
            }
            __syncwarp();
            
            if (active_lane) {
                #pragma unroll   
                for (int k = 0; k < batch_size; k++) { 
                    float a = sval[k]; 
                    int row_B = scol[k];  
                    float4 b = B[row_B * N_kroz_4 + current_col_lane]; 
                                                                    
                    
                    acc.x += a * b.x;
                    acc.y += a * b.y;
                    acc.z += a * b.z;
                    acc.w += a * b.w;
                }
            }
            __syncwarp();
        }
        
        if (active_lane) {
            C[warp * N_kroz_4 + current_col_lane] = acc; 
        }
    }
}

static void run_cusparse_validation(
    cusparseHandle_t handle,
    int M, int K, int N, int NNZ,
    int* d_ptr, int* d_idx, float* d_val,
    float* d_B, float* d_C)
{
    cusparseSpMatDescr_t A;
    cusparseDnMatDescr_t B, C;
    float alpha = 1.0f, beta = 0.0f;

    CUSPARSE_CHECK(cusparseCreateCsr(&A, M, K, NNZ, d_ptr, d_idx, d_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    CUSPARSE_CHECK(cusparseCreateDnMat(&B, K, N, N, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&C, M, N, N, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;
    
    size_t bufferSize = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, B, &beta, C, CUDA_R_32F,
        alg, &bufferSize));

    void* dBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    CUSPARSE_CHECK(cusparseSpMM_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, B, &beta, C, CUDA_R_32F,
        alg, dBuffer));

    CUSPARSE_CHECK(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, B, &beta, C, CUDA_R_32F,
        alg, dBuffer));
    
    CUDA_CHECK(cudaDeviceSynchronize());

    cusparseDestroySpMat(A);
    cusparseDestroyDnMat(B);
    cusparseDestroyDnMat(C);
    cudaFree(dBuffer);
}

static void stats_ms(const std::vector<float>& v, float& mn, float& av, float& mx) {
  if (v.empty()) { mn=0; av=0; mx=0; return; }
  mn = *std::min_element(v.begin(), v.end());
  mx = *std::max_element(v.begin(), v.end());
  av = std::accumulate(v.begin(), v.end(), 0.0f) / (float)v.size();
}

int main(int argc, char** argv) {
    enable_utf8_console();
    std::cout << std::fixed << std::setprecision(3);

    // Default parametri
    int M = 70000; 
    int K = 10000;
    int N = 10000;
    float SPARSITY_PERCENT = 1.0f;

    // Parsiranje argumenata komandne linije
    if (argc >= 5) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
        SPARSITY_PERCENT = (float)atof(argv[4]);
        std::cout << "Koriste se parametri iz komandne linije.\n";
    } else if (argc > 1) {
        std::cout << "Upotreba: " << argv[0] << " [M K N sparsity_percent]\n";
        std::cout << "Primjer: " << argv[0] << " 4096 1500 200000 5.0\n";
        std::cout << "Koriste se default parametri.\n\n";
    }

    assert(N % 4 == 0 && "N mora biti djeljivo sa 4");
    const int N_vec = N / 4;
    
    int estimated_NNZ = (int)(M * K * SPARSITY_PERCENT / 100.0f);

    std::cout << "\n=== KONFIGURACIJA MATRICA ===\n";
    std::cout << "Matrica A: " << M << "x" << K << " (sparse CSR)\n";
    std::cout << "Sparsity: " << SPARSITY_PERCENT << "% (prosječno)\n";
    std::cout << "Estimirani NNZ: ~" << estimated_NNZ << "\n";
    std::cout << "Matrica B: " << K << "x" << N << " (dense)\n";
    std::cout << "Matrica C: " << M << "x" << N << " (dense)\n";
    std::cout << "=============================\n\n";

    auto bytes_ptr = (size_t)(M+1) * sizeof(int);
    auto bytes_idx = (size_t)estimated_NNZ * sizeof(int) * 2;
    auto bytes_val = (size_t)estimated_NNZ * sizeof(float) * 2;
    auto bytes_B   = (size_t)K * N * sizeof(float);
    auto bytes_C   = (size_t)M * N * sizeof(float);

    int *d_ptr=nullptr, *d_idx=nullptr;
    float *d_val=nullptr, *d_B=nullptr, *d_C=nullptr;

    CUDA_CHECK(cudaMalloc(&d_ptr, bytes_ptr));
    CUDA_CHECK(cudaMalloc(&d_idx, bytes_idx));
    CUDA_CHECK(cudaMalloc(&d_val, bytes_val));
    CUDA_CHECK(cudaMalloc(&d_B,   bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C,   bytes_C));

    std::cout << "Generisanje podataka na GPU...\n";

    unsigned int seed = (unsigned int)time(NULL);

    int blocks_ptr = (M + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    generator_matrica_row_ptr_random_kernel<<<blocks_ptr, THREADS_PER_BLOCK>>>(
        d_ptr, M, K, SPARSITY_PERCENT, seed);
    
    prefix_sum_kernel<<<1, 1>>>(d_ptr, M);
    CUDA_CHECK(cudaDeviceSynchronize());

    int actual_NNZ;
    CUDA_CHECK(cudaMemcpy(&actual_NNZ, d_ptr + M, sizeof(int), cudaMemcpyDeviceToHost));


    int blocks_sparse = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    generator_matrica_sparse_kernel<<<blocks_sparse, THREADS_PER_BLOCK>>>(
        d_idx, d_val, d_ptr, M, K, actual_NNZ, seed);

    size_t total_B_elements = (size_t)K * N;
    int blocks_dense = (int)((total_B_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    generator_matrica_dense_kernel<<<blocks_dense, THREADS_PER_BLOCK>>>(
        d_B, total_B_elements, seed + 12345);
    
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "GPU generisanje završeno.\n";

    std::vector<int> h_row_ptr(min(M+1, 11));
    CUDA_CHECK(cudaMemcpy(h_row_ptr.data(), d_ptr, h_row_ptr.size() * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::cout << "\nPrimjer NNZ po redovima (prvih 10):\n";
    for (int i = 0; i < min(M, 10); i++) {
        std::cout << "Red " << i << ": " << (h_row_ptr[i+1] - h_row_ptr[i]) << " NNZ\n";
    }

    std::vector<float> h_C_casperSPARSE((size_t)M * N);
    std::vector<float> h_C_cusparse((size_t)M * N);
    
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // cuSPARSE - samo za validaciju (bez mjerenja vremena)
    std::cout << "\nPokrećem cuSPARSE za validaciju...\n";
    run_cusparse_validation(handle, M, K, N, actual_NNZ, d_ptr, d_idx, d_val, d_B, d_C);
    CUDA_CHECK(cudaMemcpy(h_C_cusparse.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

    // casperSPARSE - sa mjerenjima
    const int warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
    int blocks = (M + warps_per_block - 1) / warps_per_block;

    const int WARMUP = 1;
    const int ITERS  = 1;

    std::cout << "\nPokrećem casperSPARSE...\n";
    for (int i=0;i<WARMUP;i++){
      casperSPARSE_kernel<<<blocks, THREADS_PER_BLOCK>>>(
          M, N_vec, d_ptr, d_idx, d_val, (float4*)d_B, (float4*)d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> casperSPARSE_ms;
    for (int i=0;i<ITERS;i++){
      GpuTimer gt;
      gt.start();
      casperSPARSE_kernel<<<blocks, THREADS_PER_BLOCK>>>(
          M, N_vec, d_ptr, d_idx, d_val, (float4*)d_B, (float4*)d_C);
      float ms = gt.stop();
      casperSPARSE_ms.push_back(ms);
    }

    CUDA_CHECK(cudaMemcpy(h_C_casperSPARSE.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Validacija rezultata
    double max_err = 0.0;
    size_t check_limit = (size_t)M * N; 
    
    for (size_t i = 0; i < check_limit; i++) {
        double diff = (double)std::abs(h_C_casperSPARSE[i] - h_C_cusparse[i]);
        if (diff > max_err) max_err = diff;
    }

    std::cout << "\n=== VALIDACIJA REZULTATA ===\n";
    std::cout << "Max Apsolutna Greška (vs cuSPARSE): " << max_err << "\n";
    if (max_err > 1e-2) std::cout << "UPOZORENJE: Greška je prevelika!\n";
    else std::cout << "Rezultati su tačni.\n";

    // Performanse - samo casperSPARSE
    float c_mn, c_av, c_mx;
    stats_ms(casperSPARSE_ms, c_mn, c_av, c_mx);

    double flops = 2.0 * (double)actual_NNZ * (double)N;
    
    size_t actual_bytes_idx = (size_t)actual_NNZ * sizeof(int);
    size_t actual_bytes_val = (size_t)actual_NNZ * sizeof(float);
    size_t bytes_read = bytes_ptr + actual_bytes_idx + actual_bytes_val + bytes_B;
    size_t bytes_write = bytes_C;
    size_t total_bytes = bytes_read + bytes_write;

    std::cout << "\n=== PERFORMANSE casperSPARSE ===\n";
    std::cout << "Vrijeme (min/avg/max): " << c_mn << " / " << c_av << " / " << c_mx << " ms\n";
    std::cout << "GFLOPS: " << (flops/1e9)/(c_av/1e3) << "\n";
    std::cout << "Propusnost memorije: " << bytes_to_human((size_t)((total_bytes/1e9)/(c_av/1e3)*1e9)) << "/s\n";

        // Ispis prvih 10x10 elemenata matrice C (cuSPARSE)
    std::cout << "\n=== Prvih 10x10 elemenata C (cuSPARSE) ===\n";
    int rows_to_print = std::min(10, M);
    int cols_to_print = std::min(10, N);

    for (int i = 0; i < rows_to_print; i++) {
        for (int j = 0; j < cols_to_print; j++) {
            std::cout << std::setw(10) 
                      << h_C_cusparse[(size_t)i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Ispis prvih 10x10 elemenata matrice C (casperSPARSE)
    std::cout << "\n=== Prvih 10x10 elemenata C (casperSPARSE) ===\n";

    for (int i = 0; i < rows_to_print; i++) {
        for (int j = 0; j < cols_to_print; j++) {
            std::cout << std::setw(10) 
                      << h_C_casperSPARSE[(size_t)i * N + j] << " ";
        }
        std::cout << "\n";
    }


    CUSPARSE_CHECK(cusparseDestroy(handle));
    cudaFree(d_ptr); cudaFree(d_idx); cudaFree(d_val);
    cudaFree(d_B); cudaFree(d_C);

    return 0;
}
