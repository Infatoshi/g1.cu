// bench_nn.cu -- honest bulk-synchronous policy-MLP forward timing on Ampere (sm_86).
// The "GEMM-kernel" half of the bulk-synchronous baseline that fusion must beat.
//
// MLP forward over N worlds = chain of dense GEMMs with ReLU between hidden layers:
//   h0 = relu(obs @ W0 + b0)      [N x in] @ [in x H0] -> [N x H0]
//   h1 = relu(h0  @ W1 + b1)      ...
//   act = hN @ Wout               [N x H_last] @ [H_last x action_dim]
// We time the full forward, fp16 inputs/weights with fp32 accumulate via cublasGemmEx
// (tensor-core mma path on Ampere -- the fair, fast baseline a real RL stack would use).
// Bias-add + ReLU fused in a tiny elementwise kernel (cheap vs the GEMMs; included in time).
//
// Usage: ./bench_nn <N> <policy: small|medium|large> [reps]
//   small  hidden = [128,128]
//   medium hidden = [256,256]
//   large  hidden = [512,256,128]
// obs_dim=100, action_dim=29 (G1 walking policy).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
#define CB(call) do { cublasStatus_t s=(call); if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS %s:%d status=%d\n",__FILE__,__LINE__,(int)s); exit(1);} } while(0)

#define OBS_DIM 100
#define ACT_DIM 29

// bias-add + optional ReLU, row-major [N x D]
__global__ void bias_relu(half* x, const half* bias, int N, int D, int relu){
    long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    long tot = (long)N*D;
    if (i>=tot) return;
    float v = __half2float(x[i]) + __half2float(bias[i%D]);
    if (relu && v<0.f) v=0.f;
    x[i]=__float2half(v);
}

// One dense layer: C[N x out] = A[N x in] @ W[in x out], all row-major.
// cuBLAS is column-major; treat row-major [N x out] as col-major [out x N].
// col-major: C^T(out x N) = W^T(out x in) * A^T(in x N).  In cublas terms with
// column-major leading dims = the row-major row length, we compute:
//   gemm(N=out, M? ...) -- easier: use the standard trick
//   C_rm(N x out): call gemm with op(A)=W as (out x in) col-major == W_rm(in x out),
// We store W row-major as [in x out]; as col-major that array is [out x in].
// So cublasGemmEx(opA=N, opB=N, m=out, n=N, k=in,
//                 A=W (lda=out), B=obs (ldb=in -> but obs is [N x in] row-major = [in x N] colmajor, ldb=in),
//                 C (ldc=out)) gives C col-major [out x N] = C row-major [N x out]. Correct.
static void layer(cublasHandle_t h, const half* A_obs, const half* W, half* C,
                  int N, int in, int out){
    const float alpha=1.f, beta=0.f;
    CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    out, N, in,
                    &alpha,
                    W, CUDA_R_16F, out,
                    A_obs, CUDA_R_16F, in,
                    &beta,
                    C, CUDA_R_16F, out,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s N small|medium|large [reps]\n",argv[0]); return 1; }
    int N=atoi(argv[1]);
    const char* pol=argv[2];
    int reps=(argc>3)?atoi(argv[3]):200;

    std::vector<int> hid;
    if(!strcmp(pol,"small"))  hid={128,128};
    else if(!strcmp(pol,"medium")) hid={256,256};
    else if(!strcmp(pol,"large"))  hid={512,256,128};
    else { fprintf(stderr,"bad policy\n"); return 1; }

    // layer dims: obs->hid[0]->...->hid[L-1]->act
    std::vector<int> dims; dims.push_back(OBS_DIM);
    for(int h:hid) dims.push_back(h);
    dims.push_back(ACT_DIM);
    int L = dims.size()-1; // num GEMMs

    cublasHandle_t hb; CB(cublasCreate(&hb));
    CB(cublasSetMathMode(hb, CUBLAS_TENSOR_OP_MATH));

    // weights + biases (random-ish, values don't matter for timing)
    std::vector<half*> W(L), B(L), buf(L); // buf[l] = activation output of layer l, [N x dims[l+1]]
    for(int l=0;l<L;++l){
        int in=dims[l], out=dims[l+1];
        half *w,*b,*o;
        CK(cudaMalloc(&w,(size_t)in*out*sizeof(half)));
        CK(cudaMalloc(&b,(size_t)out*sizeof(half)));
        CK(cudaMalloc(&o,(size_t)N*out*sizeof(half)));
        std::vector<half> hw((size_t)in*out), hbi(out);
        for(size_t i=0;i<hw.size();++i) hw[i]=__float2half(0.02f*((i%7)-3));
        for(int i=0;i<out;++i) hbi[i]=__float2half(0.01f*((i%5)-2));
        CK(cudaMemcpy(w,hw.data(),hw.size()*sizeof(half),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(b,hbi.data(),out*sizeof(half),cudaMemcpyHostToDevice));
        W[l]=w; B[l]=b; buf[l]=o;
    }
    half* obs; CK(cudaMalloc(&obs,(size_t)N*OBS_DIM*sizeof(half)));
    { std::vector<half> ho((size_t)N*OBS_DIM); for(size_t i=0;i<ho.size();++i) ho[i]=__float2half(0.1f*((i%11)-5));
      CK(cudaMemcpy(obs,ho.data(),ho.size()*sizeof(half),cudaMemcpyHostToDevice)); }

    auto forward=[&](){
        const half* A=obs; int in=OBS_DIM;
        for(int l=0;l<L;++l){
            int out=dims[l+1];
            layer(hb, A, W[l], buf[l], N, in, out);
            int relu = (l<L-1)?1:0; // no relu on output
            long tot=(long)N*out; int bs=256; int gr=(tot+bs-1)/bs;
            bias_relu<<<gr,bs>>>(buf[l], B[l], N, out, relu);
            A=buf[l]; in=out;
        }
    };

    // warmup
    for(int i=0;i<20;++i) forward();
    CK(cudaDeviceSynchronize());

    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for(int i=0;i<reps;++i) forward();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b));
    double per = ms/reps;
    printf("NN N=%d policy=%s layers=", N, pol);
    for(size_t i=0;i<dims.size();++i) printf("%d%s", dims[i], i+1<dims.size()?"-":"");
    printf("  %.5f ms/fwd  (%.3e fwd/s)\n", per, 1.0/(per/1e3));
    return 0;
}
