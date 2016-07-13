//
//  Core.cpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 12/07/16.
//
//

#include "Core.hpp"

#include <cassert>

namespace bdg {
    
    int randm(int n, double* p){
        int res=0;
        double q=p[0];
        double u=(rand()+0.0)/RAND_MAX;
        while(u>q) q+=p[++res];
        return(res);
    }
    
    void rand_stoc_mat(int m, int n, double* p, int pseudo_count, int uniform_count) {
        double u;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                u = (rand()+0.0)/RAND_MAX;
                u = uniform_count + u * pseudo_count;  // make sure none is 0!
                p[i*n + j] = u;
            }
            normalize(n, p+i*n);
        }
    }
    
    void normalize(int n, double* v) {
        double s;
        sve(v, 1, &s, n);
        assert(s != 0);
        vsdiv(v, 1, &s, v, 1, n);
    }
    
    void vfill(const double* A, double* C, int IC, int N) {
#if defined (USE_ACCELERATE)
        vDSP_vfillD(A, C, IC, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<ArrayXd, Unaligned, Stride<1, Dynamic>> Map_C(C, N, Stride<1, Dynamic>(1, IC));
        Map_C.fill(*A);
#elif defined (USE_UNROLLED_LOOPS)
        int i = 0;
        for (; (i+4) < N; i += 4) {
            C[(i+0)*IC] = *A;
            C[(i+1)*IC] = *A;
            C[(i+2)*IC] = *A;
            C[(i+3)*IC] = *A;
        }
        
        for (; i < N; i += 1) {
            C[i*IC] = *A;
        }
#endif
    }
    
    void mtrans(double* A, int IA, double* C, int IC, int M, int N) {
#if defined (USE_ACCELERATE)
        vDSP_mtransD(A, IA, C, IC, M, N);
#elif defined (USE_EIGEN)
        assert(M == N && IA == 1 && IC == 1 && C == A);
        
        using namespace Eigen;
        Map<MatrixXd> Map_A(A, N, M);
        Map<MatrixXd> Map_C(C, M, N);
        Map_A.transposeInPlace();
        Map_C = Map_A;
#elif defined (USE_UNROLLED_LOOPS)
        assert(M == N && IA == 1 && IC == 1);
        
        // square matrix, just use N
        for (int i = 0; i < N; i++) {
            for(int j = i + 1; j < N; j++) {
                double x = A[(i*N+j)*IA];
                double y = A[(j*N+i)*IA];
                C[(j*N+i)*IC] = x;
                C[(i*N+j)*IC] = y;
            }
        }
        for (int i = 0; i < N; i++) {
            C[(i*N+i)*IC] = A[(i*N+i)*IC];
        }
#endif
    }
    
    void vsdiv(const double* A, int IA,
               const double* B, double* C, int IC, int N) {
#if defined (USE_ACCELERATE)
        vDSP_vsdivD(A, IA, B, C, IC, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_A(A, N, Stride<1, Dynamic>(1, IA));
        Map<ArrayXd, Unaligned, Stride<1, Dynamic>> Map_C(C, N, Stride<1, Dynamic>(1, IC));
        Map_C = Map_A / *B;
#elif defined (USE_UNROLLED_LOOPS)
        double invB = 1 / *B;
        int i = 0;
        for (; (i+4) < N; i += 4) {
            C[(i+0)*IC] = A[(i+0)*IA] * invB;
            C[(i+1)*IC] = A[(i+1)*IA] * invB;
            C[(i+2)*IC] = A[(i+2)*IA] * invB;
            C[(i+3)*IC] = A[(i+3)*IA] * invB;
        }
        
        for (; i < N; i += 1) {
            C[i*IC] = A[i*IA] * invB;
        }
        
#endif
    }
    
    void sve(const double* A, int IA, double* C, int N) {
#if defined (USE_ACCELERATE)
        vDSP_sveD(A, IA, C, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_A(A, N, Stride<1, Dynamic>(1, IA));
        *C = Map_A.sum();
#elif defined (USE_UNROLLED_LOOPS)
        int i = 0;
        double cum[4] = {0, 0, 0, 0};
        for (; (i+4) < N; i += 4) {
            cum[0] += A[(i+0)*IA];
            cum[1] += A[(i+1)*IA];
            cum[2] += A[(i+2)*IA];
            cum[3] += A[(i+3)*IA];
        }
        *C = cum[0] + cum[1] + cum[2] + cum[3];
        
        for (; i < N; i += 1) {
            *C += A[i*IA];
        }
        
#endif
    }
    
    void vlog(double* C, const double* A, const int* N) {
#if defined (USE_ACCELERATE)
        vvlog(C, A, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const ArrayXd> Map_A(A, *N);
        Map<ArrayXd> Map_C(C, *N);
        Map_C = Map_A.log();
#elif defined (USE_UNROLLED_LOOPS)
        int i = 0;
        for (; (i+4) < *N; i += 4) {
            C[(i+0)] = log(A[(i+0)]);
            C[(i+1)] = log(A[(i+1)]);
            C[(i+2)] = log(A[(i+2)]);
            C[(i+3)] = log(A[(i+3)]);
        }
        
        for (; i < *N; i += 1) {
            C[i] = log(A[i]);
        }
#endif
    }
    
    void dotpr(const double* A, int IA, const double* B, int IB, double* C, int N) {
#if defined (USE_ACCELERATE)
        vDSP_dotprD(A, IA, B, IB, C, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const VectorXd, Unaligned, Stride<1, Dynamic>> Map_A(A, N, Stride<1, Dynamic>(1, IA));
        Map<const VectorXd, Unaligned, Stride<1, Dynamic>> Map_B(B, N, Stride<1, Dynamic>(1, IB));
        *C = Map_A.dot(Map_B);
#elif defined (USE_UNROLLED_LOOPS)
        int i = 0;
        double cum[4] = {0, 0, 0, 0};
        for (; (i+4) < N; i += 4) {
            cum[0] += A[(i+0)*IA] * B[(i+0)*IB];
            cum[1] += A[(i+1)*IA] * B[(i+1)*IB];
            cum[2] += A[(i+2)*IA] * B[(i+2)*IB];
            cum[3] += A[(i+3)*IA] * B[(i+3)*IB];
        }
        *C = cum[0] + cum[1] + cum[2] + cum[3];
        
        for (; i < N; i += 1) {
            *C += A[i*IA] * B[i*IB];
        }
#endif
    }
    
    void vadd(const double* A, int IA,
              const double* B, int IB, double* C, int IC, int N) {
#if defined (USE_ACCELERATE)
        vDSP_vaddD(A, IA, B, IB, C, IC, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_A(A, N, Stride<1, Dynamic>(1, IA));
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_B(B, N, Stride<1, Dynamic>(1, IB));
        Map<ArrayXd, Unaligned, Stride<1, Dynamic>> Map_C(C, N, Stride<1, Dynamic>(1, IC));
        Map_C = Map_A + Map_B;
#elif defined (USE_UNROLLED_LOOPS)
        int i = 0;
        for (; (i+4) < N; i += 4) {
            C[(i+0)*IC] = A[(i+0)*IA] + B[(i+0)*IB];
            C[(i+1)*IC] = A[(i+1)*IA] + B[(i+1)*IB];
            C[(i+2)*IC] = A[(i+2)*IA] + B[(i+2)*IB];
            C[(i+3)*IC] = A[(i+3)*IA] + B[(i+3)*IB];
        }
        
        for (; i < N; i += 1) {
            C[i*IC] = A[i*IA] + B[i*IB];
        }
        
#endif
    }
    
    void vsub(const double* B, int IB,
              const double* A, int IA, double* C, int IC, int N) {
#if defined (USE_ACCELERATE)
        vDSP_vsubD(B, IB, A, IA, C, IC, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_A(A, N, Stride<1, Dynamic>(1, IA));
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_B(B, N, Stride<1, Dynamic>(1, IB));
        Map<ArrayXd, Unaligned, Stride<1, Dynamic>> Map_C(C, N, Stride<1, Dynamic>(1, IC));
        Map_C = Map_A - Map_B;
#elif defined (USE_UNROLLED_LOOPS)
        int i = 0;
        for (; (i+4) < N; i += 4) {
            C[(i+0)*IC] = A[(i+0)*IA] - B[(i+0)*IB];
            C[(i+1)*IC] = A[(i+1)*IA] - B[(i+1)*IB];
            C[(i+2)*IC] = A[(i+2)*IA] - B[(i+2)*IB];
            C[(i+3)*IC] = A[(i+3)*IA] - B[(i+3)*IB];
        }
        
        for (; i < N; i += 1) {
            C[i*IC] = A[i*IA] - B[i*IB];
        }
#endif
    }
    
    void vmul(const double* A, int IA,
              const double* B, int IB, double* C, int IC, int N) {
#if defined (USE_ACCELERATE)
        vDSP_vmulD(A, IA, B, IB, C, IC, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_A(A, N, Stride<1, Dynamic>(1, IA));
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_B(B, N, Stride<1, Dynamic>(1, IB));
        Map<ArrayXd, Unaligned, Stride<1, Dynamic>> Map_C(C, N, Stride<1, Dynamic>(1, IC));
        Map_C = Map_A * Map_B;
#elif defined (USE_UNROLLED_LOOPS)
        int i = 0;
        for (; (i+4) < N; i += 4) {
            C[(i+0)*IC] = A[(i+0)*IA] * B[(i+0)*IB];
            C[(i+1)*IC] = A[(i+1)*IA] * B[(i+1)*IB];
            C[(i+2)*IC] = A[(i+2)*IA] * B[(i+2)*IB];
            C[(i+3)*IC] = A[(i+3)*IA] * B[(i+3)*IB];
        }
        
        for (; i < N; i += 1) {
            C[i*IC] = A[i*IA] * B[i*IB];
        }
        
#endif
    }
    
    void vsadd(const double* A, int IA,
               const double* B, double* C, int IC, int N) {
#if defined (USE_ACCELERATE)
        vDSP_vsaddD(A, IA, B, C, IC, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_A(A, N, Stride<1, Dynamic>(1, IA));
        Map<ArrayXd, Unaligned, Stride<1, Dynamic>> Map_C(C, N, Stride<1, Dynamic>(1, IC));
        Map_C = Map_A + *B;
#elif defined (USE_UNROLLED_LOOPS)
        int i = 0;
        for (; (i+4) < N; i += 4) {
            C[(i+0)*IC] = A[(i+0)*IA] + *B;
            C[(i+1)*IC] = A[(i+1)*IA] + *B;
            C[(i+2)*IC] = A[(i+2)*IA] + *B;
            C[(i+3)*IC] = A[(i+3)*IA] + *B;
        }
        
        for (; i < N; i += 1) {
            C[i*IC] = A[i*IA] + *B;
        }
#endif
    }
    
    
    void svemg(const double* A, int IA, double* C, int N) {
#if defined (USE_ACCELERATE)
        vDSP_svemgD(A, IA, C, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const VectorXd, Unaligned, Stride<1, Dynamic>> Map_A(A, N, Stride<1, Dynamic>(1, IA));
        *C = Map_A.lpNorm<1>();
#elif defined (USE_UNROLLED_LOOPS)
        int i = 0;
        double cum[4] = {0, 0, 0, 0};
        for (; (i+4) < N; i += 4) {
            cum[0] += std::abs(A[(i+0)*IA]);
            cum[1] += std::abs(A[(i+1)*IA]);
            cum[2] += std::abs(A[(i+2)*IA]);
            cum[3] += std::abs(A[(i+3)*IA]);
        }
        *C = cum[0] + cum[1] + cum[2] + cum[3];
        
        for (; i < N; i += 1) {
            *C += std::abs(A[i*IA]);
        }
#endif
    }
    
    void maxvi(const double* A, int IA, double* C, int* IC, int N) {
#if defined (USE_ACCELERATE)
        vDSP_Length IC_;
        vDSP_maxviD(A, IA, C, &IC_, N);
        *IC = IC_;
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_A(A, N, Stride<1, Dynamic>(1, IA));
        MatrixXf::Index idx;
        *C = Map_A.maxCoeff(&idx);
        *IC = idx * IA;
#elif defined (USE_UNROLLED_LOOPS)
        *C = A[0];
        *IC = 0;
        for (int i=1; i < N; i ++) {
            double a = A[i*IA];
            if(a > *C) {
                *C = a;
                *IC = i*IA;
            }
        }
        
#endif
    }
    
    void vsma(const double* A, int IA, const double* B,
              const double* C, int IC, double* D, int ID, int N) {
#if defined (USE_ACCELERATE)
        vDSP_vsmaD(A, IA, B, C, IC, D, ID, N);
#elif defined (USE_EIGEN)
        using namespace Eigen;
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_A(A, N, Stride<1, Dynamic>(1, IA));
        Map<const ArrayXd, Unaligned, Stride<1, Dynamic>> Map_C(C, N, Stride<1, Dynamic>(1, IC));
        Map<ArrayXd, Unaligned, Stride<1, Dynamic>> Map_D(D, N, Stride<1, Dynamic>(1, ID));
        Map_D = (Map_A * (*B)) + Map_C;
#elif defined (USE_UNROLLED_LOOPS)
        int i = 0;
        double vB = *B;
        for (; (i+4) < N; i += 4) {
            D[(i+0)*ID] = A[(i+0)*IA] * vB + C[(i+0)*IC];
            D[(i+1)*ID] = A[(i+1)*IA] * vB + C[(i+1)*IC];
            D[(i+2)*ID] = A[(i+2)*IA] * vB + C[(i+2)*IC];
            D[(i+3)*ID] = A[(i+3)*IA] * vB + C[(i+3)*IC];
        }
        
        for (; i < N; i += 1) {
            D[i*ID] = A[i*IA] * vB + C[i*IC];
        }
        
#endif
    }
    
}  // namespace bdg


