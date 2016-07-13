//
//  Core.hpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 12/07/16.
//
//

#ifndef Core_hpp
#define Core_hpp

#include <iostream>
#include <cmath>
#include <sstream>

#define FORCE_USE_UNROLLED_LOOPS

#if defined FORCE_USE_EIGEN
#define USE_EIGEN
#elif defined FORCE_USE_UNROLLED_LOOPS
#define USE_UNROLLED_LOOPS
#elif (defined (__APPLE__))
#define USE_ACCELERATE
#elif defined (__linux__)
#define USE_UNROLLED_LOOPS
#else
#error "unknown platform"
#endif

#if defined (USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined (USE_EIGEN)
#include <eigen3/Eigen/Dense>
#include <cblas.h>
#elif defined (USE_UNROLLED_LOOPS)
#include <cblas.h>
#else
#error "unknown platform"
#endif

namespace bdg {
    
    int randm(int n, double* p);
    void rand_stoc_mat(int m, int n, double* p, int pseudo_count, int uniform_count);
    void normalize(int n, double* v);
    
    template <typename T>
    void printv(int n, T* v, int stride=1) {
        for (int i = 0; i < n * stride; i+=stride) {
            std::cout << v[i] << ", ";
        }
        std::cout << std::endl;
    }
    
    template <typename T>
    void printm(int m, int n, T* v) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << v[i*n+j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    template <typename T>
    std::string v_to_str(int n, T* v) {
        std::ostringstream out;
        for(int i = 0; i < n-1; i++) {
            out << v[i] << ", ";
        }
        out << v[n-1];
        return out.str();
    }
    
    void vfill(const double* A, double* C, int IC, int N);
    
    void vadd(const double* A, int IA,
              const double* B, int IB, double* C, int IC, int N);
    void vsub(const double* B, int IB,
              const double* A, int IA, double* C, int IC, int N);
    void vmul(const double* A, int IA,
              const double* B, int IB, double* C, int IC, int N);

    void vsadd(const double* A, int IA,
               const double* B, double* C, int IC, int N);
    void vsdiv(const double* A, int IA,
               const double* B, double* C, int IC, int N);
    
    void sve(const double* A, int IA, double* C, int N);
    void svemg(const double* A, int IA, double* C, int N);

    void maxvi(const double* A, int IA, double* C, int* IC, int N);
    
    void vsma(const double* A, int IA, const double* B,
              const double* C, int IC, double* D, int ID, int N);
    
    void mtrans(double* A, int IA, double* C, int IC, int M, int N);
    void dotpr(const double* A, int IA, const double* B, int IB, double* C, int N);
    
    void vlog(double* C, const double* A, const int* N);
    
}  // namespace bdg

#endif /* Core_hpp */
