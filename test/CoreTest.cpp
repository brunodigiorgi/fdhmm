//
//  CoreTest.cpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 12/07/16.
//
//

#include <stdio.h>
#include "gtest/gtest.h"

#if (defined (__APPLE__))

#include "Core.hpp"
#include <mach/mach_time.h>
class Timer {
public:
    double abs_to_nanos = 0;
    
    Timer() {
        mach_timebase_info_data_t timebase_info;
        mach_timebase_info(&timebase_info);
        abs_to_nanos = timebase_info.numer  / (double)timebase_info.denom;
    }
    
    double time() {
        return static_cast<double>(abs_to_nanos * mach_absolute_time());
    }
};

static int nloops {1};

namespace bdg {
    
    TEST(CoreTest, vfill) {
        double C[4];
        for(int i = 0; i < 4; i++) C[i] = i;
        
        double A = 14;
        
        Timer t;
        double st = t.time();
        for(int i=0; i<nloops; i++){
            vfill(&A, C, 2, 2);
        }
        std::cout << "vfill:" << (t.time() - st) << std::endl;
        
        ASSERT_EQ(A, C[2]);
    }
    
    TEST(CoreTest, mtrans) {
        double A[16];
        for(int i = 0; i < 16; ++i) A[i] = i;
        
        
        Timer t;
        double st = t.time();
        for(int i=0; i<nloops; i++){
            mtrans(A, 1, A, 1, 4, 4);
        }
        std::cout << "mtrans:" << (t.time() - st) << std::endl;
        
        ASSERT_EQ(A[1], 4);
        
        std::cout << "transpose" << std::endl;
        printm(4, 4, A);
    }
    
    TEST(CoreTest, vadd_vsub_vmul) {
        double A[4];
        for(int i = 0; i < 4; ++i) A[i] = i;
        double C[4];
        
        Timer t;
        double st = t.time();
        for(int i=0; i<nloops; i++){
            vadd(A, 2, A, 2, C, 2, 2);
            ASSERT_EQ(C[2], 4);
            
            vsub(A, 2, A, 2, C, 2, 2);
            ASSERT_EQ(C[2], 0);
            
            vmul(A, 2, A, 2, C, 1, 2);
            ASSERT_EQ(C[1], 4);
        }
        std::cout << "vadd_vsub_vmul:" << (t.time() - st) << std::endl;
    }
    
    TEST(CoreTest, vsadd_vsdiv_maxvi) {
        double A[4];
        for(int i = 0; i < 4; ++i) A[i] = i;
        double C[4];
        double B = 2;
        
        Timer t;
        double st = t.time();
        for(int i=0; i<nloops; i++){
            vsdiv(A, 2, &B, C, 1, 2);
            ASSERT_EQ(C[1], 1);
            
            vsadd(A, 2, &B, C, 1, 2);
            ASSERT_EQ(C[1], 4);
            
            int idx;
            maxvi(A, 2, &B, &idx, 2);
            ASSERT_EQ(2, idx);
            ASSERT_EQ(2, B);
        }
        std::cout << "vsadd_vsdiv_maxvi:" << (t.time() - st) << std::endl;
    }
    
    TEST(CoreTest, sve_svemg) {
        double A[4];
        for(int i = 0; i < 4; i++) A[i] = (i*2) - 1;
        double C;
        
        Timer t;
        double st = t.time();
        for(int i=0; i<nloops; i++){
            sve(A, 2, &C, 2);
            ASSERT_EQ(2, C);
            
            svemg(A, 2, &C, 2);
            ASSERT_EQ(C, 4);
        }
        std::cout << "sve_svemg:" << (t.time() - st) << std::endl;
    }
    
    TEST(CoreTest, log) {
        double A[4];
        for(int i = 0; i < 4; i++) A[i] = i + 1;
        double C[4];
        
        int N = 4;
        
        Timer t;
        double st = t.time();
        for(int i=0; i<nloops; i++){
            vlog(C, A, &N);
        }
        std::cout << "log:" << (t.time() - st) << std::endl;
        
        for(int i = 0; i < 4; i++) {
            ASSERT_NEAR(C[i], log(A[i]), 1e-10);
        }
    }
    
    TEST(CoreTest, vsma_dotpr) {
        double A[4];
        for(int i = 0; i < 4; i++) A[i] = i + 1;
        double B = 2;
        double C[4];
        for(int i = 0; i < 4; i++) C[i] = i + 2;
        double D[4];
        
        Timer t;
        double st = t.time();
        for(int i=0; i<nloops; i++){
            vsma(A, 2, &B, C, 2, D, 1, 2);
            ASSERT_EQ(10, D[1]);
            
            dotpr(A, 2, C, 2, &B, 2);
            ASSERT_EQ(14, B);
        }
        std::cout << "vsma_dotpr:" << (t.time() - st) << std::endl;
    }
}

#endif