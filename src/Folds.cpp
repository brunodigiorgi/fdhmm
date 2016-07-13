//
//  Folds.cpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 14/06/16.
//
//

#include "Folds.hpp"

#include <stdlib.h>
#include <math.h>
#include <cassert>

namespace bdg {
    
    Folds::Folds(int n_, int nfolds_, bool shuffle_)
    : nfolds(nfolds_), n(n_) {
        assert(n > nfolds && nfolds > 1);
        
        indices = new int[n];
        for (int i = 0; i < n; i++) {
            indices[i] = i;
        }
        
        if(shuffle_)
            shuffle(indices, n);
        
        fold_sizes = new int[nfolds];
        for (int i = 0; i < nfolds; i++) {
            fold_sizes[i] = n/nfolds;
        }
        
        for (int i = 0; i < n % nfolds; i++) {
            fold_sizes[i] += 1;
        }
        
        
    }
    
    Folds::~Folds() {
        delete[] indices;
        delete[] fold_sizes;
    }
    
    void Folds::shuffle(int *array, int n)
    {
        for (int i = 0; i < n - 1; i++)
        {
            int j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
    
    void Folds::get_fold(int ifold,
                         int* train_set, int* train_set_size,
                         int* test_set, int* test_set_size) {
        *test_set_size = fold_sizes[ifold];
        *train_set_size = n - *test_set_size;
        
        int c = 0;
        int k = 0;
        int fold_start = 0;
        
        // train_set
        for(int i = 0; i < ifold; i++) {
            for(int i_ = fold_start; i_ < fold_start + fold_sizes[i]; i_++) {
                train_set[c++] = indices[i_];
            }
            fold_start += fold_sizes[i];
        }
        
        // test_set
        for (int i = fold_start; i < fold_start + fold_sizes[ifold]; i++) {
            test_set[k++] =  indices[i];
        }
        fold_start += fold_sizes[ifold];
        
        // train_set
        for (int i = ifold + 1; i < nfolds; i++) {
            for(int i_ = fold_start; i_ < fold_start + fold_sizes[i]; i_++) {
                train_set[c++] = indices[i_];
            }
            fold_start += fold_sizes[i];
        }
        
        assert(fold_start == n);
        assert(c + k == n);
    }
    
}