//
//  Folds.hpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 14/06/16.
//
//

#ifndef Folds_hpp
#define Folds_hpp

#include <stdio.h>

namespace bdg {
    
    // This is equivalent to sklearn.cross_validation.KFold
    class Folds {
    public:
        Folds(int n, int nfolds, bool shuffle);
        ~Folds();
        static void shuffle(int *array, int n);
        
        // train_set and test_set are both [n]
        void get_fold(int ifold,
                      int* train_set, int* train_set_size,
                      int* test_set, int* test_set_size);
        int nfolds;
        int n;
    private:
        int * indices {nullptr};
        int * fold_sizes {nullptr};
        
    };
    
}

#endif /* Folds_hpp */
