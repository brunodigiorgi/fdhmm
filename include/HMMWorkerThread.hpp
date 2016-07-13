//
//  HMMWorkerThread.hpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 12/07/16.
//
//

#ifndef HMMWorkerThread_hpp
#define HMMWorkerThread_hpp

#include <stdio.h>
#include <thread>
#include "HMM.hpp"


namespace bdg {
    
    class HMMWorkerThread {
    public:
        ~HMMWorkerThread();
        
        void set_hmm(HMM* hmm);
        void set_seqs(int tr_size, int* tr_set, int te_size, int* te_set);
        void set_train_seqs(int imin, int imax);  // imax not included
        
        double* N {nullptr}; // = new double[hs*hs];
        double* M {nullptr}; // = new double[hs*os];
        double* NU {nullptr}; // = new double[hs];
        double loglik {0};
        
        std::thread* mthread {nullptr};
        void run_Estep();
        void run_crossentropy_posterior();
        void run_crossentropy_viterbi();
        
        void join();
        
        double tr_entropy, te_entropy;
        int tr_count, te_count;
        int tr_size {0};
        int te_size {0};
        
    private:
        HMM* hmm {nullptr};
        
        void estep();
        void crossentropy_posterior();
        void crossentropy_viterbi();
        
        void allocate_aux();
        void deallocate_aux();
        
        int* tr_set {nullptr};
        int* te_set {nullptr};
        
        double* phi {nullptr}; // = new double[obs.tot_len * hs];
        double* beta {nullptr}; // = new double[obs.tot_len * hs];
        double* c {nullptr}; // = new double[obs.tot_len];
        double* Naux1 {nullptr}; // = new double[hs*hs];
        double* Naux2 {nullptr}; // = new double[hs*hs];
        double* Maux {nullptr}; // = new double[hs*os];
        double* post {nullptr}; // = new double[obs.max_len * hs];
        
        double* T1 {nullptr}; // = new double[hs * obs.max_len];
        int* T2 {nullptr}; // = new int[hs * obs.max_len];
        double* aux {nullptr}; // = new double[hs];
        uint32_t* x_est {nullptr}; // = new uint32_t[obs.tot_len];        
    };
    
}  // namespace bdg

#endif /* HMMWorkerThread_hpp */
