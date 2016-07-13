//
//  HMMWorkerThread.cpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 12/07/16.
//
//

#include "HMMWorkerThread.hpp"

namespace bdg {
    
    HMMWorkerThread::~HMMWorkerThread() {
        if(mthread != nullptr) {delete mthread; mthread = nullptr;}
        if(tr_set != nullptr) {delete[] tr_set; tr_set = nullptr;}
        if(te_set != nullptr) {delete[] te_set; te_set = nullptr;}
        deallocate_aux();
    }
    
    void HMMWorkerThread::set_hmm(HMM* hmm_) {
        hmm = hmm_;
        deallocate_aux();
        allocate_aux();
    }
    
    void HMMWorkerThread::set_seqs(int tr_size_, int* tr_set_, int te_size_, int* te_set_) {
        tr_size = tr_size_;
        te_size = te_size_;
        if(tr_set != nullptr) {delete[] tr_set; tr_set = nullptr;}
        if(te_set != nullptr) {delete[] te_set; te_set = nullptr;}
        if(tr_size) {
            tr_set = new int[tr_size];
            memcpy(tr_set, tr_set_, tr_size * sizeof(int));
        }
        if(te_size) {
            te_set = new int[te_size];
            memcpy(te_set, te_set_, te_size * sizeof(int));
        }
    }
    
    void HMMWorkerThread::set_train_seqs(int imin, int imax) {
        tr_size = imax - imin;
        if(tr_set != nullptr) {delete[] tr_set; tr_set = nullptr;}
        tr_set = new int[tr_size];
        int* p = tr_set;
        for (int i=imin; i<imax; i++) {
            *p++ = i;
        }
    }
    
    void HMMWorkerThread::allocate_aux() {
        int hs = hmm->hs;
        int os = hmm->os;
        int obs_max_len = hmm->obs_max_len();
        
        T1 = new double[hs * obs_max_len];
        T2 = new int[hs * obs_max_len];
        aux = new double[hs];
        x_est = new uint32_t[obs_max_len];
        
        phi = new double[obs_max_len * hs];
        beta = new double[obs_max_len * hs];
        c = new double[obs_max_len];
        
        N = new double[hs*hs];
        Naux1 = new double[hs*hs];
        Naux2 = new double[hs*hs];
        M = new double[hs*os];
        Maux = new double[hs*os];
        NU = new double[hs];
        post = new double[obs_max_len * hs];
    }
    
    void HMMWorkerThread::deallocate_aux() {
        if(post != nullptr) {delete[] post; post = nullptr;}
        if(NU != nullptr) {delete[] NU; NU = nullptr;}
        if(Maux != nullptr){delete[] Maux; Maux = nullptr;}
        if(M != nullptr) {delete[] M; M = nullptr;}
        if(Naux2 != nullptr) {delete[] Naux2; Naux2 = nullptr;}
        if(Naux1 != nullptr) {delete[] Naux1; Naux1 = nullptr;}
        if(N != nullptr) {delete[] N; N = nullptr;}
        
        if(c != nullptr) {delete[] c; c = nullptr;}
        if(beta != nullptr) {delete[] beta; beta = nullptr;}
        if(phi != nullptr) {delete[] phi; phi = nullptr;}
        
        if(x_est != nullptr) {delete[] x_est; x_est = nullptr;}
        if(aux != nullptr) {delete[] aux; aux = nullptr;}
        if(T2 != nullptr) {delete[] T2; T2 = nullptr;}
        if(T1 != nullptr) {delete[] T1; T1 = nullptr;}
    }
    
    void HMMWorkerThread::run_Estep() {
        if(mthread != nullptr) {delete mthread; mthread = nullptr;}
        mthread = new std::thread([this]{this->estep();});
    }
    
    void HMMWorkerThread::estep() {
        hmm->Estep(tr_size, tr_set, phi, c, beta, post, N, Naux1, Naux2, M, NU, &loglik);
    }
    
    void HMMWorkerThread::run_crossentropy_posterior() {
        if(mthread != nullptr) {delete mthread; mthread = nullptr;}
        mthread = new std::thread([this]{this->crossentropy_posterior();});
    }
    
    void HMMWorkerThread::crossentropy_posterior() {
        tr_entropy = hmm->average_entropy(tr_size, tr_set, phi, c, Naux1, &tr_count);
        te_entropy = hmm->average_entropy(te_size, te_set, phi, c, Naux1, &te_count);
    }
    
    void HMMWorkerThread::run_crossentropy_viterbi() {
        if(mthread != nullptr) {delete mthread; mthread = nullptr;}
        mthread = new std::thread([this]{this->crossentropy_viterbi();});
    }
    
    void HMMWorkerThread::crossentropy_viterbi() {
        tr_entropy = hmm->average_entropy(tr_size, tr_set, x_est, T1, T2, aux, &tr_count);
        te_entropy = hmm->average_entropy(te_size, te_set, x_est, T1, T2, aux, &te_count);
    }
    
    void HMMWorkerThread::join() {
        mthread->join();
        delete mthread;
        mthread = nullptr;
    }
    
}  // namespace bdg