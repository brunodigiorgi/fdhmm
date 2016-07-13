//
//  World.hpp
//  PitchDetector
//
//  Created by Bruno Di Giorgi on 28/02/16.
//
//

#ifndef World_hpp
#define World_hpp

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

namespace bdg {    
    
    class Dataset;
    class HMMObservations {
    public:
        HMMObservations(uint32_t nseq, uint32_t tot_len, uint32_t max_len,
                        uint32_t* slen, uint32_t* sind, uint32_t* y);
        HMMObservations(Dataset& dataset);
        void dump(std::string filename);
        uint32_t nseq;
        uint32_t tot_len;
        uint32_t max_len; // maximum length of a sequence
        uint32_t* slen;  // slen[s] = length of the s sequence
        uint32_t* sind;  // sind[s] = starting index of the s sequence
        uint32_t* y;
    };
    
    class HMM {
    public:
        HMM(int hs, int os);
        ~HMM();
        
        void allocate_parameters();
        void deallocate_parameters();
        
        void set_parameters(double* nu, double* Q, double* g);
        void set_parameters(const HMM& hmm);
        void set_nu(double* nu);
        void init_random_parameters();
        void print_parameters();
        void sanity_check_parameters();
        
        void sample(uint32_t nseq, uint32_t* slen, uint32_t* x, uint32_t* y);
        
        void set_observations(HMMObservations* o);
        int obs_max_len();
        
        // phi[t*hs + k] = P(x_t=k|Y_0,...,Y_t, nu, Q, g)
        // probability of being in the state x_t=k, given observations up to t
        
        // phi [max_len * hs]
        //      where k = 0..hs, s = 0..nseq, t = 0..slen[s]
        //      hs values for each element of each sequence
        // c [tot_len] conditional likelihood
        void filter(int iseq, double* phi, double* c, double* Z);
        
        // phi [tot_len * hs]
        void filter_all(double* phi, double* c, double* Z);
        
        // beta[t*hs + k] = P(Y_t+1,..seq,Y_T | x_t=k, nu, Q, g) / P(Y_t+1,...,Y_T | Y_0,...,Y_t)
        // prob of the obs seq Y_t+1,...,Y_T, given state x_t=k
        // over the
        // prob of the obs seq Y_t+1,...,Y_T, given obs seq Y_0,...,Y_t
        
        // beta[max_len * hs]
        //      where k = 0..hs, s = 0..nseq, t = 0..slen[s]
        //      hs values for each element of each sequence
        // c [tot_len] conditional likelihood
        void smoother(int iseq, double* c, double* beta, double* Z);
        
        // beta [tot_len * hs]
        void smoother_all(double* c, double* beta, double* Z);
        
        // post[t*hs + k] = P(x_t=k | Y)
        // probability of the state x_t=k given the entire obs seq
        
        // phi, c and beta are auxiliary vectors:
        // phi and beta: [o->max_len * hs]
        // c: [o->max_len]
        // post: [o->max_len * hs]
        // N: [hs, hs]
        // Naux1: [hs, hs]
        // Naux2: [hs, hs]
        // M: [hs, os]
        void Estep_all(double* phi, double* c, double* beta, double* post,
                       double* N, double* Naux1, double* Naux2,
                       double* M, double* NU, double* loglik);
        
        // call Estep to all the [nseq] sequences contained in iseq
        void Estep(int nseq, int* iseq,
                   double* phi, double* c, double* beta, double* post,
                   double* N, double* Naux1, double* Naux2,
                   double* M, double* NU, double* loglik);
        void Estep(int iseq,
                   double* phi, double* c, double* beta, double* post,
                   double* N, double* Naux1, double* Naux2,
                   double* M, double* NU, double* loglik);
        
        // N: [hs, hs]
        // M: [hs, os]
        // this modifies Q and g and returns
        // absdif = sum(abs(oldQ-Q) + sum(abs(oldg-g)
        // loglik = sum(log(c))
        // eps is a smoothing parameter (~ Laplace smoothing)
        void Mstep(double* N, double* M, double* NU,
                   double* Naux, double* Maux,
                   double* absdif, double eps=1e-12);
        
        
        // phi: [n, hs]
        // T1 and T2 [n, k] row major order
        // y [n]
        // aux [hs]
        // t_end is not included
        void viterbi_update(uint32_t* y,
                            double* T1, int* T2, double* aux,
                            int t_start, int t_end);
        
        void viterbi(int n, uint32_t* y,
                     double* T1, int* T2, double* aux);
        
        void viterbi_backtrack(int n, double* T1, int* T2, uint32_t* x, double* logprob);
        
        // dist [os]
        void predict_dist(uint32_t x_last, double* y_next_dist);
        void predict_dist(double* x_last_dist, double* y_next_dist);
        
        double prob_next_obs(uint32_t x_last, uint32_t y_next);
        double prob_next_obs(double* x_last_dist, uint32_t y_next);
        
        // x, T1, T2, aux [max_len]
        double average_entropy(int nseq, int* iseq,
                               uint32_t* x_est, double* T1, int* T2, double* aux, int* count,
                               bool verbose=false);
        double average_entropy(int nseq, int* iseq,
                               double* phi, double* c, double* Z, int* count,
                               bool verbose=false);
        
        void print_update(int i, int n, int n_step);
        
        void save_parameters(std::string filename);
        void load_parameters(std::string filename);
        
        
        
        int hs, os;
        
        // row-major order
        double* nu {nullptr};  // initial distribution (hs)
        double* Q {nullptr};  // transitions (hs x hs)
        double* g {nullptr};  // emissions (hs x os)
        
        void update_log_parameters();
        double* nu_log {nullptr};  // initial distribution (hs)
        double* Q_log {nullptr};  // transitions (hs x hs)
        double* g_log {nullptr};  // emissions (hs x os)
        
    private:
        
        HMMObservations* o {nullptr};
        
        double* T {nullptr};  // auxiliary for prediction
    };
    
}  // namespace bdg

#endif /* World_hpp */
