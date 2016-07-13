//
//  World.cpp
//  PitchDetector
//
//  Created by Bruno Di Giorgi on 28/02/16.
//
//

#include "HMM.hpp"
#include <cstring>
#include <iostream>

#include "Dataset.hpp"
#include "FileIO.hpp"
#include "Core.hpp"

#include <cmath>

namespace bdg {
    
    HMMObservations::HMMObservations(uint32_t nseq_, uint32_t tot_len_, uint32_t max_len_,
                                     uint32_t* slen_, uint32_t* sind_, uint32_t* y_)
    : nseq(nseq_), tot_len(tot_len_), max_len(max_len_), slen(slen_), sind(sind_), y(y_) {}
    
    HMMObservations::HMMObservations(Dataset& d)
    : nseq(d.nseq), tot_len(d.tot_len), max_len(d.max_len), slen(d.slen), sind(d.sind), y(d.y) {}
    
    void HMMObservations::dump(std::string filename) {
        FILE* fp = fopen(filename.c_str(), "w");
        assert(fp);
        FileIO fio;
        fio.writeUInt32(fp, nseq);
        fio.writeUInt32Array(fp, slen, nseq);
        fio.writeUInt32Array(fp, y, tot_len);
        fclose(fp);
    }
    
    HMM::HMM(int hs_, int os_)
    : hs(hs_), os(os_) {
        allocate_parameters();
        init_random_parameters();
    }
    
    HMM::~HMM() {
        deallocate_parameters();
    }
    
    void HMM::allocate_parameters() {
        deallocate_parameters();
        nu = new double[hs];
        Q = new double[hs * hs];
        g = new double[hs * os];
        nu_log = new double[hs];
        Q_log = new double[hs * hs];
        g_log = new double[hs * os];
        T = new double[os];
    }
    
    void HMM::deallocate_parameters() {
        if(nu != nullptr) {delete[] nu; nu = nullptr;}
        if(Q != nullptr) {delete[] Q; Q = nullptr;}
        if(g != nullptr) {delete[] g; g = nullptr;}
        if(nu_log != nullptr) {delete[] nu_log; nu_log = nullptr;}
        if(Q_log != nullptr) {delete[] Q_log; Q_log = nullptr;}
        if(g_log != nullptr) {delete[] g_log; g_log = nullptr;}
        if(T != nullptr) {delete[] T; T = nullptr;}
    }
    
    void HMM::set_parameters(double* nu_, double* Q_, double* g_) {
        memcpy(nu, nu_, hs * sizeof(double));
        memcpy(Q, Q_, hs * hs * sizeof(double));
        memcpy(g, g_, hs * os * sizeof(double));
    }
    
    void HMM::set_parameters(const HMM& hmm) {
        assert(hmm.hs == hs && hmm.os == os);
        memcpy(nu, hmm.nu, hs * sizeof(double));
        memcpy(Q, hmm.Q, hs * hs * sizeof(double));
        memcpy(g, hmm.g, hs * os * sizeof(double));
    }
    
    void HMM::set_nu(double* nu_) {
        memcpy(nu, nu_, hs * sizeof(double));
    }
    
    void HMM::init_random_parameters() {
        rand_stoc_mat(1, hs, nu, 30, 1);
        rand_stoc_mat(hs, hs, Q, 30, 1);
        rand_stoc_mat(hs, os, g, 30, 1);
        update_log_parameters();
    }
    
    void HMM::update_log_parameters() {
        int Q_size = hs*hs;
        int g_size = hs*os;
        vlog(nu_log, nu, &hs);
        vlog(Q_log, Q, &Q_size);
        vlog(g_log, g, &g_size);
    }
    
    void HMM::print_parameters() {
        std::cout << "initial distribution:" << std::endl;
        printv(hs, nu);
        
        std::cout << "transition distribution:" << std::endl;
        printm(hs, hs, Q);
        
        std::cout << "emission distribution:" << std::endl;
        printm(hs, os, g);
    }
    
    void HMM::sanity_check_parameters() {
        // assert(all([sum(g[:, i]) != 0 for i in range(os)]))
        for (int ios = 0; ios < os; ios++) {
            double sum;
            sve(g + ios, hs, &sum, hs);
            assert(sum != 0);
        }
    }
    
    void HMM::sample(uint32_t nseq, uint32_t* slen, uint32_t* x, uint32_t* y){
        int s_ind = 0;
        int x_, y_;
        for(int s = 0; s < nseq; s++) {
            x_ = randm(hs, nu);
            y_ = randm(os, g + x_*os);
            x[s_ind] = x_;
            y[s_ind] = y_;
            for(int t = 1; t < slen[s]; t++){
                x_ = randm(hs, Q + x[s_ind + t - 1]*hs);
                y_ = randm(os, g + x_*os);
                x[s_ind + t] = x_;
                y[s_ind + t] = y_;
            }
            s_ind += slen[s];
        }
    }
    
    void HMM::set_observations(HMMObservations* o_) {
        o = o_;
    }
    
    int HMM::obs_max_len() {
        if(o == nullptr)
            return 0;
        return o->max_len;        
    }
    
    void HMM::filter_all(double* phi, double* c, double* Z) {
        assert(o != nullptr);
        
        for(int iseq = 0; iseq < o->nseq; iseq++) {
            int si = o->sind[iseq];
            filter(iseq, phi + si*hs, c + si, Z);
        }
    }
    
    void HMM::filter(int iseq, double* phi, double* c, double* Z) {
        assert(o != nullptr);
        
        // DEBUG
        // sanity_check_parameters();
        
        int si = o->sind[iseq];    // start index of the current sequence
        int sl = o->slen[iseq];
        uint32_t* y_ = o->y + si;  // pointer to the current sequence
        
        vmul(nu, 1, g + y_[0], os, Z, 1, hs);
        sve(Z, 1, c, hs);                         // sum
        vsdiv(Z, 1, c, phi, 1, hs);               // normalize
        
        for(int t = 1; t < sl; t++) {
            
            // recursion step
            cblas_dgemv(CblasRowMajor, CblasTrans, hs, hs, 1, Q, hs,
                        phi + (t-1)*hs, 1, 0, Z, 1);
            
            vmul(Z, 1, g + y_[t], os, Z, 1, hs);
            sve(Z, 1, c + t, hs);
            vsdiv(Z, 1, c + t, phi + t*hs, 1, hs);
            
        }
    }
    
    void HMM::smoother_all(double* c, double* beta, double* Z) {
        assert(o != nullptr);
        
        for(int iseq = 0; iseq < o->nseq; iseq++) {
            int si = o->sind[iseq];  // start index of the current sequence
            smoother(iseq, c + si, beta + si*hs, Z);
        }
    }
    
    void HMM::smoother(int iseq, double* c, double* beta, double* Z) {
        int si = o->sind[iseq];           // start index of the current sequence
        int sl = o->slen[iseq];           // length of the current sequence
        uint32_t* y_ = o->y + si;         // pointer to the current sequence
        
        double one = 1;
        
        // initialize: fill last frame of beta with ones
        vfill(&one, beta + (sl-1)*hs, 1, hs);
        
        for(int t = sl - 2; t > -1; t--) {
            double* out = beta + t*hs;
            vmul(beta + (t+1)*hs, 1, g + y_[t+1], os, Z, 1, hs);
            
            // using out as a temp vector because in cblas_dgemv, x cannot be y
            cblas_dgemv(CblasRowMajor, CblasNoTrans, hs, hs, 1, Q, hs,
                        Z, 1, 0, out, 1);
            
            vsdiv(out, 1, c + t+1, out, 1, hs);
        }
    }
    
    void HMM::Estep_all(double* phi, double* c, double* beta, double* post,
                        double* N, double* Naux1, double* Naux2,
                        double* M, double* NU, double* loglik) {
        memset(NU, 0, hs*sizeof(double));
        memset(N, 0, hs*hs*sizeof(double));
        memset(M, 0, hs*os*sizeof(double));
        *loglik = 0;
        
        for(int iseq = 0; iseq < o->nseq; iseq++) {
            Estep(iseq,
                  phi, c, beta, post,
                  N, Naux1, Naux2, M, NU, loglik);
        }
    }
    
    void HMM::Estep(int nseq, int* iseq,
                    double* phi, double* c, double* beta, double* post,
                    double* N, double* Naux1, double* Naux2,
                    double* M, double* NU, double* loglik) {
        memset(NU, 0, hs*sizeof(double));
        memset(N, 0, hs*hs*sizeof(double));
        memset(M, 0, hs*os*sizeof(double));
        *loglik = 0;
        
        for(int i = 0; i < nseq; i++) {
            Estep(iseq[i], phi, c, beta, post, N, Naux1, Naux2, M, NU, loglik);
        }
    }
    
    void HMM::Estep(int iseq,
                    double* phi, double* c, double* beta, double* post,
                    double* N, double* Naux1, double* Naux2,
                    double* M, double* NU, double* loglik) {
        
        int si = o->sind[iseq];    // start index of the current sequence
        int sl = o->slen[iseq];    // sequence length
        uint32_t* y_ = o->y + si;  // pointer to the current sequence
        
        filter(iseq, phi, c, Naux1);
        smoother(iseq, c, beta, Naux1);
        
        vmul(phi, 1, beta, 1, post, 1, hs * sl);
        
        // beta[:, 1:] = beta[:, 1:]*g[:, np.int_(y[1:])]/np.tile(c[1:], [k, 1])
        for(int t = 1; t < sl; t++) {
            double* slice = beta + t*hs;
            vmul(slice, 1, g + y_[t], os, slice, 1, hs);
            vsdiv(slice, 1, c + t, slice, 1, hs);
        }
        
        // a = np.dot(phi[:, 0:-1], beta[:, 1:])
        double* A = phi;
        double* B = beta + hs;
        double* C = Naux1;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, hs, hs, sl-1, 1, A, hs, B, hs, 0, C, hs);
        
        // a = np.transpose(a) * Q
        double* D = Naux2;
        mtrans(C, 1, D, 1, hs, hs);
        
        vmul(D, 1, Q, 1, D, 1, hs * hs);
        
        // N += a
        vadd(N, 1, D, 1, N, 1, hs*hs);
        
        // for each t, update only the col y_[t], with the expected number of states
        for(int t = 0; t < sl; t++) {
            vadd(M + y_[t], os, post + t*hs, 1, M + y_[t], os, hs);
        }
        
        // accumulate the expected number of states at t=0
        vadd(NU, 1, post, 1, NU, 1, hs);
        
        // loglik (destructive for c)
        double loglik_;
        vlog(c, c, &sl);
        sve(c, 1, &loglik_, sl);
        *loglik += loglik_;
    }
    
    void HMM::Mstep(double* N, double* M, double* NU,
                    double* Naux, double* Maux,
                    double* absdif, double eps) {
        // normalize NU, N, M
        
        double sum;
        vsadd(NU, 1, &eps, NU, 1, hs);
        sve(NU, 1, &sum, hs);
        vsdiv(NU, 1, &sum, NU, 1, hs);
        for(int i = 0; i < hs; i++) {
            vsadd(N+i*hs, 1, &eps, N+i*hs, 1, hs);
            sve(N+i*hs, 1, &sum, hs);
            vsdiv(N+i*hs, 1, &sum, N+i*hs, 1, hs);
            
            vsadd(M+i*os, 1, &eps, M+i*os, 1, os);
            sve(M+i*os, 1, &sum, os);
            vsdiv(M+i*os, 1, &sum, M+i*os, 1, os);
        }
        
        // absdif
        *absdif = 0;
        double absdif_;
        
        vsub(N, 1, Q, 1, Naux, 1, hs*hs);
        svemg(Naux, 1, &absdif_, hs*hs);
        *absdif += absdif_;
        
        vsub(M, 1, g, 1, Maux, 1, hs*os);
        svemg(Maux, 1, &absdif_, hs*os);
        *absdif += absdif_;
        
        vsub(NU, 1, nu, 1, Naux, 1, hs);
        svemg(Naux, 1, &absdif_, hs);
        *absdif += absdif_;
        
        // nu=NU Q=N g=M
        memcpy(nu, NU, hs*sizeof(double));
        memcpy(Q, N, hs*hs*sizeof(double));
        memcpy(g, M, hs*os*sizeof(double));
    }
    
    /*
     Reference python implementation of viterbi
     
     def viterbi(obs, hs, start_p, trans_p, emit_p):
     ----splog = np.log(start_p)
     ----tplog = np.log(trans_p)
     ----eplog = np.log(emit_p)
     ----os = eplog.shape[1]
     ----n = len(obs)
     ----T1 = np.zeros((n, hs))
     ----T2 = np.zeros((n, hs), dtype=np.int)
     
     ----# time 0 : initialization, first column of T1
     ----T1[0, :] = splog + eplog[:, obs[0]]
     
     ----# time > 0
     ----for t in range(1, n):
     --------for st in range(hs):
     ------------prob = T1[t-1, :] + tplog[:, st]
     ------------maxi = np.argmax(prob)
     
     ------------T2[t,st] = maxi
     ------------T1[t,st] = prob[maxi] + eplog[st, obs[t]]
     
     ----# best last state
     ----x = np.zeros(n, dtype=np.int)
     ----x[n-1] = np.argmax(T1[n-1, :])
     ----for t in range(n-2, -1, -1):
     --------x[t] = T2[t+1, x[t+1]]
     ----return x
     */
    
    void HMM::viterbi_update(uint32_t* y,
                             double* T1, int* T2, double* aux,
                             int t_start, int t_end) {
        
        for(int t = t_start; t < t_end; t++) {
            if(t == 0) {
                update_log_parameters();
                
                // Initialize T1[0]
                vadd(nu_log, 1, g_log + y[0], os, T1, 1, hs);
            } else {
                double* T1p_ = T1 + (t-1)*hs;
                double* T1_ = T1 + t*hs;
                int* T2_ = T2 + t*hs;
                
                for(int s = 0; s < hs; s++) {  // for each current state
                    
                    // mlt with transitions: aux = T1prev * Q  (for every previous state)
                    vadd(T1p_, 1, Q_log + s, hs, aux, 1, hs);
                    
                    // find best previous state
                    int maxi;
                    maxvi(aux, 1, T1_ + s, &maxi, hs);
                    T2_[s] = maxi;
                    
                    // update with emission g[s, y[t]]
                    T1_[s] += g_log[s*os + y[t]];
                }
            }
        }
    }
    
    void HMM::viterbi(int n, uint32_t* y,
                      double* T1, int* T2, double* aux) {
        viterbi_update(y, T1, T2, aux, 0, n);
    }
    
    void HMM::viterbi_backtrack(int n, double* T1, int* T2, uint32_t* x, double* logprob) {
        
        int maxi;
        int t = n-1;
        maxvi(T1 + t*hs, 1, logprob, &maxi, hs);
        x[t] = maxi;
        
        for(int t = n-2; t >= 0; t--) {
            x[t] = T2[(t+1)*hs + x[t+1]];
        }
    }
    
    void HMM::predict_dist(uint32_t x_last, double* y_next_dist) {
        cblas_dgemv(CblasRowMajor, CblasTrans, hs, os, 1, g, os, Q + x_last*hs, 1, 0, y_next_dist, 1);
    }
    
    void HMM::predict_dist(double* x_last_dist, double* y_next_dist) {
        memset(y_next_dist, 0, os * sizeof(double));
        for (uint32_t i = 0; i < hs; i++) {
            predict_dist(i, T);
            vsma(T, 1, x_last_dist + i, y_next_dist, 1, y_next_dist, 1, os);
        }
    }   
    
    double HMM::prob_next_obs(uint32_t x_last, uint32_t y_next) {
        double out;
        dotpr(g + y_next, os, Q + x_last*hs, 1, &out, hs);
        return out;
    }
    
    double HMM::prob_next_obs(double* x_last_dist, uint32_t y_next) {
        double out = 0;
        for (uint32_t i = 0; i < hs; i++) {
            out += x_last_dist[i] * prob_next_obs(i, y_next);
        }
        
        return out;
    }
    
    
    double HMM::average_entropy(int nseq, int* iseq,
                                uint32_t* x_est, double* T1, int* T2, double* aux, int* count,
                                bool verbose /* = false */) {
        // DEBUG
        // sanity_check_parameters();
        
        double logprob;
        double p_sumlog = 0;
        *count = 0;
        
        for(int i = 0; i < nseq; i++) {
            if(verbose)
                print_update(i, nseq, 20);
            
            int iseq_ = iseq[i];
            int sl = o->slen[iseq_];
            
            for(int t = 0; t < sl - 1; t++) {
                uint32_t y_next = o->y[o->sind[iseq_] + t + 1];
                
                // if prediction with last state by viterbi
                viterbi_update(o->y + o->sind[iseq_], T1, T2, aux, t, t+1);
                viterbi_backtrack(t+1, T1, T2, x_est, &logprob);
                double p = prob_next_obs(x_est[t], y_next);
                
                p_sumlog += log2(p);
                (*count)++;
            }
        }
        
        if(verbose)
            std::cout << std::endl;
        
        double entropy = - p_sumlog;
        return entropy;
    }
    
    // prediction with P(x_t|y_0, ..., y_t)
    double HMM::average_entropy(int nseq, int* iseq,
                                double* phi, double* c, double* Z, int* count,
                                bool verbose /* = false */) {
        double p_sumlog = 0;
        *count = 0;
        
        for(int i = 0; i < nseq; i++) {
            if(verbose)
                print_update(i, nseq, 20);
            
            int iseq_ = iseq[i];
            int sl = o->slen[iseq_];
            
            filter(iseq_, phi, c, Z);
            
            for(int t = 0; t < sl - 1; t++) {
                uint32_t y_next = o->y[o->sind[iseq_] + t + 1];
                
                // phi + t*hs represents P(x_t|y_0, ..., y_t)
                double p = prob_next_obs(phi + t*hs, y_next);
                
                p_sumlog += log2(p);
                (*count)++;
            }
        }
        
        if(verbose)
            std::cout << std::endl;
        
        double entropy = - p_sumlog;
        return entropy;
    }
    
    
    void HMM::print_update(int i, int n, int n_step) {
        for(int i_step = 0; i_step < n_step; i_step++) {
            int thr = (n / (double)n_step) * i_step;
            if(i-1 < thr && i >= thr) {
                std::cout << i_step << "/" << n_step << " ";
            }
        }
    }
    
    void HMM::save_parameters(std::string filename) {
        FileIO fio;
        FILE* fp = fopen(filename.c_str(), "w");
        assert(fp);
        
        fio.writeUInt32(fp, hs);
        fio.writeUInt32(fp, os);
        fio.writeDoubleArray(fp, nu, hs);
        fio.writeDoubleArray(fp, Q, hs*hs);
        fio.writeDoubleArray(fp, g, hs*os);
        
        fclose(fp);        
    }
    
    void HMM::load_parameters(std::string filename) {
        FileIO fio;
        FILE* fp = fopen(filename.c_str(), "r");
        assert(fp);
        
        uint32_t hs_ {0};
        fio.readUInt32(fp, &hs_);
        hs = (int)hs_;
        
        uint32_t os_ {0};
        fio.readUInt32(fp, &os_);
        os = (int)os_;
        
        allocate_parameters();
        
        fio.readDoubleArray(fp, nu, hs);
        fio.readDoubleArray(fp, Q, hs*hs);
        fio.readDoubleArray(fp, g, hs*os);
        update_log_parameters();
        
        fclose(fp);
    }
    
    
}  // namespace bdg
