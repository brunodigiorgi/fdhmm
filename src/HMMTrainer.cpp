//
//  HMMTrainer.cpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 13/07/16.
//
//

#include "HMMTrainer.hpp"
#include <cmath>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <string.h>
#include "Core.hpp"

namespace bdg {
    
    HMMTrainer::HMMTrainer(std::string dataset,
                           std::string logfile_,
                           std::string logprogressfile_,
                           int hs_, int EMiterations_, int nworkers_):
    logfile(logfile_, std::ofstream::app), logprogressfile(logprogressfile_, std::ofstream::out)
    {
        assert(logfile.good());
        assert(logprogressfile.good());
        
        nworkers = nworkers_;
        for(int iworkers=0; iworkers<nworkers; iworkers++)
            workers.push_back(HMMWorkerThread());
        
        hs = hs_;
        EMiterations = EMiterations_;
        
        set_dataset(dataset);
    }
    
    HMMTrainer::~HMMTrainer() {
        deallocate_aux();
        
        logfile.close();
        
        if(hmm != nullptr) {delete hmm; hmm = nullptr;}
        if(obs != nullptr) {delete obs; obs = nullptr;}
        if(d != nullptr) {delete d; d = nullptr;}
    }
    
    void HMMTrainer::allocate_aux() {
        N = new double[hs*hs];
        Naux1 = new double[hs*hs];
        M = new double[hs*os];
        Maux = new double[hs*os];
        NU = new double[hs];
        
        EM_loglik = new double[EMiterations];
    }
    
    void HMMTrainer::deallocate_aux() {
        if(NU != nullptr) {delete[] NU; NU = nullptr;}
        if(Maux != nullptr){delete[] Maux; Maux = nullptr;}
        if(M != nullptr) {delete[] M; M = nullptr;}
        if(Naux1 != nullptr) {delete[] Naux1; Naux1 = nullptr;}
        if(N != nullptr) {delete[] N; N = nullptr;}
        
        if(EM_loglik != nullptr) {delete[] EM_loglik; EM_loglik = nullptr;}
    }
    
    void HMMTrainer::set_dataset(std::string dataset) {
        if(d != nullptr) {delete d; d = nullptr;}
        if(obs != nullptr) {delete obs; obs = nullptr;}
        if(hmm != nullptr) {delete hmm; hmm = nullptr;}
        
        d = new Dataset(dataset);
        obs = new HMMObservations(*d);
        os = d->alphabet_size;
        hmm = new HMM(hs, os);
        hmm->set_observations(obs);
        
        for(HMMWorkerThread & w : workers)
            w.set_hmm(hmm);
        
        deallocate_aux();
        allocate_aux();
    }
    
    void HMMTrainer::set_hs(int hs_) {
        hs = hs_;
        
        if(hmm != nullptr) {delete hmm; hmm = nullptr;}
        hmm = new HMM(hs, os);
        hmm->set_observations(obs);
        
        for(HMMWorkerThread & w : workers)
            w.set_hmm(hmm);
        
        deallocate_aux();
        allocate_aux();
    }
    
    void HMMTrainer::load_parameters(std::string filename) {
        hmm->load_parameters(filename);
    }
    
    void HMMTrainer::run() {
        
        auto start_t = std::time(nullptr);
        auto start_time = *std::localtime(&start_t);
        
        hmm->init_random_parameters();
        
        // subdivide sequences into worker threads
        assign_sequences_to_workers();
        
        // learn parameters
        double absdif;
        for(int it = 0; it < EMiterations; it++) {
            
            for(HMMWorkerThread & w : workers)
                w.run_Estep();
            for(HMMWorkerThread & w : workers)
                w.join();
            
            EM_loglik[it] = 0;
            memset(NU, 0, hs*sizeof(double));
            memset(N, 0, hs*hs*sizeof(double));
            memset(M, 0, hs*os*sizeof(double));
            for(HMMWorkerThread & w : workers) {
                EM_loglik[it] += w.loglik;
                vadd(NU, 1, w.NU, 1, NU, 1, hs);
                vadd(N, 1, w.N, 1, N, 1, hs*hs);
                vadd(M, 1, w.M, 1, M, 1, hs*os);
            }
            
            hmm->Mstep(N, M, NU, Naux1, Maux, &absdif);
            
            auto now = std::time(nullptr);
            auto now_ = *std::localtime(&now);
            logprogressfile << std::put_time(&now_, "%Y-%m-%d %H:%M:%S") << " ";
            
            logprogressfile << "EM iteration " << it << ", loglik = " << EM_loglik[it] << std::endl;
        }
        
        // hmm->print_parameters();
        std::string filename = model_filename();
        hmm->save_parameters(filename);
        
        // LOG
        auto end_t = std::time(nullptr);
        auto end_time = *std::localtime(&end_t);
        auto duration = std::difftime(end_t, start_t);
        logfile << "{" << std::endl;
        logfile << "\"start_time\": \"" << std::put_time(&start_time, "%d-%m-%Y %H:%M:%S") << "\", " << std::endl;
        logfile << "\"end_time\": \"" << std::put_time(&end_time, "%d-%m-%Y %H:%M:%S") << "\", " << std::endl;
        logfile << "\"duration\": " << duration << ", " << std::endl;
        logfile << "\"Dataset\": \"" << d->filename << "\", " << std::endl;
        logfile << "\"EMiterations\": " << EMiterations << ", " << std::endl;
        logfile << "\"EM_loglik\": " << "[" << v_to_str(EMiterations, EM_loglik) << "], " << std::endl;
        logfile << "\"hs\": " << hs << ", " << std::endl;
        logfile << "\"nworkers\": " << nworkers << ", " << std::endl;
        logfile << "\"fn_params\": \"" << filename << "\", " << std::endl;
        logfile << "}," << std::endl;
        logfile << std::endl;
    }
    
    void HMMTrainer::assign_sequences_to_workers() {
        double tr_k = obs->nseq / nworkers;
        for (int i=0; i<nworkers; i++) {
            int tr_imin = floor(i * tr_k);
            int tr_imax = floor((i+1) * tr_k);  // not included
            
            workers[i].set_train_seqs(tr_imin, tr_imax);
        }
    }
    
    std::string HMMTrainer::model_filename() {
        std::ostringstream filename;
        filename << "models/" << time(0) << ".model";
        return filename.str();
    }
    
    
    
}