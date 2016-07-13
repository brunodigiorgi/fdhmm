//
//  HMMExperiment.hpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 14/06/16.
//
//

#ifndef HMMExperiment_hpp
#define HMMExperiment_hpp

#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "HMM.hpp"
#include "HMMWorkerThread.hpp"
#include "Dataset.hpp"
#include "Folds.hpp"

namespace bdg {
    
    class HMMExperiment  {
    public:
        
        typedef enum PredictionType {
            viterbi,
            posterior
        } PredictionType;
        
        HMMExperiment(std::string dataset, std::string logfile, std::string logprogressfile,
                      int hs, int nfolds, bool shuffle, int EMiterations, int nworkers,
                      PredictionType prediction_type);
        ~HMMExperiment();
        
        void set_dataset(std::string dataset);
        void set_nseq(int nseq);
        void set_hs(int hs);
        void load_parameters(std::string filename); // proxy to hmm
        
        void run();
        void print_update(int i, int n, int n_step);
        
        int nfolds {1};
        int EMiterations {10};
        int hs {2};
        int os {2};
        
    private:
        HMM* hmm {nullptr};
        
        std::ofstream logfile;
        std::ofstream logprogressfile;
        
        bool shuffle {true};
        Folds* folds {nullptr};
        Dataset* d {nullptr};
        HMMObservations* obs {nullptr};
        PredictionType prediction_type {PredictionType::viterbi};
        
        int nworkers {4};
        std::vector<HMMWorkerThread> workers;
        void assign_sequences_to_workers(int tr_size, int* tr_set, int te_size, int* te_set);
        
        void allocate_aux();
        void deallocate_aux();
        
        // auxiliary memory chunks
        double* N {nullptr}; // = new double[hs*hs];
        double* Naux1 {nullptr}; // = new double[hs*hs];
        double* M {nullptr}; // = new double[hs*os];
        double* Maux {nullptr}; // = new double[hs*os];
        double* NU {nullptr}; // = new double[hs];
        
        double* EM_loglik {nullptr}; // = new double[EMiterations];
        double* EM_te_entropy {nullptr}; // = new double[EMiterations];
        double* EM_tr_entropy {nullptr}; // = new double[EMiterations];
        
        int* tr_set {nullptr}; // = new int[n];
        int* te_set {nullptr}; // = new int[n];
        int tr_size {0};
        int te_size {0};
        
        std::string model_filename();
    };  
    
}

#endif /* HMMExperiment_hpp */
