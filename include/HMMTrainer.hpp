//
//  HMMTrainer.hpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 13/07/16.
//
//

#ifndef HMMTrainer_hpp
#define HMMTrainer_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include "HMM.hpp"
#include "HMMWorkerThread.hpp"
#include "Dataset.hpp"
#include "Folds.hpp"

namespace bdg {
    class HMMTrainer {
    public:
        
        HMMTrainer(std::string dataset,
                   std::string logfile,
                   std::string logprogressfile,
                   int hs, int EMiterations, int nworkers);
        ~HMMTrainer();
        
        void set_dataset(std::string dataset);
        void set_hs(int hs);
        void load_parameters(std::string filename); // proxy to hmm
        void run();
        
        int EMiterations {10};
        int hs {2};
        int os {2};
        
    private:
        HMM* hmm {nullptr};
        
        std::ofstream logfile;
        std::ofstream logprogressfile;
        
        std::string model_filename();
        
        Dataset* d {nullptr};
        HMMObservations* obs {nullptr};
        
        int nworkers {4};
        std::vector<HMMWorkerThread> workers;
        void assign_sequences_to_workers();
        
        void allocate_aux();
        void deallocate_aux();
        
        //auxiliary memory
        double* N {nullptr}; // = new double[hs*hs];
        double* Naux1 {nullptr}; // = new double[hs*hs];
        double* M {nullptr}; // = new double[hs*os];
        double* Maux {nullptr}; // = new double[hs*os];
        double* NU {nullptr}; // = new double[hs];
        
        double* EM_loglik {nullptr}; // = new double[EMiterations];
    };
}

#endif /* HMMTrainer_hpp */
