//
//  main.cpp
//  PitchDetector_test
//
//  Created by Bruno Di Giorgi on 20/01/15.
//  Copyright (c) 2015 Subsequently and Furthermore, Inc. All rights reserved.
//

#include "gtest/gtest.h"

#include "fdhmm.hpp"

/*
 hs: hidden node size
 os: observation node size
 nu: initial distribution  (np.array shape = (k))
 Q:  transition distribution  (np.array shape = (k, k))
 g:  emission distribution  (np.array shape = (k, r))
 
 x: hidden states
 y: observations
 
 nseq: number of sequences
 slen: sequence length
 */



namespace bdg {
    
    TEST(HMMTest, sampleAndLearn) {
        
        int seed = time(NULL);
        srand(seed);
        
        // I will generate [nseq] sequences of constant length [slen_]
        uint32_t nseq = 1000;  // how many sequences
        int slen_ = 100;  // length of the sequences
        
        // slen_ is used for the sake of clarity.
        // The sequences can be of different lengths
        
        uint32_t slen[nseq];  // an array containing the lengths of the sequences
        uint32_t sind[nseq];  // an array containing the start indices of the sequences
        for(int i=0; i<nseq; i++) {
            sind[i] = i * slen_;
            slen[i] = slen_;
        }
        uint32_t tot_len = slen_ * nseq;  // the sum of the lengths of the sequences
        uint32_t max_len = slen_;  // the length of the longest sequence
        uint32_t* y = new uint32_t[tot_len];  // observed values
        uint32_t* x = new uint32_t[tot_len];  // hidden values
        
        const int hs = 4;  // hidden node size
        const int os = 3;  // observed node size (alphabet size)
        
        HMM target_hmm(hs, os);  // parameters are initialized randomly
        
        // generate the sequences
        target_hmm.sample(nseq, slen, x, y);
        
        // wrap the generated sequences into the HMMObservations structure
        HMMObservations obs(nseq, tot_len, max_len, slen, sind, y);
        
        // store the observations for later reuse
        obs.dump("../../dataset/dataset.dat");
        
        // create an hmm for learning
        HMM hmm(hs, os);
        
        // ...and assign the observation to this hmm
        hmm.set_observations(&obs);
        
        // as a sanity check, I may initialize the parameters of the hmm to the target hmm
        // hmm.set_parameters(target_hmm);
        
        // this are all the auxiliary data that should be allocated
        // this is tedious and verbose but is actually what allow me to get this
        // efficient in terms of memory
        
        // -allocating-
        double* phi = new double[obs.max_len * hs];
        double* beta = new double[obs.max_len * hs];
        double* c = new double[obs.max_len];
        double* N = new double[hs*hs];
        double* Naux1 = new double[hs*hs];
        double* Naux2 = new double[hs*hs];
        double* M = new double[hs*os];
        double* Maux = new double[hs*os];
        double* NU = new double[hs];
        double* post = new double[obs.max_len * hs];
        double absdif, loglik;
        // ------------
        
        int nit = 100;  // number of Expectation Maximization iterations
        for(int it = 0; it < nit; it++) {
            hmm.Estep_all(phi, c, beta, post, N, Naux1, Naux2, M, NU, &loglik);
            hmm.Mstep(N, M, NU, Naux1, Maux, &absdif);
            std::cout << "EM iteration " << it << ", loglik = " << loglik << std::endl;
        }
        
        // print target parameters
        std::cout << "*** target ***" << std::endl;
        target_hmm.print_parameters();
        std::cout << std::endl;
        
        // print learned parameters
        std::cout << "*** learned model ***" << std::endl;
        hmm.print_parameters();
        std::cout << std::endl;
        
        // -deallocating-
        delete[] post;
        delete[] N;
        delete[] Naux1;
        delete[] Naux2;
        delete[] M;
        delete[] Maux;
        delete[] NU;
        delete[] phi;
        delete[] c;
        delete[] beta;
        // --------------
        
        // deallocate the generated sequence
        delete[] y;
        delete[] x;
    }
    
    TEST(HMMTest, trainer) {
        int seed = time(NULL);
        srand(seed);
        
        const int hs = 4;      // how many configurations of hidden node size
        int EMiterations = 100;    // how many Expectation Maximization iterations
        int nworkers = 4;        // for parallelization
        
        std::string dataset = "../../dataset/dataset.dat";
        
        std::string logfile = "results_logfile.txt";
        std::string logprogressfile = "logfile.txt";
        
        HMMTrainer trainer(dataset, logfile, logprogressfile,
                           hs, EMiterations, nworkers);
        trainer.run();
    }
    
    TEST(HMMTest, viterbi) {
        
        int seed = time(NULL);
        srand(seed);
        
        // I will generate [nseq] sequences of constant length [slen_]
        uint32_t nseq = 10000;  // how many sequences
        int slen_ = 100;  // length of the sequences
        
        // slen_ is used for the sake of clarity.
        // The sequences can be of different lengths
        
        uint32_t slen[nseq];  // an array containing the lengths of the sequences
        uint32_t sind[nseq];  // an array containing the start indices of the sequences
        for(int i=0; i<nseq; i++) {
            sind[i] = i * slen_;
            slen[i] = slen_;
        }
        uint32_t tot_len = slen_ * nseq;  // the sum of the lengths of the sequences
        uint32_t max_len = slen_;  // the length of the longest sequence
        uint32_t* y = new uint32_t[tot_len];  // observed values
        uint32_t* x = new uint32_t[tot_len];  // hidden values
        
        const int hs = 2;  // hidden node size
        const int os = 4;  // observed node size (alphabet size)
        HMM target_hmm(hs, os);
        
        // this time I'll assign specific parameters to the target hmm
        double nu[hs] = {0., 1.};
        double Q[hs*hs] = {0.8, 0.2, 0.1, 0.9};
        double g[hs*os] = {0.25, 0.25, 0.25, 0.25, 0.1, 0.1, 0.4, 0.4};
        target_hmm.set_parameters(nu, Q, g);
        
        target_hmm.sample(nseq, slen, x, y);
        HMMObservations obs(nseq, tot_len, max_len, slen, sind, y);
        
        HMM hmm(hs, os);
        hmm.set_observations(&obs);
        
        // allocate the memory needed for viterbi
        
        // -allocating-
        double logprob;
        double* T1 = new double[hs * obs.max_len];
        int* T2 = new int[hs * obs.max_len];
        double* aux = new double[hs];
        uint32_t* x_est = new uint32_t[obs.tot_len];
        // ------------
        
        for(int s = 0; s < obs.nseq; s++) {
            
            // I have 2 alternative ways for using viterbi:
            // a. having the entire sequence
            // b. casually from 0 to t<slen
            
            // a.
            hmm.viterbi(obs.slen[s], obs.y + obs.sind[s], T1, T2, aux);
                
            // b.
            // for(int t = 0; t < obs.slen[s]; t++)
            //     hmm.viterbi_update(obs.y + obs.sind[s], T1, T2, aux, t, t+1);
            
            // backtracing
            hmm.viterbi_backtrack(obs.slen[s], T1, T2, x_est + obs.sind[s], &logprob);
        }
        
        // here I can analyze x_est, to find cooccurrence with the ground truth x
        
        // -deallocating-
        delete[] x_est;
        delete[] T1;
        delete[] T2;
        delete[] aux;
        // -------------
        
        delete[] y;
        delete[] x;
    }
    
    // TODO: examples with predictions using posterior and viterbi
    TEST(HMMTest, predict) {
        const int hs = 2;
        const int os = 4;
        
        HMM hmm(hs, os);
        
        double* dist = new double[os];
        
        uint32_t x_last = 1;
        hmm.predict_dist(x_last, dist);
        
        uint32_t y_next = 3;
        double p = hmm.prob_next_obs(x_last, y_next);
        ASSERT_NEAR(dist[y_next], p, 0.000001);
        std::cout << p << std::endl;
        
        printv(os, dist);
        
        delete[] dist;
    }
    
    TEST(HMMTest, DatasetLoading) {
        
        std::string dataset_filename = "../../dataset/data.dat";
        // the dataset file should contain all the sequences of observed values
        // packed as uint32_t in the following way:
        // nseq slen[0] ... slen[nseq-1] y[0] ... y[sum(slen) - 1]
        
        Dataset d(dataset_filename);
        d.print();
    }
    
    TEST(HMMTest, HMMExperiment) {
        
        // The HMMExperiment is used for tests about how well the model can
        // predict the next symbol
        
        // 1. allocating/deallocating memory
        // 2. parallelization (with n worker threads)
        // 3. kfold cross-validation
        // 4. saving the model files
        // 5. saving training metadata (cross-entropy on train and test set for each iteration)
        
        srand(0);  // set the seed
        
        const int n_hs = 1;      // how many configurations of hidden node size
        int hs_arr[n_hs] = {2};  // hidden node sizes
        int EMiterations = 4;    // how many Expectation Maximization iterations
        int nfolds = 5;          // k-fold cross validation
        bool shuffle = false;    // shuffle the sequences before k-fold splitting
        int nrep = 1;            // how many repetitions
        int nworkers = 4;        // for parallelization
        
        std::string dataset = "../../dataset/dataset.dat";
        
        std::string logfile = "results_logfile.txt";
        std::string logprogressfile = "logfile.txt";
        
        HMMExperiment hmmExp(dataset,
                             logfile,
                             logprogressfile,
                             hs_arr[0],
                             nfolds,
                             shuffle,
                             EMiterations,
                             nworkers,
                             HMMExperiment::PredictionType::posterior);
        
        for(int i = 0; i < n_hs; i++) {
            int hs = hs_arr[i];
            
            std::cout << "*** hidden size: " << hs << " *** " << std::endl;
            hmmExp.set_hs(hs);
            
            for(int irep = 0; irep < nrep; irep++)
                hmmExp.run();
        }
    }
    
    TEST(HMMTest, save_load_parameters) {
        // initialize an hmm with 4 hidden symbols and 2 observed symbols
        HMM hmm(4, 2);
        
        // save and print the parameters
        hmm.save_parameters("models/prova.model");
        hmm.print_parameters();
        
        // initialize another hmm with different specs
        HMM hmm2(8, 4);
        
        // loading the previously saved parameters will restore the previous hmm
        hmm2.load_parameters("models/prova.model");
        hmm2.print_parameters();
    }
    
    
}
