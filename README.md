# fdhmm
Fast Discrete Hidden Markov Model (HMM)

Advantages: as fast and as efficient as possible. 
Disadvantages: contains just one model (guess what).

Capabilities:
- Expectation Maximization
- Viterbi
- Prediction of the next symbol
- Cross-entropy computation and cross-validation

It is written in C++ with the computation intensive task written at lowest possible level.
Array operations can exploit 3 different computation cores:
1. framework Accelerate (OSX)
2. Eigen (Cross-platform)
3. Unrolled loops (Cross-platform)

For the particular application it seems that 1. and 3. greatly outperform 2.

The library can easily handle datasets of millions of sequences of different lengths, with minimum memory footprint.

Facilities are provided for training hmm model for prediction of the next observed symbol, using k-fold cross-validation. Model parameters can be saved and restored.
Thread parallelization is used and can lead to x2 and higher speedup when using multi-core cpu.

Examples are momentarily provided in the test folder as test cases. The file HMMTest.cpp is thoroughly commented, better start from that.
