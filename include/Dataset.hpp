//
//  Dataset.hpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 13/06/16.
//
//

#ifndef Dataset_hpp
#define Dataset_hpp

#include <stdio.h>
#include <string>

namespace bdg {
    
    class Dataset {
    public:
        Dataset(std::string filename);
        ~Dataset();
        
        void print();
        void print_symbol_count();
        
        std::string filename;
        int alphabet_size {0};
        uint32_t nseq {0};
        uint32_t tot_len {0};
        uint32_t max_len {0};
        uint32_t* slen;
        uint32_t* sind;
        uint32_t* count_symbol;
        uint32_t* y;
        
    };
}

#endif /* Dataset_hpp */
