//
//  Dataset.cpp
//  bdg_hmm
//
//  Created by Bruno Di Giorgi on 13/06/16.
//
//

#include "Dataset.hpp"
#include "FileIO.hpp"
#include <iostream>
#include <cassert>

namespace bdg {
    Dataset::Dataset(std::string filename_)
    :filename(filename_) {
        FILE* fp = fopen(filename.c_str(), "r");
        assert(fp != nullptr);
        
        FileIO fio;
        
        // how many sequences
        fio.readUInt32(fp, &nseq);
        slen = new uint32_t[nseq];
        sind = new uint32_t[nseq];
        
        // length of the sequences
        fio.readUInt32Array(fp, slen, nseq);
        max_len = *(std::max_element(slen, slen + nseq));
        
        
        // how many elements in total
        tot_len = 0;
        for(int iseq = 0; iseq < nseq; iseq++) {
            sind[iseq] = tot_len;
            tot_len += slen[iseq];
        }
        
        // read elements
        y = new uint32_t[tot_len];
        uint32_t* ptr = y;
        uint32_t max = 0;
        for(int iseq = 0; iseq < nseq; iseq++) {
            fio.readUInt32Array(fp, ptr, slen[iseq]);
            
            uint32_t* max_t = std::max_element(ptr, ptr + slen[iseq]);
            max = max > *max_t ? max : *max_t;
            
            ptr += slen[iseq];
        }
        
        alphabet_size = 1 + max;  // remember the 0 symbol
        count_symbol = new uint32_t[alphabet_size];
        memset(count_symbol, 0, alphabet_size * sizeof(uint32_t));
        
        for (int i = 0; i < tot_len; i++) {
            count_symbol[y[i]] += 1;
        }
        
        fclose(fp);
    }
    
    Dataset::~Dataset() {
        delete[] slen;
        delete[] sind;
        delete[] y;
        delete[] count_symbol;
    }
    
    void Dataset::print() {
        std::cout << "*** Dataset ***" << std::endl;
        std::cout << "n sequences: " << nseq << std::endl;
        std::cout << "longest sequences: " << max_len << std::endl;
        std::cout << "total length: " << tot_len << std::endl;
        std::cout << "total length: " << tot_len << std::endl;
        std::cout << std::endl;
    }
    
    void Dataset::print_symbol_count() {
        for (int i = 0; i < alphabet_size; i++) {
            std::cout << i << ": " << count_symbol[i] << std::endl;
        }
    }
}