//
//  FileIO.cpp
//  DBNModels
//
//  Created by Bruno Di Giorgi on 10/06/16.
//
//

#include "FileIO.hpp"
#include <cassert>

namespace bdg {
    
    FileIO::FileIO(){
        checkEndianness();
    }
    
    void FileIO::checkEndianness(){
        int a = 0x12345678;
        unsigned char *c = (unsigned char*)(&a);
        isLittleEndian = (*c == 0x78);
    }
    
    void FileIO::readUInt32(FILE * fp, uint32_t * out){
        // assume the unsigned int is stored in little endian order
        readFourBytes(fp, out);
    }
    
    void FileIO::readUInt32Array(FILE * fp, uint32_t * out, uint32_t size){
        if(isLittleEndian){
            size_t r = fread(out, 4, size, fp);
            assert(r == size);
        }
        else{
            for (int i = 0; i < size; i++) {
                readUInt32(fp, out + i);
            }
        }
    }
    
    void FileIO::writeUInt32(FILE * fp, uint32_t in){
        // write in into fp in little endian order
        writeFourBytes(fp, &in);
    }
    
    void FileIO::writeUInt32Array(FILE * fp, uint32_t * in, uint32_t len){
        if(isLittleEndian){
            size_t w = fwrite(in, 4, len, fp);
            assert(w == len);
        }
        else{
            for(int i = 0; i < len; i++){
                writeUInt32(fp, in[i]);
            }
        }
    }
    
    void FileIO::readDouble(FILE * fp, double * out){
        // assume the unsigned int is stored in little endian order
        readEightBytes(fp, out);
    }
    
    void FileIO::readDoubleArray(FILE * fp, double * out, uint32_t size){
        if(isLittleEndian){
            size_t r = fread(out, 8, size, fp);
            assert(r == size);
        }
        else{
            for (int i = 0; i < size; i++) {
                readDouble(fp, out + i);
            }
        }
    }
    
    void FileIO::writeDouble(FILE * fp, double in){
        writeEightBytes(fp, &in);
    }
    
    void FileIO::writeDoubleArray(FILE * fp, double * in, uint32_t len){
        if(isLittleEndian){
            size_t w = fwrite(in, 8, len, fp);
            assert(w == len);
        }
        else{
            for(int i = 0; i < len; i++){
                writeDouble(fp, in[i]);
            }
        }
    }

    
    size_t FileIO::readFourBytes(FILE * fp, void * buffer){
        size_t r = fread(buffer, 4, 1, fp);
        if(!isLittleEndian){
            uint32_t * b = (uint32_t *)buffer;
            *b = SwapFourBytes(*b);
        }
        return r;
    }
    
    size_t FileIO::readEightBytes(FILE * fp, void * buffer){
        size_t r = fread(buffer, 8, 1, fp);
        if(!isLittleEndian){
            uint64_t * b = (uint64_t *)buffer;
            *b = SwapEightBytes(*b);
        }
        return r;
    }
    
    size_t FileIO::writeFourBytes(FILE * fp, void * buffer){
        if(!isLittleEndian){
            uint32_t * b = (uint32_t *)buffer;
            *b = SwapFourBytes(*b);
        }
        size_t w = fwrite(buffer, 4, 1, fp);
        return w;
    }
    
    size_t FileIO::writeEightBytes(FILE * fp, void * buffer){
        if(!isLittleEndian){
            uint64_t * b = (uint64_t *)buffer;
            *b = SwapEightBytes(*b);
        }
        size_t w = fwrite(buffer, 8, 1, fp);
        return w;
    }
}