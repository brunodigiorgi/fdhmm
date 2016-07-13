//
//  FileIO.hpp
//  DBNModels
//
//  Created by Bruno Di Giorgi on 10/06/16.
//
//

#ifndef FileIO_hpp
#define FileIO_hpp

#include <stdio.h>
#include <cstdint>

#define SwapTwoBytes(data) \
( (((data) >> 8) & 0x00FF) | (((data) << 8) & 0xFF00) )

#define SwapFourBytes(data)   \
( (((data) >> 24) & 0x000000FF) | (((data) >>  8) & 0x0000FF00) | \
(((data) <<  8) & 0x00FF0000) | (((data) << 24) & 0xFF000000) )

#define SwapEightBytes(data)   \
( (((data) >> 56) & 0x00000000000000FF) | (((data) >> 40) & 0x000000000000FF00) | \
(((data) >> 24) & 0x0000000000FF0000) | (((data) >>  8) & 0x00000000FF000000) | \
(((data) <<  8) & 0x000000FF00000000) | (((data) << 24) & 0x0000FF0000000000) | \
(((data) << 40) & 0x00FF000000000000) | (((data) << 56) & 0xFF00000000000000) )

namespace bdg {
    class FileIO {
        
        bool isLittleEndian;
        size_t readFourBytes(FILE * fp, void * buffer);
        size_t readEightBytes(FILE * fp, void * buffer);
        size_t writeFourBytes(FILE * fp, void * buffer);
        size_t writeEightBytes(FILE * fp, void * buffer);
        
    public:
        
        FileIO();
        void checkEndianness();
        
        void readUInt32(FILE * fp, uint32_t * out);
        void readUInt32Array(FILE * fp, uint32_t * out, uint32_t size);
        void writeUInt32(FILE * fp, uint32_t in);
        void writeUInt32Array(FILE * fp, uint32_t * in, uint32_t len);
        
        void readDouble(FILE * fp, double * out);
        void readDoubleArray(FILE * fp, double * out, uint32_t size);
        void writeDouble(FILE * fp, double in);
        void writeDoubleArray(FILE * fp, double * in, uint32_t len);
        
    };
}

#endif /* FileIO_hpp */
