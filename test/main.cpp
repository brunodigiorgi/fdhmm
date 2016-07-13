//
//  main.cpp
//  PitchDetector_test
//
//  Created by Bruno Di Giorgi on 20/01/15.
//  Copyright (c) 2015 Subsequently and Furthermore, Inc. All rights reserved.
//

#include "gtest/gtest.h"

int main(int argc, char * argv[]) {
    
    testing::InitGoogleTest(&argc, argv);
    testing::FLAGS_gtest_filter = "HMMTest.trainer";
    assert(RUN_ALL_TESTS() == 0);
    
    return 0;
}