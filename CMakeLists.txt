cmake_minimum_required(VERSION 2.6)
project(fdhmm CXX)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

# output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out)

# cmake modules dir
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.11)
endif()

set(test_path "test")
set(test_files
    ${test_path}/main.cpp
    ${test_path}/HMMTest.cpp
    ${test_path}/CoreTest.cpp
)

set(src_path "src")
set(src_files
    ${src_path}/HMM.cpp
    ${src_path}/HMMTrainer.cpp
    ${src_path}/HMMExperiment.cpp
    ${src_path}/HMMWorkerThread.cpp
    ${src_path}/FileIO.cpp
    ${src_path}/Dataset.cpp
    ${src_path}/Folds.cpp
    ${src_path}/Core.cpp
)

set(include_path "include")
set(include_files
    ${include_path}/fdhmm.hpp
    ${include_path}/HMM.hpp
    ${include_path}/HMMTrainer.hpp
    ${include_path}/HMMExperiment.hpp
    ${include_path}/HMMWorkerThread.hpp
    ${include_path}/FileIO.hpp
    ${include_path}/Dataset.hpp
    ${include_path}/Folds.hpp
    ${include_path}/Core.hpp
)

# Dependencies
find_package(GTest REQUIRED)

# libraries and Frameworks

if(APPLE)
    find_library(frm_Accelerate NAMES Accelerate)
elseif(LINUX)
    find_package(Eigen3)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

set(include_options
    -I${PROJECT_SOURCE_DIR}/${include_path}/
    -I/usr/local/include
    ${GTEST_INCLUDE_DIRS}
)

#project compile options
set(project_compile_options
    ${include_options} -std=gnu++11 -stdlib=libc++)

# library target
# **************

# including include_files is not needed, because cmake automatically generate dependencies
# this is only needed for generating Xcode projects including header files
add_library(fdhmm ${src_files} ${include_files})

target_compile_options(fdhmm PUBLIC ${project_compile_options})

# fdhmm_test
# **************
add_executable(fdhmm_test ${test_files})

target_compile_options(fdhmm_test PUBLIC ${project_compile_options} )

target_link_libraries(fdhmm_test
    ${GTEST_LIBRARIES}
    ${frm_Accelerate}
    fdhmm
)

