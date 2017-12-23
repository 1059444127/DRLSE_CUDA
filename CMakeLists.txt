cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(DRLSE_CUDA LANGUAGES CXX CUDA)

# QT5
find_package(Qt5Widgets)

# VTK
set(VTK_DIR "C:/Users/Daniel/Dropbox/C++/Resources/VTK-8.0.1-build/" CACHE PATH directory FORCE)
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
#message(${VTK_LIBRARIES})

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
#message(STATUS "CUDA_LIBRARIES: ${CUDA_INCLUDE_DIRS} ${CUDA_LIBRARIES}")

# Adds all the desired files to their lists
set(SOURCES
    main.cpp
    mainwindow.cpp
    cudadriver.cpp
    )

set(KERNELS
    kernelOnly.cu
    )

set(HEADERS
    mainwindow.h
    cudadriver.h
    )

set(UI
    mainwindow.ui
    )

set(RESOURCES
    )

# Processes Qt files
QT5_WRAP_CPP(HEADERS_MOC ${HEADERS})
QT5_WRAP_UI(UI_MOC ${UI})
QT5_ADD_RESOURCES(RESOURCES_RCC ${RESOURCES})

# Include bin directories so we can find MOC'd stuff later
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories (${PROJECT_SOURCE_DIR})
include_directories (${PROJECT_BINARY_DIR})

# Compile our CUDA kernel library
list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_61,code=sm_61)
add_library(CudaLib STATIC ${KERNELS})
target_compile_features(CudaLib PUBLIC cxx_std_11)
set_target_properties(CudaLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(CudaLib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Compile Qt+VTK+Cpp into an executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS_MOC} ${UI_MOC} ${RESOURCES_RCC})
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link libraries to the executable
target_link_libraries(${PROJECT_NAME}
    CudaLib
    Qt5::Widgets
    ${VTK_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    )
