cmake_minimum_required(VERSION 3.10)
project(system_monitor C)

set(CMAKE_C_STANDARD 11)

add_executable(telemetryd
    src/main.c
)

find_library(PCI_LIB pci)
find_library(NVML_LIB nvidia-ml)
find_library(ROCM_SMI rocm_smi64)

if (NVML_LIB)
    target_link_libraries(telemetryd PRIVATE ${NVML_LIB})
    target_compile_definitions(telemetryd PRIVATE USE_NVIDIA)
endif()

if (ROCM_SMI)
    target_link_libraries(telemetryd PRIVATE ${ROCM_SMI})
    target_compile_definitions(telemetryd PRIVATE USE_ROCM)
endif()

if (PCI_LIB)
    target_link_libraries(telemetryd PRIVATE ${PCI_LIB})
    target_compile_definitions(telemetryd PRIVATE USE_PCI)
endif()
