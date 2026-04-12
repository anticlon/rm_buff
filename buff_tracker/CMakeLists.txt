cmake_minimum_required(VERSION 3.10)
project(buff_tracker LANGUAGES CXX )

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Export compile commands for clangd

# 条件编译
set(BAYER_IMAGE off)  # 接收图像格式 on:Bayer格式 off:RGB格式
set(DETAIL off)       # 显示更多调试信息

# find dependencies
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
find_package(Eigen3 REQUIRED)
# find_package(CUDA REQUIRED)

find_package(OpenCV REQUIRED)

# 配合小电脑上Opencv环境
# find_package(OpenCV)
# set(OpenCV_LIBS 
#     /lib/aarch64-linux-gnu/libopencv_core.so.4.9.0
#     /lib/aarch64-linux-gnu/libopencv_imgproc.so.4.9.0
#     /lib/aarch64-linux-gnu/libopencv_calib3d.so.4.9.0
# )
# # 验证路径是否存在
# foreach(LIB ${OpenCV_LIBS})
#     if(NOT EXISTS ${LIB})
#         message(FATAL_ERROR "OpenCV library not found: ${LIB}")
#     endif()
# endforeach()

include_directories(${EIGEN3_INCLUDE_DIRS})

file(GLOB_RECURSE CPP_SOURCE_FILES src/*.*)
ament_auto_add_library(${PROJECT_NAME} SHARED ${CPP_SOURCE_FILES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 72)

target_compile_definitions(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:DEBUG=1>")
target_compile_definitions(${PROJECT_NAME} PUBLIC "$<$<NOT:$<CONFIG:Debug>>:DEBUG=0>")
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if (BAYER_IMAGE)
  target_compile_definitions(${PROJECT_NAME} PUBLIC BAYER_IMAGE=1)
else()
  target_compile_definitions(${PROJECT_NAME} PUBLIC BAYER_IMAGE=0)
endif()

if (DETAIL)
  target_compile_definitions(${PROJECT_NAME} PUBLIC DETAIL=1)
else()
  target_compile_definitions(${PROJECT_NAME} PUBLIC DETAIL=0)
endif()

# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_72,code=sm_72")
#target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_cuda)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN ${PROJECT_NAME}::SectorFinder
  EXECUTABLE ${PROJECT_NAME}_finder
)

ament_auto_package(
  INSTALL_TO_SHARE
  meshes
)