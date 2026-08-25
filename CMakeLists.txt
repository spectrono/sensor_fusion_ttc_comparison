cmake_minimum_required(VERSION 4.4.2 FATAL_ERROR)

add_definitions(-std=c++17)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

# On macOS with Homebrew, OpenCV 5 installs its CMake config in /opt/homebrew/opt/opencv/lib/cmake/opencv5
if(NOT OpenCV_DIR AND EXISTS "/opt/homebrew/opt/opencv/lib/cmake/opencv5")
    set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv5")
endif()
find_package(OpenCV 5.0 REQUIRED COMPONENTS core highgui imgproc features xfeatures2d)

# On macOS with Homebrew, yaml-cpp installs its CMake config in /opt/homebrew/opt/yaml-cpp/lib/cmake/yaml-cpp
if(NOT yaml-cpp_DIR AND EXISTS "/opt/homebrew/opt/yaml-cpp/lib/cmake/yaml-cpp")
    set(yaml-cpp_DIR "/opt/homebrew/opt/yaml-cpp/lib/cmake/yaml-cpp")
endif()
find_package(yaml-cpp REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS} src)
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable (3D_object_tracking 
    src/camFusion_Student.cpp 
    src/FinalProject_Camera.cpp 
    src/lidarData.cpp 
    src/objectDetection2D.cpp
)
target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES} yaml-cpp::yaml-cpp)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS} src src/keypoint_lib)