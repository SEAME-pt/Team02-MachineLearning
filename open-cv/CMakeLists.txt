cmake_minimum_required(VERSION 3.10)
project(OpenCV_Display_Image)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(display_image display_image.cpp)

target_link_libraries(display_image ${OpenCV_LIBS})
