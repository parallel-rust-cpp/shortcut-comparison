cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native -Wall -pedantic")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # metadata for text editor
set(EXECUTABLE_NAME "shortcut")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/target)
set(SOURCES main.cpp step.cpp)
add_executable(${EXECUTABLE_NAME} ${SOURCES})
target_include_directories(${EXECUTABLE_NAME} PRIVATE .)
