cmake_minimum_required(VERSION 3.10.0)
project(YoloServer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-win-x64-gpu")
set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

find_package(OpenCV REQUIRED)

include_directories("include/")

# Add YoloServer
add_executable(YoloServer
        src/yolo_server.cpp)

target_include_directories(YoloServer PRIVATE "${ONNXRUNTIME_DIR}/include")
target_link_libraries(YoloServer ${OpenCV_LIBS})

if (WIN32)
    message(STATUS "We are building on Windows!")
    # Specific Windows build commands or flags
    target_link_libraries(YoloServer
            ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib
            debug ${LIB_DIR}/cryptlibd.lib
            optimized ${LIB_DIR}/cryptlib.lib
    )

    target_compile_options(YoloServer PRIVATE /source-charset:utf-8)
    target_compile_definitions(YoloServer PRIVATE UNICODE _UNICODE)

    # 获取所有 DLL 文件列表（适用于 Windows）
    file(GLOB RUNTIME_DLLS "${CMAKE_CURRENT_SOURCE_DIR}/dll/*.dll")
    # 在可执行文件生成后，复制 DLL 文件到输出目录
    add_custom_command(TARGET YoloServer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${RUNTIME_DLLS}
            $<TARGET_FILE_DIR:YoloServer>
            COMMENT "Copying Runtime DLLs to output directory"
    )
endif (WIN32)
