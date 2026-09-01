cmake_minimum_required(VERSION 3.15)

project(ThePankeyEngine_Library_File)

if(NOT TARGET ThePankeyEngine_Library_File)

    add_library(ThePankeyEngine_Library_File INTERFACE)

    target_include_directories(
        ThePankeyEngine_Library_File
          INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}
    )

endif()
