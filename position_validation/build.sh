#!/bin/bash

cmake -S . -B build -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/clang_toolchain.cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release

cmake --build build
