#!/bin/bash
set -e

# Build several release builds of msim with differing configuration

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR"/build -G Ninja \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++ \
    -DMSIM_BENCHMARK=1 \
    -DMSIM_STANDALONE=0 \
    -DMSIM_CONNECTIVITY_DETECTION=1 \
    -DMSIM_COPY_REGIONS=1
cmake --build "$SCRIPT_DIR"/build --config Release
cp "$SCRIPT_DIR/msim" "$SCRIPT_DIR/msim_cpu_std"

cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR"/build -G Ninja \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++ \
    -DMSIM_BENCHMARK=1 \
    -DMSIM_STANDALONE=0 \
    -DMSIM_CONNECTIVITY_DETECTION=2 \
    -DMSIM_COPY_REGIONS=1
cmake --build "$SCRIPT_DIR"/build --config Release
cp "$SCRIPT_DIR/msim" "$SCRIPT_DIR/msim_cpu_emil"

cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR"/build -G Ninja \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++ \
    -DMSIM_BENCHMARK=1 \
    -DMSIM_STANDALONE=0 \
    -DMSIM_CONNECTIVITY_DETECTION=3 \
    -DMSIM_COPY_REGIONS=1
cmake --build "$SCRIPT_DIR"/build --config Release
cp "$SCRIPT_DIR/msim" "$SCRIPT_DIR/msim_cpu"

cmake -S "$SCRIPT_DIR" -B "$SCRIPT_DIR"/build -G Ninja \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++ \
    -DMSIM_BENCHMARK=1 \
    -DMSIM_STANDALONE=0 \
    -DMSIM_CONNECTIVITY_DETECTION=4 \
    -DMSIM_COPY_REGIONS=1
cmake --build "$SCRIPT_DIR"/build --config Release
cp "$SCRIPT_DIR/msim" "$SCRIPT_DIR/msim_gpu"
