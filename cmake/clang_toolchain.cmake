# This toolchain configuration file can be used for llvm based compilation

MESSAGE("Compiling with toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

# Define the environment for compiling with 64-bit clang
set(CMAKE_C_COMPILER     /usr/bin/clang       )
set(CMAKE_CXX_COMPILER   /usr/bin/clang++     )
set(CMAKE_RC_COMPILER    /usr/bin/llvm-windres)
set(CMAKE_RANLIB         /usr/bin/llvm-ranlib )
set(CMAKE_AR             /usr/bin/llvm-ar     )
set(CMAKE_STRIP          /usr/bin/llvm-strip  )

# Note: Despite its name '-static-libstdc++' the option causes libc++ to be linked statically
set(CMAKE_CXX_FLAGS_INIT           "-stdlib=libc++"                )
set(CMAKE_EXE_LINKER_FLAGS_INIT    "-fuse-ld=lld -static-libstdc++")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld -static-libstdc++")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld -static-libstdc++")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE    TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
