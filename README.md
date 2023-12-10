# GSIM
GSIM stands for GPU-based mobility Simulator.  
It is based on the project [movement-sim](https://github.com/COM8/movement-sim) and  
has been integrated with [The ONE simulator](https://github.com/crydsch/the-one) to accelerate mobility simulations.

## Building
### Requirements

#### Fedora
```
sudo dnf install gtkmm4.0-devel libadwaita-devel libcurl-devel g++ clang cmake git
sudo dnf install mesa-libEGL-devel vulkan-devel glslc vulkan-tools glslang
```

If you want to enable support for debugging Vulkan shaders with [RenderDoc](https://renderdoc.org/) via setting the `MOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API` to `ON` during CMake configuration (e.g. `cmake .. -DMOVEMENT_SIMULATOR_ENABLE_RENDERDOC_API=ON`) the following needs to be installed as well:
```
sudo dnf install renderdoc-devel
```

### Compiling
```
git clone https://github.com/crydsch/gsim gsim
cmake -S gsim -B gsim/build -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=gsim/cmake/clang_toolchain.cmake \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
    -DGSIM_BENCHMARK=1 \
    -DGSIM_STANDALONE=0 \
    -DGSIM_COPY_REGIONS=1
cmake --build gsim/build
```


## Troubleshooting

### Disable GPU Hangcheck on Intel
Source: https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-hpc-linux/top/before-you-begin.html

```
grub2-mkconfig -o "$(readlink -e /etc/grub2.cfg)"
```

Intel Bug: https://gitlab.freedesktop.org/drm/intel/-/issues/562
