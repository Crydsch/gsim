# Movement Simulator
GPU accelerated human movement simulator.

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
git clone https://github.com/crydsch/movement-sim.git msim
cmake -S msim -B msim/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++
cmake --build msim/build
```

### Installing
```
sudo cmake --build msim/build --target install
```


## Troubleshooting

### Disable GPU Hangcheck on Intel
Source: https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-intel-oneapi-hpc-linux/top/before-you-begin.html

```
grub2-mkconfig -o "$(readlink -e /etc/grub2.cfg)"
```

Intel Bug: https://gitlab.freedesktop.org/drm/intel/-/issues/562
