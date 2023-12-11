
# Toolchain
To use the llvm toolchain in vscode
place the file 'cmake-kits.json' into your '.vscode' directory.
Then select the 'LLVM Kit' in the cmake-tools extension.
Restarting vscode may be necessary.

# Debugging
To enable debugging add this to your .vscode/settings.json
```
"cmake.debugConfig": {
        "MIMode": "gdb",
        "miDebuggerPath": "/usr/bin/gdb",
        "args": [
            // Config for debugging
            "--pipes=/tmp/msim",
            "--num-entities=1000000",
            "--map-width=10000",
            "--map-height=10000",
            "--interface-range=10.000000",
            "--waypoint-buffer-size=4",
            "--waypoint-buffer-threshold=2",
            "--quadtree-depth=9",
            "--quadtree-node-cap=10",
            "--quadtree-nodes=10"
        ]
    }
```
