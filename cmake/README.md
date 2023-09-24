
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
        "miDebuggerPath": "/usr/bin/gdb"
    }
```
