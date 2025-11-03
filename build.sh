#!/bin/bash

set -e

echo "Country Style Dough Inspector - Build Script"
echo "============================================="

# Clone Dear ImGui if not present
if [ ! -d "external/imgui" ]; then
    echo "Cloning Dear ImGui..."
    mkdir -p external
    cd external
    git clone https://github.com/ocornut/imgui.git
    cd imgui
    git checkout v1.90.4  # Stable version
    cd ../..
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "Build complete!"
echo "Binary: build/country_style_inspector"
echo ""
echo "Run with: ./build/country_style_inspector"
