# Country Style Dough Inspector

High-performance C++ vision inspection system with polygon-based teaching for dough quality control at Country Style Foods.

## Features

- **Polygon Teaching Interface**: Intuitive click-to-draw annotation for teaching good/bad samples
- **Real-time Inference**: Fast HSV-based color segmentation with contour detection
- **ROI Support**: Define regions of interest for focused inspection
- **SIMD Optimization**: AVX2 vectorization for high-speed image processing (target <10ms per frame)
- **Interactive GUI**: Dear ImGui-based interface with OpenGL rendering
- **Flexible Visualization**: Toggle bounding boxes, contours, and mask overlays

## Architecture

Pure C++ application built with:
- **OpenCV 4.6+**: Computer vision algorithms
- **Dear ImGui**: Immediate-mode GUI
- **GLFW3 + OpenGL 3.3**: Window management and rendering
- **AVX2 SIMD**: Hardware-accelerated HSV conversion

## Building

### Linux Prerequisites

```bash
sudo apt-get install -y \
    build-essential cmake \
    libopencv-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    nlohmann-json3-dev \
    zenity
```

### Linux Build

```bash
./build.sh
```

### Windows Prerequisites

1. **Visual Studio 2019 or 2022** (with C++ desktop development workload) or **MinGW**
2. **CMake** (3.10 or later) - [Download](https://cmake.org/download/)
3. **OpenCV** - Install via vcpkg or download pre-built binaries
   - Using vcpkg: `vcpkg install opencv:x64-windows`
   - Or download from [OpenCV releases](https://opencv.org/releases/)
4. **GLFW3** - Install via vcpkg or download from [GLFW website](https://www.glfw.org/download.html)
   - Using vcpkg: `vcpkg install glfw3:x64-windows`
5. **Git** (for downloading ImGui automatically)

### Windows Build

Option 1: Using PowerShell (Recommended)
```powershell
.\build.ps1
```

Option 2: Using Batch file
```cmd
build.bat
```

Option 3: Manual CMake build
```cmd
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

The build script automatically downloads Dear ImGui v1.90.4 and compiles with platform-appropriate optimization flags.

## Usage

### Run

**Linux:**
```bash
./build/country_style_inspector
```

**Windows:**
```cmd
.\build\Release\country_style_inspector.exe
```
Or if using MinGW:
```cmd
.\build\country_style_inspector.exe
```

### Teach Mode Workflow

1. **Load Training Image**: Click "Load Training Image" or File → Load Image
2. **Optional: Draw ROI**: 
   - Check "Enable ROI"
   - Click-drag on image to define rectangular region
   - Uncheck "Enable ROI" to proceed with polygon drawing
3. **Annotate Samples**:
   - Ensure "Good Sample (Green)" is checked for dough pieces
   - Uncheck for background/defects
   - Left-click to add polygon points
   - Right-click or Enter to close polygon
   - Repeat for multiple samples
4. **Learn**: Click "Learn from Polygons" to train the detector
5. **Switch to Inference Mode**: Mode → Inference Mode

### Inference Mode Workflow

1. **Load Test Image**: Click "Load Test Image"
2. **Optional: Set ROI**: Enable and draw ROI if needed
3. **Configure Display**:
   - Toggle "Show Bounding Boxes" (red rectangles with labels)
   - Toggle "Show Contours" (green outlines)
   - Toggle "Show Mask Overlay" (cyan segmentation)
4. **Run Detection**: Click "Run Detection"
5. **Review Results**: View detected count, performance metrics, and status
6. **Save**: Click "Save Result Image" to export annotated result

## Key Shortcuts

- **Ctrl+O**: Load image
- **Ctrl+S**: Save annotations
- **F1**: Toggle help overlay
- **Esc**: Cancel polygon drawing / Exit
- **Enter**: Complete polygon

## Technical Details

### Vision Pipeline

1. **Color Space Conversion**: BGR → HSV with SIMD acceleration
2. **Color Segmentation**: HSV range-based thresholding
3. **Morphological Operations**: Noise removal and blob enhancement
4. **Contour Detection**: Connected component analysis
5. **Rule-Based Filtering**: Area, circularity, and aspect ratio constraints

### Performance

- **Target**: <10ms per frame
- **Optimizations**:
  - AVX2 intrinsics for HSV conversion
  - Pre-allocated memory buffers
  - Link-time optimization (LTO)
  - Native CPU architecture tuning

### Learning Algorithm

Teaches from polygon annotations:
- **HSV ranges**: Mean ± 30 from good samples (with minimum saturation/value thresholds)
- **Area rules**: 0.5x to 2x the drawn polygon sizes
- **Shape tolerance**: Relaxed circularity (0.2-1.0) and aspect ratio (0.3-3.0)

## Project Structure

```
CountryStyleDoughInspector/
├── build.sh                    # Build script with auto-dependency download
├── CMakeLists.txt              # CMake configuration
├── config/
│   └── default_config.json     # Default detection parameters
├── include/                    # Header files
│   ├── simd_hsv_convert.h
│   ├── fast_color_segmentation.h
│   ├── vision_pipeline.h
│   ├── main_application.h
│   ├── rule_engine.h
│   ├── contour_detector.h
│   ├── config_manager.h
│   └── camera_interface.h
├── src/
│   ├── vision/                 # Vision processing modules
│   │   ├── simd_hsv_convert.cpp
│   │   ├── fast_color_segmentation.cpp
│   │   ├── vision_pipeline.cpp
│   │   ├── rule_engine.cpp
│   │   ├── contour_detector.cpp
│   │   ├── config_manager.cpp
│   │   └── camera_interface.cpp
│   └── gui/
│       └── polygon_teaching_app.cpp  # Main GUI application
└── external/
    └── imgui/                  # Auto-downloaded Dear ImGui
```

## Configuration

Edit `config/default_config.json` to adjust:
- Default HSV color ranges
- Morphological kernel sizes
- Detection rule thresholds
- Camera parameters (for future real-time mode)

## Future Enhancements

- Real-time camera integration
- Multi-class detection
- Annotation import/export (JSON)
- Configuration profiles
- Performance profiling overlay
- Batch processing mode

## License

MIT License - See LICENSE file

## Acknowledgments

Built for Country Style Foods dough inspection systems.
