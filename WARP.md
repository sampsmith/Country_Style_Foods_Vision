# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

High-performance C++ vision inspection system for dough quality control at Country Style Foods. Uses polygon-based teaching interface for annotating training samples, then performs real-time HSV color segmentation with contour detection and rule-based quality validation.

## Build & Development Commands

### Building
```bash
./build.sh
```
The build script automatically downloads Dear ImGui v1.90.4 and compiles with optimization flags (`-O3 -march=native -mavx2 -flto`).

### Prerequisites Installation
```bash
sudo apt-get install -y \
    build-essential cmake \
    libopencv-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    nlohmann-json3-dev \
    zenity
```

### Running the Application
```bash
./build/country_style_inspector
```

### Clean Build
```bash
rm -rf build external/imgui
./build.sh
```

## Architecture Overview

### Core Pipeline Flow
1. **Color Space Conversion** (SIMD-accelerated): BGR → HSV using AVX2 intrinsics (`simd_hsv_convert.cpp`)
2. **Segmentation** (`fast_color_segmentation.cpp`): HSV range-based thresholding with morphological cleanup
3. **Contour Detection** (`contour_detector.cpp`): Connected component analysis and feature extraction
4. **Rule-Based Filtering** (`rule_engine.cpp`): Area, circularity, and aspect ratio constraints
5. **Quality Validation** (`vision_pipeline.cpp`): Per-detection and batch-level pass/fail analysis

### Key Components

**VisionPipeline** (`src/vision/vision_pipeline.cpp`, `include/vision_pipeline.h`)
- Orchestrates the entire detection pipeline with sub-10ms performance target
- Manages ROI processing (filters detections by center point, not image cropping)
- Tracks timing metrics for each stage (segmentation, contours, rules)
- Handles quality thresholds for width, height, count, and shape validation
- Returns `DetectionResult` with fault flags, measurements, and detailed per-detection data

**FastColorSegmentation** (`src/vision/fast_color_segmentation.cpp`)
- Uses `SimdHsvConverter` for hardware-accelerated HSV conversion
- Pre-allocates buffers to avoid memory allocation overhead
- Target: <5ms for 640x480 images

**Teaching System** (`src/gui/polygon_teaching_app.cpp`)
- Two-mode interface: Teach Mode and Inference Mode
- **Teach Mode**: User draws polygons on good/bad samples to generate HSV ranges and detection rules
- **Inference Mode**: Runs detection on test images with configurable visualization options
- Learning algorithm: HSV ranges = mean ± 30 from good samples; Area rules = 0.5x to 2x polygon sizes

**Configuration** (`config/default_config.json`, `config_manager.cpp`)
- JSON-based configuration for HSV ranges, ROI, detection rules, and camera settings
- Use `ConfigManager` class to load/save configurations programmatically

### Performance Optimizations
- **SIMD**: AVX2 intrinsics for HSV conversion (check with `SimdHsvConverter::hasAvx2Support()`)
- **LTO**: Link-time optimization enabled in CMakeLists.txt
- **Pre-allocated Buffers**: Avoid runtime memory allocation in hot paths
- **Native Tuning**: `-march=native` compiles for the local CPU architecture

## Code Patterns & Conventions

### Namespace
All code is in the `country_style` namespace.

### Vision Results Structure
- `DetectionResult`: Top-level pipeline output with contours, bounding boxes, fault flags, and timing
- `DetectionMeasurement`: Per-detection data with dimensions, circularity, and individual pass/fail
- `QualityThresholds`: Configurable thresholds for count, size, and shape validation

### ROI Handling
ROI is applied as a **filter on detection centers**, not as image cropping. The full frame is always segmented, then detections outside the ROI are discarded. This is intentional to avoid re-segmentation overhead.

### Fault Detection Logic
Quality validation happens in `VisionPipeline::processFrame()`:
- Count faults: `fault_count_low`, `fault_count_high`
- Size faults: `fault_undersized`, `fault_oversized` (per-detection)
- Shape faults: `fault_shape_defect` (circularity, aspect ratio)
- Individual measurements have `meets_specs` flag and `fault_reason` string

### GUI Layout
- Main application window uses Dear ImGui immediate-mode paradigm
- OpenGL textures for image display (`image_texture_`, `result_texture_`)
- Menu bar with File, View, Mode options
- Keyboard shortcuts: Ctrl+O (load), Ctrl+S (save), F1 (help), Esc (exit/cancel)

## Modifying Detection Behavior

### Changing HSV Color Range
Edit `config/default_config.json` or use `VisionPipeline::updateColorRange()` at runtime:
```cpp
vision_pipeline_->updateColorRange(cv::Scalar(h_min, s_min, v_min), 
                                    cv::Scalar(h_max, s_max, v_max));
```

### Adjusting Detection Rules
Update `DetectionRules` via `RuleEngine::setRules()` or modify config file:
```cpp
DetectionRules rules;
rules.min_area = 500;
rules.max_area = 50000;
rules.min_circularity = 0.3;
rules.max_circularity = 1.0;
rule_engine_->setRules(rules);
```

### Adding New Measurements
1. Add field to `DetectionMeasurement` struct in `vision_pipeline.h`
2. Calculate measurement in `VisionPipeline::processFrame()` loop (line 104+)
3. Update GUI rendering in `polygon_teaching_app.cpp` to display new field

## File Organization

- `include/`: Header files for all vision and GUI components
- `src/vision/`: Core vision processing (SIMD, segmentation, contours, rules)
- `src/gui/`: ImGui-based GUI applications (teaching interface, config editor)
- `config/`: JSON configuration files
- `external/imgui/`: Auto-downloaded Dear ImGui library (not in repo)

## ImGui Integration

Dear ImGui is downloaded by `build.sh` at v1.90.4. The following backends are used:
- `imgui_impl_glfw.cpp`: GLFW3 window integration
- `imgui_impl_opengl3.cpp`: OpenGL 3.3 rendering

ImGui sources are listed explicitly in CMakeLists.txt and compiled into the executable.

## Dependencies

**Required system packages:**
- OpenCV 4.6+ (cv::Mat, color conversion, morphology)
- GLFW3 (window management)
- OpenGL 3.3 (rendering)
- nlohmann-json (JSON config parsing)
- zenity (file dialogs)

**Build-time downloads:**
- Dear ImGui v1.90.4 (GUI framework)

## Performance Targets

- **Total pipeline**: <10ms per frame
- **Segmentation**: <5ms (SIMD-accelerated)
- **Contour detection**: ~2-3ms
- **Rule application**: <1ms

Monitor actual performance via `DetectionResult` timing fields or `VisionPipeline::getPerformanceStats()`.

## Future Enhancements (from README)

- Real-time camera integration (camera_interface.cpp skeleton exists)
- Multi-class detection
- Annotation import/export (JSON)
- Configuration profiles
- Batch processing mode
