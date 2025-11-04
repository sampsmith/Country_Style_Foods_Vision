cmake_minimum_required(VERSION 3.10)
project(CountryStyleDoughInspector VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Performance optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -msse4.2")  # SIMD support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")  # Link-time optimization

# Find packages
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/imgui
    ${PROJECT_SOURCE_DIR}/external/imgui/backends
    ${OpenCV_INCLUDE_DIRS}
)

# ImGui sources
set(IMGUI_DIR ${PROJECT_SOURCE_DIR}/external/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Vision core sources
set(VISION_SOURCES
    src/vision/fast_color_segmentation.cpp
    src/vision/simd_hsv_convert.cpp
    src/vision/vision_pipeline.cpp
    src/vision/rule_engine.cpp
    src/vision/contour_detector.cpp
    src/vision/config_manager.cpp
    src/vision/camera_interface.cpp
    src/vision/recipe_manager.cpp
)

# GUI sources
set(GUI_SOURCES
    src/gui/teach_mode_window.cpp
    src/gui/live_inference_window.cpp
    src/gui/config_editor_window.cpp
    src/gui/recipe_manager_window.cpp
    src/gui/main_application.cpp
)

# Main executable (polygon teaching version)
add_executable(country_style_inspector
    src/gui/polygon_teaching_app.cpp
    ${VISION_SOURCES}
    ${IMGUI_SOURCES}
)

# Link libraries
target_link_libraries(country_style_inspector
    ${OpenCV_LIBS}
    OpenGL::GL
    glfw
    dl
    pthread
)

# Install target
install(TARGETS country_style_inspector
    RUNTIME DESTINATION bin
)
