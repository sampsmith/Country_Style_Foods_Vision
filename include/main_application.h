#ifndef MAIN_APPLICATION_H
#define MAIN_APPLICATION_H

#include <memory>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include "vision_pipeline.h"
#include "camera_interface.h"

namespace country_style {

class MainApplication {
public:
    MainApplication();
    ~MainApplication();
    
    bool initialize();
    void run();
    void shutdown();
    
private:
    // GUI state
    GLFWwindow* window_;
    bool show_teach_mode_;
    bool show_config_editor_;
    bool show_recipe_manager_;
    bool show_performance_stats_;
    
    // Vision pipeline
    std::unique_ptr<VisionPipeline> vision_pipeline_;
    std::unique_ptr<CameraInterface> camera_;
    
    // Current frame and results
    cv::Mat current_frame_;
    DetectionResult last_result_;
    
    // ROI drawing state
    bool drawing_roi_;
    cv::Point roi_start_;
    cv::Point roi_end_;
    
    // OpenGL textures for display
    GLuint camera_texture_;
    GLuint segmented_texture_;
    
    // Application state
    bool is_running_;
    bool camera_active_;
    std::string config_path_;
    
    // GUI rendering functions
    void renderMainMenuBar();
    void renderLiveView();
    void renderTeachMode();
    void renderConfigEditor();
    void renderRecipeManager();
    void renderPerformanceStats();
    
    // Helper functions
    void updateCameraTexture(const cv::Mat& frame);
    void handleMouseInput();
    void loadConfig(const std::string& path);
    void saveConfig(const std::string& path);
};

} // namespace country_style

#endif // MAIN_APPLICATION_H
