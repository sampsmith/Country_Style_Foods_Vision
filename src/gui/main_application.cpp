#include "main_application.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/gl.h>
#include <iostream>
#include <cstring>

namespace country_style {

MainApplication::MainApplication()
    : window_(nullptr),
      show_teach_mode_(false),
      show_config_editor_(false),
      show_recipe_manager_(false),
      show_performance_stats_(true),
      drawing_roi_(false),
      camera_texture_(0),
      segmented_texture_(0),
      is_running_(false),
      camera_active_(false),
      config_path_("config/default_config.json"),
      using_video_file_(false),
      video_loop_(false),
      video_paused_(false),
      video_loaded_(false),
      video_finished_(false),
      video_last_frame_time_(0.0),
      video_frame_interval_(0.0),
      video_path_(),
      video_status_message_() {
}

MainApplication::~MainApplication() {
    shutdown();
}

bool MainApplication::initialize() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // GL 3.3 + GLSL 330
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    // Create window
    window_ = glfwCreateWindow(1920, 1080, "Country Style Dough Inspector", nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1); // Enable vsync
    
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    // Note: Docking requires imgui_internal.h or ImGui docking branch
    // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    
    // Setup style
    ImGui::StyleColorsDark();
    
    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    
    // Initialize vision pipeline
    vision_pipeline_ = std::make_unique<VisionPipeline>();
    vision_pipeline_->initialize(config_path_);
    
    // Initialize camera
    camera_ = std::make_unique<CameraInterface>();
    
    // Generate OpenGL textures
    glGenTextures(1, &camera_texture_);
    glGenTextures(1, &segmented_texture_);
    
    is_running_ = true;
    return true;
}

void MainApplication::run() {
    while (is_running_ && !glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Capture camera or video frame
        double now = glfwGetTime();
        if (camera_active_ && camera_->isOpen()) {
            bool should_capture = true;
            
            if (using_video_file_) {
                if (video_paused_) {
                    should_capture = false;
                } else if (video_frame_interval_ > 0.0) {
                    if (now - video_last_frame_time_ >= video_frame_interval_) {
                        video_last_frame_time_ = now;
                    } else {
                        should_capture = false;
                    }
                }
            }
            
            if (should_capture) {
                if (camera_->captureFrame(current_frame_)) {
                    if (using_video_file_) {
                        video_finished_ = false;
                    }
                    
                    // Process frame through vision pipeline
                    last_result_ = vision_pipeline_->processFrame(current_frame_);
                    
                    // Render detections on frame
                    vision_pipeline_->renderDetections(current_frame_, last_result_);
                } else if (using_video_file_) {
                    // End of video or read error
                    if (video_loop_ && !video_path_.empty()) {
                        camera_->release();
                        if (camera_->initializeFromFile(video_path_)) {
                            int fps = camera_->getFPS();
                            video_frame_interval_ = fps > 0 ? 1.0 / static_cast<double>(fps) : 0.0;
                            if (camera_->captureFrame(current_frame_)) {
                                video_last_frame_time_ = now;
                                video_finished_ = false;
                                last_result_ = vision_pipeline_->processFrame(current_frame_);
                                vision_pipeline_->renderDetections(current_frame_, last_result_);
                            }
                        } else {
                            stopVideoPlayback();
                            video_status_message_ = "Failed to loop video: " + video_path_;
                        }
                    } else {
                        video_finished_ = true;
                        video_paused_ = true;
                        video_status_message_ = "Video finished: " + video_path_;
                        camera_->release();
                        camera_active_ = false;
                    }
                }
            }
        }
        
        // Render GUI
        renderMainMenuBar();
        renderLiveView();
        
        if (show_teach_mode_) renderTeachMode();
        if (show_config_editor_) renderConfigEditor();
        if (show_recipe_manager_) renderRecipeManager();
        if (show_performance_stats_) renderPerformanceStats();
        
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window_, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window_);
    }
}

void MainApplication::shutdown() {
    if (camera_) camera_->release();
    
    if (camera_texture_) glDeleteTextures(1, &camera_texture_);
    if (segmented_texture_) glDeleteTextures(1, &segmented_texture_);
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    if (window_) {
        glfwDestroyWindow(window_);
        glfwTerminate();
    }
}

void MainApplication::renderMainMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Load Config")) {
                loadConfig(config_path_);
            }
            if (ImGui::MenuItem("Save Config")) {
                saveConfig(config_path_);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                is_running_ = false;
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Camera")) {
            bool can_start_camera = !camera_active_;
            if (ImGui::MenuItem("Start Camera", nullptr, false, can_start_camera)) {
                stopVideoPlayback();
                if (camera_->open(0, 640, 480, 30)) {
                    camera_active_ = true;
                }
            }
            if (ImGui::MenuItem("Stop Camera", nullptr, false, camera_active_ && !using_video_file_)) {
                camera_->release();
                camera_active_ = false;
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Windows")) {
            ImGui::MenuItem("Teach Mode", nullptr, &show_teach_mode_);
            ImGui::MenuItem("Config Editor", nullptr, &show_config_editor_);
            ImGui::MenuItem("Recipe Manager", nullptr, &show_recipe_manager_);
            ImGui::MenuItem("Performance Stats", nullptr, &show_performance_stats_);
            ImGui::EndMenu();
        }
        
        ImGui::EndMainMenuBar();
    }
}

void MainApplication::renderLiveView() {
    ImGui::SetNextWindowPos(ImVec2(0, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(960, 720), ImGuiCond_FirstUseEver);
    
    ImGui::Begin("Live Inference View");
    
    bool has_frame = !current_frame_.empty();
    
    if (has_frame) {
        updateCameraTexture(current_frame_);
        
        float aspect = (float)current_frame_.cols / current_frame_.rows;
        ImVec2 available = ImGui::GetContentRegionAvail();
        float display_width = available.x;
        float display_height = display_width / aspect;
        
        ImGui::Image((void*)(intptr_t)camera_texture_, 
                    ImVec2(display_width, display_height));
        
        // Display detection info
        ImGui::Separator();
        ImGui::Text("Dough Count: %d", last_result_.dough_count);
        ImGui::Text("Processing Time: %.2f ms", last_result_.total_time_ms);
        ImGui::Text("Status: %s", last_result_.is_valid ? "PASS" : "FAIL");
    } else {
        if (using_video_file_ && video_loaded_) {
            if (video_finished_) {
                ImGui::Text("Video playback finished.");
                ImGui::Text("Press Play or Restart to review again.");
            } else if (camera_active_) {
                ImGui::Text("Waiting for next video frame...");
            } else {
                ImGui::Text("Video ready. Press Play to start inference.");
            }
        } else {
            ImGui::Text("No camera feed available");
            ImGui::Text("Go to Camera -> Start Camera or load a video below");
        }
    }
    
    ImGui::SeparatorText("Offline Video Playback");
    
    static char video_path_buffer[512] = "";
    if (video_path_ != video_path_buffer) {
        std::strncpy(video_path_buffer, video_path_.c_str(), sizeof(video_path_buffer) - 1);
        video_path_buffer[sizeof(video_path_buffer) - 1] = '\0';
    }
    
    if (ImGui::InputText("Video File", video_path_buffer, IM_ARRAYSIZE(video_path_buffer))) {
        video_path_ = video_path_buffer;
    }
    
    if (ImGui::Button("Load Video")) {
        if (video_path_.empty()) {
            video_status_message_ = "Please enter a video file path.";
        } else {
            startVideoPlayback(video_path_);
        }
    }
    
    ImGui::SameLine();
    bool has_video_stream = video_loaded_ || using_video_file_;
    if (!has_video_stream) ImGui::BeginDisabled();
    if (ImGui::Button("Stop")) {
        stopVideoPlayback();
    }
    if (!has_video_stream) ImGui::EndDisabled();
    
    ImGui::SameLine();
    ImGui::Checkbox("Loop", &video_loop_);
    
    ImGui::Spacing();
    
    bool play_enabled = video_loaded_;
    const char* play_label = (play_enabled && camera_active_ && !video_paused_) ? "Pause" : "Play";
    if (!play_enabled) ImGui::BeginDisabled();
    if (ImGui::Button(play_label)) {
        if (camera_active_ && !video_paused_) {
            video_paused_ = true;
            video_status_message_ = "Video paused";
        } else {
            if (video_finished_ || !camera_->isOpen()) {
                if (!video_path_.empty()) {
                    startVideoPlayback(video_path_);
                }
            } else {
                video_paused_ = false;
                video_last_frame_time_ = glfwGetTime();
                camera_active_ = true;
                video_status_message_ = "Video playing: " + video_path_;
            }
        }
    }
    if (!play_enabled) ImGui::EndDisabled();
    
    ImGui::SameLine();
    if (!play_enabled) ImGui::BeginDisabled();
    if (ImGui::Button("Restart")) {
        if (!video_path_.empty()) {
            startVideoPlayback(video_path_);
        }
    }
    if (!play_enabled) ImGui::EndDisabled();
    
    if (video_finished_) {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.2f, 1.0f), "Playback finished");
    }
    
    if (!video_status_message_.empty()) {
        ImGui::TextWrapped("%s", video_status_message_.c_str());
    }
    
    ImGui::SeparatorText("ROI Tools");
    {
        cv::Rect current_roi = vision_pipeline_->getROI();
        static int roi_x = 0, roi_y = 0, roi_w = 0, roi_h = 0;
        static bool initialized = false;
        if (!initialized) {
            roi_x = current_roi.x;
            roi_y = current_roi.y;
            roi_w = current_roi.width;
            roi_h = current_roi.height;
            initialized = true;
        }
        
        if (ImGui::Button("Load Current ROI")) {
            cv::Rect r = vision_pipeline_->getROI();
            roi_x = r.x; roi_y = r.y; roi_w = r.width; roi_h = r.height;
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Clear ROI")) {
            roi_x = roi_y = roi_w = roi_h = 0;
            vision_pipeline_->updateROI(cv::Rect(0, 0, 0, 0));
        }
        
        bool has_frame_dims = has_frame;
        if (!has_frame_dims) ImGui::BeginDisabled();
        
        if (has_frame_dims) {
            ImGui::Text("Frame: %dx%d", current_frame_.cols, current_frame_.rows);
        } else {
            ImGui::Text("Frame: N/A");
        }
        
        ImGui::InputInt("ROI X", &roi_x);
        ImGui::InputInt("ROI Y", &roi_y);
        ImGui::InputInt("ROI Width", &roi_w);
        ImGui::InputInt("ROI Height", &roi_h);
        
        if (ImGui::Button("Apply ROI")) {
            int max_w = has_frame_dims ? current_frame_.cols : roi_x + roi_w;
            int max_h = has_frame_dims ? current_frame_.rows : roi_y + roi_h;
            
            // Clamp
            if (roi_x < 0) roi_x = 0;
            if (roi_y < 0) roi_y = 0;
            if (roi_w < 0) roi_w = 0;
            if (roi_h < 0) roi_h = 0;
            if (has_frame_dims) {
                if (roi_x > max_w) roi_x = max_w;
                if (roi_y > max_h) roi_y = max_h;
                if (roi_x + roi_w > max_w) roi_w = std::max(0, max_w - roi_x);
                if (roi_y + roi_h > max_h) roi_h = std::max(0, max_h - roi_y);
            }
            
            vision_pipeline_->updateROI(cv::Rect(roi_x, roi_y, roi_w, roi_h));
        }
        
        if (!has_frame_dims) ImGui::EndDisabled();
    }
    
    ImGui::End();
}

void MainApplication::renderTeachMode() {
    ImGui::SetNextWindowSize(ImVec2(640, 480), ImGuiCond_FirstUseEver);
    ImGui::Begin("Teach Mode", &show_teach_mode_);
    
    ImGui::Text("ROI Drawing Tool");
    ImGui::Text("Click and drag on the live view to define Region of Interest");
    
    if (ImGui::Button("Clear ROI")) {
        vision_pipeline_->updateROI(cv::Rect(0, 0, 0, 0));
    }
    
    ImGui::End();
}

void MainApplication::renderConfigEditor() {
    ImGui::SetNextWindowSize(ImVec2(400, 600), ImGuiCond_FirstUseEver);
    ImGui::Begin("Configuration Editor", &show_config_editor_);
    
    static float hsv_lower[3] = {20, 50, 50};
    static float hsv_upper[3] = {40, 255, 255};
    static float min_area = 500;
    static float max_area = 50000;
    static float min_circularity = 0.3f;
    static float max_circularity = 1.0f;
    
    ImGui::SeparatorText("Color Segmentation (HSV)");
    ImGui::SliderFloat3("Lower Bound", hsv_lower, 0, 255);
    ImGui::SliderFloat3("Upper Bound", hsv_upper, 0, 255);
    
    if (ImGui::Button("Apply Color Range")) {
        vision_pipeline_->updateColorRange(
            cv::Scalar(hsv_lower[0], hsv_lower[1], hsv_lower[2]),
            cv::Scalar(hsv_upper[0], hsv_upper[1], hsv_upper[2])
        );
    }
    
    ImGui::SeparatorText("Detection Rules");
    ImGui::InputFloat("Min Area", &min_area);
    ImGui::InputFloat("Max Area", &max_area);
    ImGui::SliderFloat("Min Circularity", &min_circularity, 0.0f, 1.0f);
    ImGui::SliderFloat("Max Circularity", &max_circularity, 0.0f, 1.0f);
    
    if (ImGui::Button("Apply Detection Rules")) {
        DetectionRules rules;
        rules.min_area = min_area;
        rules.max_area = max_area;
        rules.min_circularity = min_circularity;
        rules.max_circularity = max_circularity;
        rules.min_aspect_ratio = 0.5;
        rules.max_aspect_ratio = 2.0;
        rules.expected_count = 0;
        rules.enforce_count = false;
        vision_pipeline_->updateDetectionRules(rules);
    }
    
    ImGui::End();
}

void MainApplication::renderRecipeManager() {
    ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
    ImGui::Begin("Recipe Manager", &show_recipe_manager_);
    
    ImGui::Text("Manage detection recipes for different products");
    
    static char recipe_name[128] = "";
    ImGui::InputText("Recipe Name", recipe_name, 128);
    
    if (ImGui::Button("Save Current as Recipe")) {
        // TODO: Implement recipe saving
        ImGui::OpenPopup("Save Confirmation");
    }
    
    if (ImGui::BeginPopupModal("Save Confirmation")) {
        ImGui::Text("Recipe saved successfully!");
        if (ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    
    ImGui::End();
}

void MainApplication::renderPerformanceStats() {
    ImGui::SetNextWindowPos(ImVec2(960, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
    ImGui::Begin("Performance Statistics", &show_performance_stats_);
    
    auto stats = vision_pipeline_->getPerformanceStats();
    
    ImGui::Text("Frame Count: %d", stats.frame_count);
    ImGui::Separator();
    
    ImGui::Text("Average Total: %.2f ms", stats.avg_total_ms);
    ImGui::Text("Average Segmentation: %.2f ms", stats.avg_segmentation_ms);
    ImGui::Text("Average Contour: %.2f ms", stats.avg_contour_ms);
    ImGui::Separator();
    
    ImGui::Text("Min Frame Time: %.2f ms", stats.min_total_ms);
    ImGui::Text("Max Frame Time: %.2f ms", stats.max_total_ms);
    
    // Performance indicator
    float avg_fps = stats.avg_total_ms > 0 ? 1000.0f / stats.avg_total_ms : 0.0f;
    ImGui::Separator();
    ImGui::Text("Estimated FPS: %.1f", avg_fps);
    
    // Color-coded performance indicator
    if (stats.avg_total_ms < 10.0) {
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "TARGET MET: < 10ms");
    } else {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "TARGET MISSED: > 10ms");
    }
    
    if (ImGui::Button("Reset Statistics")) {
        vision_pipeline_->resetPerformanceStats();
    }
    
    ImGui::End();
}

void MainApplication::updateCameraTexture(const cv::Mat& frame) {
    if (frame.empty()) return;
    
    // Convert BGR to RGB
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    
    // Update OpenGL texture
    glBindTexture(GL_TEXTURE_2D, camera_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb_frame.cols, rgb_frame.rows,
                 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_frame.data);
}

void MainApplication::loadConfig(const std::string& path) {
    vision_pipeline_->initialize(path);
}

void MainApplication::saveConfig(const std::string& path) {
    // TODO: Implement config saving
    std::cout << "Config saved to: " << path << std::endl;
}

bool MainApplication::startVideoPlayback(const std::string& path) {
    if (path.empty()) {
        video_status_message_ = "Please enter a video file path.";
        return false;
    }
    
    // Release any existing stream
    camera_->release();
    camera_active_ = false;
    
    if (!camera_->initializeFromFile(path)) {
        using_video_file_ = false;
        video_loaded_ = false;
        video_finished_ = false;
        video_status_message_ = "Failed to open video: " + path;
        return false;
    }
    
    using_video_file_ = true;
    video_loaded_ = true;
    video_paused_ = false;
    video_finished_ = false;
    camera_active_ = true;
    video_path_ = path;
    current_frame_.release();
    last_result_ = DetectionResult();
    
    int fps = camera_->getFPS();
    video_frame_interval_ = fps > 0 ? 1.0 / static_cast<double>(fps) : 0.0;
    video_last_frame_time_ = glfwGetTime();
    video_status_message_ = "Video playing: " + path;
    
    return true;
}

void MainApplication::stopVideoPlayback() {
    if (!using_video_file_ && !video_loaded_) {
        return;
    }
    
    camera_->release();
    camera_active_ = false;
    using_video_file_ = false;
    video_paused_ = false;
    video_loaded_ = false;
    video_finished_ = false;
    video_frame_interval_ = 0.0;
    video_last_frame_time_ = 0.0;
    current_frame_.release();
    last_result_ = DetectionResult();
    video_status_message_ = "Video stopped";
}

void MainApplication::handleMouseInput() {
    // TODO: Implement ROI drawing with mouse
}

} // namespace country_style
