#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#ifdef _WIN32
    #include <windows.h>
    #include <commdlg.h>
#endif
#include <GLFW/glfw3.h>
#ifdef _WIN32
    #include <gl/GL.h>
#else
    #include <GL/gl.h>
#endif
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "vision_pipeline.h"
#include "recipe_manager.h"

using json = nlohmann::json;

namespace country_style {

struct Polygon {
    std::vector<cv::Point2f> points;
    bool is_good_sample;  // true = good dough, false = background/defect
    cv::Scalar color;
};

class PolygonTeachingApp {
public:
    PolygonTeachingApp() 
        : window_(nullptr),
          image_texture_(0),
          result_texture_(0),
          has_image_(false),
          has_results_(false),
          is_drawing_(false),
          show_help_(true),
          teach_mode_(true),
          image_scale_(1.0f),
          image_offset_x_(0.0f),
          image_offset_y_(0.0f),
          show_recipe_dialog_(false),
          show_new_recipe_dialog_(false),
          has_video_(false),
          video_loaded_(false),
          video_playing_(false),
          video_paused_(false),
          video_loop_(false),
          video_frame_interval_(0.0),
          video_last_time_(0.0) {
        
        vision_pipeline_ = std::make_unique<VisionPipeline>();
        vision_pipeline_->initialize("config/default_config.json");
        
        recipe_manager_ = std::make_unique<RecipeManager>();
        recipe_manager_->initialize("config/recipes");
        
        // Load available recipes
        refreshRecipeList();
    }
    
    ~PolygonTeachingApp() {
        shutdown();
    }
    
    bool initialize() {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return false;
        }
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        window_ = glfwCreateWindow(1600, 1000, "Country Style Dough Inspector - Polygon Teaching", nullptr, nullptr);
        if (!window_) {
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);
        
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        
        // Larger font for better readability (optional - falls back to default if not found)
        ImFont* font = nullptr;
#ifdef _WIN32
        // Try Windows font paths
        const char* windowsFonts[] = {
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\tahoma.ttf",
            "C:\\Windows\\Fonts\\segoeui.ttf"
        };
        for (const char* fontPath : windowsFonts) {
            font = io.Fonts->AddFontFromFileTTF(fontPath, 16.0f);
            if (font) break;
        }
#else
        // Try Linux font paths
        font = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16.0f);
        if (!font) {
            font = io.Fonts->AddFontFromFileTTF("/usr/share/fonts/TTF/DejaVuSans.ttf", 16.0f);
        }
#endif
        if (!font) {
            // Use default font if custom font not found
            // This is fine - ImGui has a built-in default font
        }
        
        // Custom modern styling
        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        
        // Modern color scheme
        ImVec4* colors = style.Colors;
        colors[ImGuiCol_Text]                   = ImVec4(0.95f, 0.95f, 0.95f, 1.00f);
        colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
        colors[ImGuiCol_WindowBg]               = ImVec4(0.10f, 0.10f, 0.12f, 0.98f);
        colors[ImGuiCol_ChildBg]                = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
        colors[ImGuiCol_PopupBg]                = ImVec4(0.12f, 0.12f, 0.14f, 0.98f);
        colors[ImGuiCol_Border]                 = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
        colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.30f);
        colors[ImGuiCol_FrameBg]                = ImVec4(0.18f, 0.18f, 0.22f, 1.00f);
        colors[ImGuiCol_FrameBgHovered]          = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
        colors[ImGuiCol_FrameBgActive]          = ImVec4(0.30f, 0.30f, 0.35f, 1.00f);
        colors[ImGuiCol_TitleBg]                = ImVec4(0.15f, 0.15f, 0.18f, 1.00f);
        colors[ImGuiCol_TitleBgActive]          = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
        colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
        colors[ImGuiCol_MenuBarBg]               = ImVec4(0.12f, 0.12f, 0.15f, 1.00f);
        colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.08f, 0.08f, 0.10f, 1.00f);
        colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.45f, 0.45f, 0.50f, 1.00f);
        colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.55f, 0.55f, 0.60f, 1.00f);
        colors[ImGuiCol_CheckMark]              = ImVec4(0.40f, 0.85f, 0.50f, 1.00f);
        colors[ImGuiCol_SliderGrab]             = ImVec4(0.40f, 0.65f, 0.90f, 1.00f);
        colors[ImGuiCol_SliderGrabActive]        = ImVec4(0.50f, 0.75f, 1.00f, 1.00f);
        colors[ImGuiCol_Button]                  = ImVec4(0.25f, 0.35f, 0.50f, 1.00f);
        colors[ImGuiCol_ButtonHovered]          = ImVec4(0.35f, 0.50f, 0.70f, 1.00f);
        colors[ImGuiCol_ButtonActive]           = ImVec4(0.45f, 0.60f, 0.85f, 1.00f);
        colors[ImGuiCol_Header]                  = ImVec4(0.30f, 0.40f, 0.55f, 1.00f);
        colors[ImGuiCol_HeaderHovered]           = ImVec4(0.40f, 0.55f, 0.75f, 1.00f);
        colors[ImGuiCol_HeaderActive]           = ImVec4(0.50f, 0.65f, 0.85f, 1.00f);
        colors[ImGuiCol_Separator]              = ImVec4(0.30f, 0.30f, 0.35f, 1.00f);
        colors[ImGuiCol_SeparatorHovered]        = ImVec4(0.40f, 0.40f, 0.45f, 1.00f);
        colors[ImGuiCol_SeparatorActive]         = ImVec4(0.50f, 0.50f, 0.55f, 1.00f);
        colors[ImGuiCol_ResizeGrip]             = ImVec4(0.35f, 0.35f, 0.40f, 1.00f);
        colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.45f, 0.45f, 0.50f, 1.00f);
        colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.55f, 0.55f, 0.60f, 1.00f);
        colors[ImGuiCol_Tab]                    = ImVec4(0.20f, 0.20f, 0.25f, 1.00f);
        colors[ImGuiCol_TabHovered]              = ImVec4(0.30f, 0.40f, 0.55f, 1.00f);
        colors[ImGuiCol_TabActive]              = ImVec4(0.35f, 0.50f, 0.70f, 1.00f);
        colors[ImGuiCol_TableHeaderBg]           = ImVec4(0.15f, 0.15f, 0.20f, 1.00f);
        colors[ImGuiCol_TableBorderStrong]       = ImVec4(0.30f, 0.30f, 0.35f, 1.00f);
        colors[ImGuiCol_TableBorderLight]       = ImVec4(0.25f, 0.25f, 0.30f, 1.00f);
        colors[ImGuiCol_TableRowBg]             = ImVec4(0.10f, 0.10f, 0.12f, 1.00f);
        colors[ImGuiCol_TableRowBgAlt]          = ImVec4(0.12f, 0.12f, 0.15f, 1.00f);
        colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.30f, 0.50f, 0.70f, 0.50f);
        colors[ImGuiCol_DragDropTarget]         = ImVec4(0.40f, 0.85f, 0.50f, 0.80f);
        colors[ImGuiCol_NavHighlight]           = ImVec4(0.40f, 0.65f, 0.90f, 1.00f);
        colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(0.40f, 0.65f, 0.90f, 0.80f);
        colors[ImGuiCol_NavWindowingDimBg]     = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);
        colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(0.00f, 0.00f, 0.00f, 0.60f);
        
        // Enhanced spacing and sizing
        style.WindowRounding = 10.0f;
        style.ChildRounding = 8.0f;
        style.FrameRounding = 6.0f;
        style.PopupRounding = 8.0f;
        style.ScrollbarRounding = 6.0f;
        style.GrabRounding = 4.0f;
        style.TabRounding = 6.0f;
        
        style.WindowPadding = ImVec2(12, 12);
        style.FramePadding = ImVec2(10, 6);
        style.ItemSpacing = ImVec2(10, 8);
        style.ItemInnerSpacing = ImVec2(8, 6);
        style.TouchExtraPadding = ImVec2(0, 0);
        style.IndentSpacing = 24.0f;
        style.ScrollbarSize = 14.0f;
        style.GrabMinSize = 12.0f;
        
        style.WindowBorderSize = 1.0f;
        style.ChildBorderSize = 1.0f;
        style.PopupBorderSize = 1.0f;
        style.FrameBorderSize = 0.0f;
        style.TabBorderSize = 1.0f;
        
        style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
        style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
        style.SelectableTextAlign = ImVec2(0.0f, 0.5f);
        
        style.AntiAliasedLines = true;
        style.AntiAliasedFill = true;
        style.CurveTessellationTol = 1.25f;
        
        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init("#version 330");
        
        glGenTextures(1, &image_texture_);
        glGenTextures(1, &result_texture_);
        
        // Load session state if it exists
        loadSession();
        
        return true;
    }
    
    void run() {
        last_autosave_ = std::chrono::steady_clock::now();
        
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            
            // Periodic auto-save
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_autosave_).count();
            if (elapsed >= AUTOSAVE_INTERVAL_SECONDS) {
                saveSession();
                last_autosave_ = now;
            }
            
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            
            renderUI();
            
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window_, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            
            glfwSwapBuffers(window_);
        }
    }
    
    void shutdown() {
        // Save session state before shutting down
        saveSession();
        
        if (image_texture_) glDeleteTextures(1, &image_texture_);
        if (result_texture_) glDeleteTextures(1, &result_texture_);
        
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        
        if (window_) {
            glfwDestroyWindow(window_);
            glfwTerminate();
        }
    }
    
private:
    GLFWwindow* window_;
    GLuint image_texture_;
    GLuint result_texture_;
    
    cv::Mat current_image_;
    cv::Mat result_image_;
    DetectionResult last_result_;
    std::string current_image_path_;  // For session persistence
    
    bool has_image_;
    bool has_results_;
    bool is_drawing_;
    bool show_help_;
    bool teach_mode_;
    
    float image_scale_;
    float image_offset_x_;
    float image_offset_y_;
    
    // Video playback state
    cv::VideoCapture video_cap_;
    bool has_video_;
    bool video_loaded_;
    bool video_playing_;
    bool video_paused_;
    bool video_loop_;
    double video_frame_interval_;
    double video_last_time_;
    std::string video_path_;
    std::string video_status_;
    
    std::vector<Polygon> polygons_;
    std::vector<cv::Point2f> current_polygon_;
    bool current_is_good_ = true;
    
    // ROI drawing
    bool enable_roi_ = false;
    bool drawing_roi_ = false;
    cv::Point2f roi_start_;
    cv::Point2f roi_end_;
    cv::Rect roi_rect_;
    
    // Display options for inference
    bool show_bounding_boxes_ = true;
    bool show_contours_ = true;
    bool show_mask_overlay_ = true;
    bool show_measurements_ = true;
    
    // Quality thresholds
    QualityThresholds quality_thresholds_;
    
    // Calibration: pixels per mm
    float pixels_per_mm_ = 1.0f;
    bool calibrating_ = false;
    cv::Point2f calib_start_;
    
    // Session loading flag
    bool session_loaded_ = false;
    cv::Point2f calib_end_;
    bool drawing_calib_line_ = false;
    
    std::unique_ptr<VisionPipeline> vision_pipeline_;
    std::unique_ptr<RecipeManager> recipe_manager_;
    
    // Recipe management
    std::vector<std::string> recipe_names_;
    int current_recipe_index_ = -1;
    bool show_recipe_dialog_;
    bool show_new_recipe_dialog_;
    char new_recipe_name_[256] = "";
    char new_recipe_desc_[512] = "";
    
    // Recipe editing
    bool editing_recipe_ = false;
    Recipe edited_recipe_;
    
    ImVec2 image_display_pos_;
    ImVec2 image_display_size_;
    
    // Session auto-save
    std::chrono::steady_clock::time_point last_autosave_;
    static constexpr int AUTOSAVE_INTERVAL_SECONDS = 30;  // Auto-save every 30 seconds
    
    void renderUI() {
        // Main window
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        
        ImGui::Begin("Polygon Teaching Interface", nullptr, 
                    ImGuiWindowFlags_NoTitleBar | 
                    ImGuiWindowFlags_NoResize | 
                    ImGuiWindowFlags_NoMove | 
                    ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_MenuBar);
        
        // Update video frame if playing
        if (video_playing_ && !video_paused_) {
            double now = glfwGetTime();
            if (video_frame_interval_ <= 0.0 || (now - video_last_time_) >= video_frame_interval_) {
                video_last_time_ = now;
                cv::Mat frame;
                if (video_cap_.isOpened() && video_cap_.read(frame)) {
                    current_image_ = frame;
                    has_image_ = true;
                    // Auto-run inference for live playback
                    runInference();
                } else {
                    // End or error
                    if (video_loop_ && !video_path_.empty()) {
                        video_cap_.release();
                        if (video_cap_.open(video_path_)) {
                            double fps = video_cap_.get(cv::CAP_PROP_FPS);
                            video_frame_interval_ = fps > 0 ? 1.0 / fps : 0.0;
                            video_status_ = "Looping video...";
                        } else {
                            video_status_ = "Failed to loop video";
                            video_playing_ = false;
                        }
                    } else {
                        video_status_ = "Video finished";
                        video_playing_ = false;
                    }
                }
            }
        }
        
        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Load Image", "Ctrl+O")) loadImage();
                if (ImGui::MenuItem("Save Annotations", "Ctrl+S", false, has_image_)) saveAnnotations();
                if (ImGui::MenuItem("Load Annotations", nullptr, false, has_image_)) loadAnnotations();
                ImGui::Separator();
                if (ImGui::MenuItem("Exit", "Esc")) glfwSetWindowShouldClose(window_, true);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("View")) {
                ImGui::MenuItem("Show Help", "F1", &show_help_);
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Mode")) {
                if (ImGui::MenuItem("Teach Mode", nullptr, teach_mode_)) {
                    teach_mode_ = true;
                    has_results_ = false;
                }
                if (ImGui::MenuItem("Inference Mode", nullptr, !teach_mode_)) {
                    teach_mode_ = false;
                    has_results_ = false;
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Recipe")) {
                if (ImGui::MenuItem("Manage Recipes...")) {
                    show_recipe_dialog_ = true;
                }
                if (ImGui::MenuItem("New Recipe...")) {
                    show_new_recipe_dialog_ = true;
                }
                ImGui::Separator();
                
                // Quick recipe selection
                for (size_t i = 0; i < recipe_names_.size(); i++) {
                    bool is_selected = (current_recipe_index_ == (int)i);
                    if (ImGui::MenuItem(recipe_names_[i].c_str(), nullptr, is_selected)) {
                        loadRecipe(recipe_names_[i]);
                    }
                }
                
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }
        
        // Recipe dialogs
        if (show_recipe_dialog_) {
            renderRecipeManagerDialog();
        }
        if (show_new_recipe_dialog_) {
            renderNewRecipeDialog();
        }
        
        // Help overlay
        if (show_help_) {
            renderHelpOverlay();
        }
        
        // Main layout
        ImGui::Columns(2, "MainColumns", true);
        ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.75f);
        
        // Left: Image display with polygon drawing
        renderImagePanel();
        
        ImGui::NextColumn();
        
        // Right: Controls
        if (teach_mode_) {
            renderTeachControls();
        } else {
            renderInferenceControls();
        }
        
        ImGui::Columns(1);
        ImGui::End();
    }
    
    void renderHelpOverlay() {
        ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 420, 30));
        ImGui::SetNextWindowSize(ImVec2(400, 0));
        ImGui::Begin("Quick Help", &show_help_, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);
        
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "TEACH MODE:");
        ImGui::BulletText("Left click to add polygon points");
        ImGui::BulletText("Right click or Enter to close polygon");
        ImGui::BulletText("Switch 'Good Sample' for background");
        ImGui::BulletText("Delete last polygon with 'Undo'");
        ImGui::BulletText("'Learn from Polygons' to train");
        
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "INFERENCE MODE:");
        ImGui::BulletText("Load image and click 'Run Detection'");
        ImGui::BulletText("View results and save annotated image");
        
        ImGui::End();
    }
    
    void renderImagePanel() {
        ImGui::BeginChild("ImagePanel", ImVec2(0, 0), true, ImGuiWindowFlags_NoScrollbar);
        
        if (!has_image_) {
            ImVec2 center = ImVec2(ImGui::GetWindowSize().x * 0.5f, ImGui::GetWindowSize().y * 0.5f);
            ImGui::SetCursorPos(ImVec2(center.x - 150, center.y - 50));
            ImGui::BeginGroup();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No image loaded");
            ImGui::Spacing();
            if (ImGui::Button("Click here to load an image", ImVec2(300, 40))) {
                loadImage();
            }
            ImGui::EndGroup();
        } else {
            // Calculate proper image display
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float img_aspect = (float)current_image_.cols / current_image_.rows;
            float avail_aspect = avail.x / avail.y;
            
            if (img_aspect > avail_aspect) {
                // Image is wider - fit to width
                image_display_size_.x = avail.x;
                image_display_size_.y = avail.x / img_aspect;
            } else {
                // Image is taller - fit to height
                image_display_size_.y = avail.y;
                image_display_size_.x = avail.y * img_aspect;
            }
            
            // Center the image
            image_offset_x_ = (avail.x - image_display_size_.x) * 0.5f;
            image_offset_y_ = (avail.y - image_display_size_.y) * 0.5f;
            
            image_display_pos_ = ImGui::GetCursorScreenPos();
            image_display_pos_.x += image_offset_x_;
            image_display_pos_.y += image_offset_y_;
            
            ImGui::SetCursorPos(ImVec2(image_offset_x_, image_offset_y_));
            
            // Display image with overlays
            cv::Mat display_mat;
            
            if (teach_mode_) {
                // Always show polygon overlay in teach mode
                display_mat = drawPolygonsOnImage();
            } else if (has_results_) {
                // Show inference results with mask overlay
                display_mat = result_image_.clone();
                // Draw ROI if enabled
                if (enable_roi_ && (drawing_roi_ || roi_rect_.width > 0)) {
                    drawROIOnImage(display_mat);
                }
            } else {
                // Just show the image with ROI or calibration line
                display_mat = current_image_.clone();
                if (enable_roi_ && (drawing_roi_ || roi_rect_.width > 0)) {
                    drawROIOnImage(display_mat);
                }
                // Draw calibration line if in calibration mode
                if (calibrating_ && drawing_calib_line_) {
                    cv::line(display_mat, 
                            cv::Point(calib_start_.x, calib_start_.y),
                            cv::Point(calib_end_.x, calib_end_.y),
                            cv::Scalar(0, 255, 255), 3);
                    cv::circle(display_mat, cv::Point(calib_start_.x, calib_start_.y), 6, cv::Scalar(0, 255, 0), -1);
                    cv::circle(display_mat, cv::Point(calib_end_.x, calib_end_.y), 6, cv::Scalar(0, 0, 255), -1);
                    
                    // Show length on the line
                    float len = std::sqrt(std::pow(calib_end_.x - calib_start_.x, 2) + 
                                         std::pow(calib_end_.y - calib_start_.y, 2));
                    std::string len_text = std::to_string((int)len) + "px";
                    cv::Point mid((calib_start_.x + calib_end_.x) / 2, (calib_start_.y + calib_end_.y) / 2);
                    cv::putText(display_mat, len_text, cv::Point(mid.x + 10, mid.y - 10),
                               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
                }
            }
            
            updateTexture(display_mat, teach_mode_ || !has_results_ ? image_texture_ : result_texture_);
            GLuint tex = teach_mode_ || !has_results_ ? image_texture_ : result_texture_;
            
            ImGui::Image((void*)(intptr_t)tex, image_display_size_);
            
            // Handle mouse input
            if (ImGui::IsItemHovered()) {
                if (teach_mode_) {
                    // Polygon or ROI drawing in teach mode
                    handlePolygonDrawing();
                } else if (calibrating_) {
                    // Calibration line drawing
                    handleCalibrationDrawing();
                } else if (enable_roi_) {
                    // ROI drawing in inference mode
                    handleROIDrawing();
                }
            }
        }
        
        ImGui::EndChild();
    }
    
    void renderTeachControls() {
        ImGui::BeginChild("TeachControls");
        
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "TEACH MODE");
        ImGui::Separator();
        ImGui::Spacing();
        
        if (!has_image_) {
            if (ImGui::Button("Load Training Image", ImVec2(-1, 50))) {
                loadImage();
            }
        } else {
            // ROI Controls
            ImGui::Checkbox("Enable ROI (Region of Interest)", &enable_roi_);
            if (enable_roi_) {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Click-drag to draw ROI box");
                if (roi_rect_.width > 0) {
                    ImGui::Text("ROI: %dx%d at (%d,%d)", 
                               roi_rect_.width, roi_rect_.height,
                               roi_rect_.x, roi_rect_.y);
                    if (ImGui::Button("Clear ROI", ImVec2(-1, 30))) {
                        roi_rect_ = cv::Rect(0, 0, 0, 0);
                        vision_pipeline_->updateROI(roi_rect_);
                    }
                }
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            ImGui::Text("Annotate Samples:");
            ImGui::Spacing();
            
            ImGui::Checkbox("Good Sample (Green)", &current_is_good_);
            if (!current_is_good_) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1, 0, 0, 1), " Bad Sample (Red)");
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            ImGui::Text("Polygons drawn: %d", (int)polygons_.size());
            int good_count = 0, bad_count = 0;
            for (const auto& poly : polygons_) {
                if (poly.is_good_sample) good_count++; else bad_count++;
            }
            ImGui::Text(" - Good samples: %d", good_count);
            ImGui::Text(" - Bad samples: %d", bad_count);
            
            ImGui::Spacing();
            
            if (is_drawing_) {
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "Drawing... (%d points)", (int)current_polygon_.size());
                if (ImGui::Button("Cancel Drawing (Esc)", ImVec2(-1, 30))) {
                    current_polygon_.clear();
                    is_drawing_ = false;
                }
            }
            
            ImGui::Spacing();
            
            if (ImGui::Button("Undo Last Polygon", ImVec2(-1, 35))) {
                if (!polygons_.empty()) {
                    polygons_.pop_back();
                }
            }
            
            if (ImGui::Button("Clear All Polygons", ImVec2(-1, 35))) {
                polygons_.clear();
                current_polygon_.clear();
                is_drawing_ = false;
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Training:");
            
            if (polygons_.size() > 0) {
                if (ImGui::Button("Learn from Polygons", ImVec2(-1, 60))) {
                    learnFromPolygons();
                }
            } else {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
                                 "Draw polygons to train");
            }
            
            if (has_results_) {
                ImGui::Spacing();
                ImGui::Text("Learned parameters!");
                ImGui::Text("Switch to Inference Mode to test");
            }
        }
        
        ImGui::EndChild();
    }
    
    void renderInferenceControls() {
        ImGui::BeginChild("InferenceControls");
        
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "INFERENCE MODE");
        ImGui::Separator();
        ImGui::Spacing();
        
        // Load sources
        if (ImGui::Button("Load Test Image", ImVec2(-1, 40))) {
            loadImage();
        }
        if (ImGui::Button("Load Video...", ImVec2(-1, 40))) {
            loadVideo();
        }
        
        // Video controls
        if (video_loaded_) {
            ImGui::Separator();
            ImGui::Text("Video: %s", video_path_.empty() ? "(unspecified)" : video_path_.c_str());
            ImGui::Checkbox("Loop", &video_loop_);
            ImGui::SameLine();
            if (ImGui::Button(video_playing_ && !video_paused_ ? "Pause" : "Play", ImVec2(110, 30))) {
                if (video_playing_ && !video_paused_) {
                    video_paused_ = true;
                    video_status_ = "Paused";
                } else {
                    if (!video_cap_.isOpened() && !video_path_.empty()) {
                        // Try reopen
                        if (video_cap_.open(video_path_)) {
                            double fps = video_cap_.get(cv::CAP_PROP_FPS);
                            video_frame_interval_ = fps > 0 ? 1.0 / fps : 0.0;
                        }
                    }
                    video_playing_ = true;
                    video_paused_ = false;
                    video_last_time_ = glfwGetTime();
                    video_status_ = "Playing";
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Restart", ImVec2(110, 30))) {
                if (!video_path_.empty()) {
                    video_cap_.release();
                    if (video_cap_.open(video_path_)) {
                        double fps = video_cap_.get(cv::CAP_PROP_FPS);
                        video_frame_interval_ = fps > 0 ? 1.0 / fps : 0.0;
                        video_playing_ = true;
                        video_paused_ = false;
                        video_last_time_ = glfwGetTime();
                        video_status_ = "Restarted";
                    } else {
                        video_status_ = "Failed to restart";
                    }
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Stop", ImVec2(110, 30))) {
                stopVideo();
                video_status_ = "Stopped";
            }
            if (!video_status_.empty()) {
                ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.3f, 1.0f), "%s", video_status_.c_str());
            }
        }
        
        if (has_image_) {
            // ROI Controls
            ImGui::Checkbox("Enable ROI", &enable_roi_);
            if (enable_roi_) {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Click-drag to draw ROI box");
                if (roi_rect_.width > 0) {
                    ImGui::Text("ROI: %dx%d at (%d,%d)", 
                               roi_rect_.width, roi_rect_.height,
                               roi_rect_.x, roi_rect_.y);
                    if (ImGui::Button("Clear ROI", ImVec2(-1, 30))) {
                        roi_rect_ = cv::Rect(0, 0, 0, 0);
                        vision_pipeline_->updateROI(roi_rect_);
                    }
                }
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // Display options
            ImGui::Text("Display Options:");
            ImGui::Checkbox("Show Bounding Boxes", &show_bounding_boxes_);
            ImGui::Checkbox("Show Contours", &show_contours_);
            ImGui::Checkbox("Show Mask Overlay", &show_mask_overlay_);
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // Calibration
            ImGui::TextColored(ImVec4(0.4, 1, 1, 1), "Calibration:");
            
            if (!calibrating_) {
                ImGui::InputFloat("Pixels per mm", &pixels_per_mm_, 0.1f, 1.0f, "%.2f");
                if (pixels_per_mm_ < 0.01f) pixels_per_mm_ = 0.01f;
                ImGui::Text("(1 mm = %.2f pixels)", pixels_per_mm_);
                
                if (ImGui::Button("Calibrate from Image", ImVec2(-1, 35))) {
                    calibrating_ = true;
                    drawing_calib_line_ = false;
                }
            } else {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "CALIBRATION MODE");
                ImGui::Text("Draw a line on a known distance");
                
                static float known_distance_mm = 100.0f;
                ImGui::InputFloat("Known Distance (mm)", &known_distance_mm, 1.0f, 10.0f, "%.1f");
                if (known_distance_mm < 0.1f) known_distance_mm = 0.1f;
                
                if (drawing_calib_line_) {
                    float line_length_px = std::sqrt(
                        std::pow(calib_end_.x - calib_start_.x, 2) + 
                        std::pow(calib_end_.y - calib_start_.y, 2)
                    );
                    ImGui::Text("Line length: %.1f pixels", line_length_px);
                    
                    if (ImGui::Button("Apply Calibration", ImVec2(-1, 35))) {
                        if (line_length_px > 1.0f) {
                            pixels_per_mm_ = line_length_px / known_distance_mm;
                            std::cout << "Calibrated: " << pixels_per_mm_ << " px/mm" << std::endl;
                        }
                        calibrating_ = false;
                        drawing_calib_line_ = false;
                    }
                }
                
                if (ImGui::Button("Cancel", ImVec2(-1, 30))) {
                    calibrating_ = false;
                    drawing_calib_line_ = false;
                }
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // Quality thresholds with enable/disable checkboxes
            ImGui::TextColored(ImVec4(1, 0.8, 0.4, 1), "Quality Thresholds:");
            ImGui::Spacing();
            
            // Enable/Disable checkboxes for each threshold type
            ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Select Thresholds to Monitor:");
            ImGui::Checkbox("✓ Area Check", &quality_thresholds_.enable_area_check);
            ImGui::Checkbox("✓ Width Check", &quality_thresholds_.enable_width_check);
            ImGui::Checkbox("✓ Length Check", &quality_thresholds_.enable_height_check);
            ImGui::Checkbox("✓ Aspect Ratio Check", &quality_thresholds_.enable_aspect_ratio_check);
            ImGui::Checkbox("✓ Circularity Check", &quality_thresholds_.enable_circularity_check);
            ImGui::Checkbox("✓ Count Check", &quality_thresholds_.enable_count_check);
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // Area thresholds (shown if enabled)
            if (quality_thresholds_.enable_area_check) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Area (pixels²):");
                static double min_area_val = 0;
                static double max_area_val = 0;
                static bool area_vals_initialized = false;
                if (!area_vals_initialized) {
                    min_area_val = quality_thresholds_.min_area;
                    max_area_val = quality_thresholds_.max_area;
                    area_vals_initialized = true;
                } else if (session_loaded_) {
                    // Sync once when session loads (first render after session loaded)
                    static bool area_synced = false;
                    if (!area_synced) {
                        min_area_val = quality_thresholds_.min_area;
                        max_area_val = quality_thresholds_.max_area;
                        area_synced = true;
                    }
                }
                ImGui::InputDouble("Min Area##area", &min_area_val, 10.0, 100.0, "%.0f");
                ImGui::InputDouble("Max Area##area", &max_area_val, 100.0, 1000.0, "%.0f");
                quality_thresholds_.min_area = min_area_val;
                quality_thresholds_.max_area = max_area_val;
                ImGui::Text("  Range: %.0f - %.0f px²", quality_thresholds_.min_area, quality_thresholds_.max_area);
                ImGui::Spacing();
            }
            
            // Width thresholds (shown if enabled)
            if (quality_thresholds_.enable_width_check) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Width (pixels):");
                static double min_width_val = 0;
                static double max_width_val = 0;
                static bool width_vals_initialized = false;
                if (!width_vals_initialized) {
                    min_width_val = quality_thresholds_.min_width;
                    max_width_val = quality_thresholds_.max_width;
                    width_vals_initialized = true;
                } else if (session_loaded_) {
                    // Sync once when session loads (first render after session loaded)
                    static bool width_synced = false;
                    if (!width_synced) {
                        min_width_val = quality_thresholds_.min_width;
                        max_width_val = quality_thresholds_.max_width;
                        width_synced = true;
                    }
                }
                ImGui::InputDouble("Min Width##width", &min_width_val, 1.0, 10.0, "%.0f");
                ImGui::InputDouble("Max Width##width", &max_width_val, 1.0, 10.0, "%.0f");
                quality_thresholds_.min_width = min_width_val;
                quality_thresholds_.max_width = max_width_val;
                ImGui::Text("  Range: %.0f - %.0f px", quality_thresholds_.min_width, quality_thresholds_.max_width);
                ImGui::Spacing();
            }
            
            // Length/Height thresholds (shown if enabled)
            if (quality_thresholds_.enable_height_check) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Length (pixels):");
                static double min_height_val = 0;
                static double max_height_val = 0;
                static bool height_vals_initialized = false;
                if (!height_vals_initialized) {
                    min_height_val = quality_thresholds_.min_height;
                    max_height_val = quality_thresholds_.max_height;
                    height_vals_initialized = true;
                } else if (session_loaded_) {
                    // Sync once when session loads (first render after session loaded)
                    static bool height_synced = false;
                    if (!height_synced) {
                        min_height_val = quality_thresholds_.min_height;
                        max_height_val = quality_thresholds_.max_height;
                        height_synced = true;
                    }
                }
                ImGui::InputDouble("Min Length##length", &min_height_val, 1.0, 10.0, "%.0f");
                ImGui::InputDouble("Max Length##length", &max_height_val, 1.0, 10.0, "%.0f");
                quality_thresholds_.min_height = min_height_val;
                quality_thresholds_.max_height = max_height_val;
                ImGui::Text("  Range: %.0f - %.0f px", quality_thresholds_.min_height, quality_thresholds_.max_height);
                ImGui::Spacing();
            }
            
            // Aspect Ratio thresholds (shown if enabled)
            if (quality_thresholds_.enable_aspect_ratio_check) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Aspect Ratio:");
                static double min_ar_val = 0;
                static double max_ar_val = 0;
                static bool ar_vals_initialized = false;
                if (!ar_vals_initialized) {
                    min_ar_val = quality_thresholds_.min_aspect_ratio;
                    max_ar_val = quality_thresholds_.max_aspect_ratio;
                    ar_vals_initialized = true;
                } else if (session_loaded_) {
                    // Sync once when session loads (first render after session loaded)
                    static bool ar_synced = false;
                    if (!ar_synced) {
                        min_ar_val = quality_thresholds_.min_aspect_ratio;
                        max_ar_val = quality_thresholds_.max_aspect_ratio;
                        ar_synced = true;
                    }
                }
                ImGui::InputDouble("Min Aspect Ratio##ar", &min_ar_val, 0.1, 1.0, "%.2f");
                ImGui::InputDouble("Max Aspect Ratio##ar", &max_ar_val, 0.1, 1.0, "%.2f");
                quality_thresholds_.min_aspect_ratio = min_ar_val;
                quality_thresholds_.max_aspect_ratio = max_ar_val;
                ImGui::Text("  Range: %.2f - %.2f", quality_thresholds_.min_aspect_ratio, quality_thresholds_.max_aspect_ratio);
                ImGui::Spacing();
            }
            
            // Circularity thresholds (shown if enabled)
            if (quality_thresholds_.enable_circularity_check) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Circularity:");
                static double min_circ_val = 0;
                static double max_circ_val = 0;
                static bool circ_vals_initialized = false;
                if (!circ_vals_initialized) {
                    min_circ_val = quality_thresholds_.min_circularity;
                    max_circ_val = quality_thresholds_.max_circularity;
                    circ_vals_initialized = true;
                } else if (session_loaded_) {
                    // Sync once when session loads (first render after session loaded)
                    static bool circ_synced = false;
                    if (!circ_synced) {
                        min_circ_val = quality_thresholds_.min_circularity;
                        max_circ_val = quality_thresholds_.max_circularity;
                        circ_synced = true;
                    }
                }
                ImGui::InputDouble("Min Circularity##circ", &min_circ_val, 0.01, 0.1, "%.2f");
                ImGui::InputDouble("Max Circularity##circ", &max_circ_val, 0.01, 0.1, "%.2f");
                quality_thresholds_.min_circularity = min_circ_val;
                quality_thresholds_.max_circularity = max_circ_val;
                ImGui::Text("  Range: %.2f - %.2f", quality_thresholds_.min_circularity, quality_thresholds_.max_circularity);
                ImGui::Spacing();
            }
            
            // Count thresholds (shown if enabled)
            if (quality_thresholds_.enable_count_check) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Count:");
                ImGui::InputInt("Expected Count##count", &quality_thresholds_.expected_count);
                ImGui::Checkbox("Enforce Exact Count##count", &quality_thresholds_.enforce_exact_count);
                if (!quality_thresholds_.enforce_exact_count) {
                    ImGui::InputInt("Min Count##count", &quality_thresholds_.min_count);
                    ImGui::InputInt("Max Count##count", &quality_thresholds_.max_count);
                }
                ImGui::Spacing();
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            if (ImGui::Button("Run Detection", ImVec2(-1, 60))) {
                // Apply thresholds before running
                vision_pipeline_->updateQualityThresholds(quality_thresholds_);
                runInference();
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            if (has_results_) {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "RESULTS:");
                ImGui::Separator();
                
                ImGui::Text("Dough Count: %d", last_result_.dough_count);
                
                // Status with color coding
                if (last_result_.is_valid) {
                    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Status: PASS ✓");
                } else {
                    ImGui::TextColored(ImVec4(1, 0, 0, 1), "Status: FAIL ✗");
                }
                
                // Show fault flags if any
                if (!last_result_.is_valid && !last_result_.fault_messages.empty()) {
                    ImGui::Spacing();
                    ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "FAULTS:");
                    for (const auto& fault : last_result_.fault_messages) {
                        ImGui::TextWrapped(" • %s", fault.c_str());
                    }
                }
                
                // Measurements table
                if (show_measurements_ && !last_result_.measurements.empty()) {
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.5, 1, 1, 1), "MEASUREMENTS:");
                    
                    if (ImGui::BeginTable("Measurements", 7, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 150))) {
                        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 30);
                        ImGui::TableSetupColumn("W(px)", ImGuiTableColumnFlags_WidthFixed, 50);
                        ImGui::TableSetupColumn("W(mm)", ImGuiTableColumnFlags_WidthFixed, 50);
                        ImGui::TableSetupColumn("L(px)", ImGuiTableColumnFlags_WidthFixed, 50);
                        ImGui::TableSetupColumn("L(mm)", ImGuiTableColumnFlags_WidthFixed, 50);
                        ImGui::TableSetupColumn("Area(px²)", ImGuiTableColumnFlags_WidthFixed, 70);
                        ImGui::TableSetupColumn("Area(mm²)", ImGuiTableColumnFlags_WidthFixed, 70);
                        ImGui::TableHeadersRow();
                        
                        for (const auto& meas : last_result_.measurements) {
                            ImGui::TableNextRow();
                            
                            // ID column - color coded by status
                            ImGui::TableSetColumnIndex(0);
                            if (meas.meets_specs) {
                                ImGui::TextColored(ImVec4(0, 1, 0, 1), "#%d", meas.id);
                            } else {
                                ImGui::TextColored(ImVec4(1, 0, 0, 1), "#%d", meas.id);
                            }
                            
                            // Width (px)
                            ImGui::TableSetColumnIndex(1);
                            if (quality_thresholds_.enable_width_check && 
                                quality_thresholds_.min_width > 0 && 
                                (meas.width_pixels < quality_thresholds_.min_width || meas.width_pixels > quality_thresholds_.max_width)) {
                                ImGui::TextColored(ImVec4(1, 0, 0, 1), "%d", (int)meas.width_pixels);
                            } else {
                                ImGui::Text("%d", (int)meas.width_pixels);
                            }
                            
                            // Width (mm)
                            ImGui::TableSetColumnIndex(2);
                            float w_mm = meas.width_pixels / pixels_per_mm_;
                            ImGui::Text("%d", (int)w_mm);
                            
                            // Length (px)
                            ImGui::TableSetColumnIndex(3);
                            if (quality_thresholds_.enable_height_check && 
                                quality_thresholds_.min_height > 0 && 
                                (meas.height_pixels < quality_thresholds_.min_height || meas.height_pixels > quality_thresholds_.max_height)) {
                                ImGui::TextColored(ImVec4(1, 0, 0, 1), "%d", (int)meas.height_pixels);
                            } else {
                                ImGui::Text("%d", (int)meas.height_pixels);
                            }
                            
                            // Length (mm)
                            ImGui::TableSetColumnIndex(4);
                            float h_mm = meas.height_pixels / pixels_per_mm_;
                            ImGui::Text("%d", (int)h_mm);
                            
                            // Area (px²)
                            ImGui::TableSetColumnIndex(5);
                            if (quality_thresholds_.enable_area_check && 
                                quality_thresholds_.min_area > 0 && 
                                (meas.area_pixels < quality_thresholds_.min_area || meas.area_pixels > quality_thresholds_.max_area)) {
                                ImGui::TextColored(ImVec4(1, 0, 0, 1), "%.0f", meas.area_pixels);
                            } else {
                                ImGui::Text("%.0f", meas.area_pixels);
                            }
                            
                            // Area (mm²)
                            ImGui::TableSetColumnIndex(6);
                            float area_mm2 = meas.area_pixels / (pixels_per_mm_ * pixels_per_mm_);
                            ImGui::Text("%.1f", area_mm2);
                        }
                        ImGui::EndTable();
                    }
                }
                
                ImGui::Spacing();
                ImGui::Text("Performance:");
                ImGui::Text(" Total: %.2f ms", last_result_.total_time_ms);
                ImGui::Text(" Segmentation: %.2f ms", last_result_.segmentation_time_ms);
                
                if (last_result_.total_time_ms < 10.0) {
                    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Target <10ms: MET ✓");
                } else {
                    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Target <10ms: %.1fms", 
                                     last_result_.total_time_ms);
                }
                
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                
                if (ImGui::Button("Save Result Image", ImVec2(-1, 40))) {
                    saveResultImage();
                }
            }
        }
        
        ImGui::EndChild();
    }
    
    void handleROIDrawing() {
        // Get mouse position relative to image
        ImVec2 mouse_pos = ImGui::GetMousePos();
        cv::Point2f img_point = screenToImageCoords(mouse_pos);
        
        // Check if mouse is within image bounds
        if (img_point.x < 0 || img_point.y < 0 || 
            img_point.x >= current_image_.cols || img_point.y >= current_image_.rows) {
            return;
        }
        
        // Start drawing ROI
        if (ImGui::IsMouseClicked(0)) {
            drawing_roi_ = true;
            roi_start_ = img_point;
            roi_end_ = img_point;
        }
        
        // Update ROI while dragging
        if (drawing_roi_ && ImGui::IsMouseDown(0)) {
            roi_end_ = img_point;
        }
        
        // Finish ROI
        if (drawing_roi_ && ImGui::IsMouseReleased(0)) {
            drawing_roi_ = false;
            
            // Create rectangle from start and end points
            int x = std::min(roi_start_.x, roi_end_.x);
            int y = std::min(roi_start_.y, roi_end_.y);
            int w = std::abs(roi_end_.x - roi_start_.x);
            int h = std::abs(roi_end_.y - roi_start_.y);
            
            roi_rect_ = cv::Rect(x, y, w, h);
            vision_pipeline_->updateROI(roi_rect_);
            
            std::cout << "ROI set: " << roi_rect_.x << "," << roi_rect_.y 
                     << " " << roi_rect_.width << "x" << roi_rect_.height << std::endl;
        }
    }
    
    void handleCalibrationDrawing() {
        // Get mouse position relative to image
        ImVec2 mouse_pos = ImGui::GetMousePos();
        cv::Point2f img_point = screenToImageCoords(mouse_pos);
        
        // Check if mouse is within image bounds
        if (img_point.x < 0 || img_point.y < 0 || 
            img_point.x >= current_image_.cols || img_point.y >= current_image_.rows) {
            return;
        }
        
        // Start drawing calibration line
        if (ImGui::IsMouseClicked(0)) {
            drawing_calib_line_ = true;
            calib_start_ = img_point;
            calib_end_ = img_point;
        }
        
        // Update line while dragging
        if (drawing_calib_line_ && ImGui::IsMouseDown(0)) {
            calib_end_ = img_point;
        }
        
        // Finish line
        if (drawing_calib_line_ && ImGui::IsMouseReleased(0)) {
            // Line is drawn, user can now apply calibration from UI
        }
    }
    
    void handlePolygonDrawing() {
        ImGuiIO& io = ImGui::GetIO();
        
        // Get mouse position relative to image
        ImVec2 mouse_pos = ImGui::GetMousePos();
        cv::Point2f img_point = screenToImageCoords(mouse_pos);
        
        // Check if mouse is within image bounds
        if (img_point.x < 0 || img_point.y < 0 || 
            img_point.x >= current_image_.cols || img_point.y >= current_image_.rows) {
            return;
        }
        
        // ROI DRAWING MODE
        if (enable_roi_) {
            // Start drawing ROI
            if (ImGui::IsMouseClicked(0)) {
                drawing_roi_ = true;
                roi_start_ = img_point;
                roi_end_ = img_point;
            }
            
            // Update ROI while dragging
            if (drawing_roi_ && ImGui::IsMouseDown(0)) {
                roi_end_ = img_point;
            }
            
            // Finish ROI on release
            if (drawing_roi_ && ImGui::IsMouseReleased(0)) {
                drawing_roi_ = false;
                
                // Create rectangle from start and end points
                int x = std::max(0, std::min((int)roi_start_.x, (int)roi_end_.x));
                int y = std::max(0, std::min((int)roi_start_.y, (int)roi_end_.y));
                int x2 = std::min(current_image_.cols, std::max((int)roi_start_.x, (int)roi_end_.x));
                int y2 = std::min(current_image_.rows, std::max((int)roi_start_.y, (int)roi_end_.y));
                int w = x2 - x;
                int h = y2 - y;
                
                // Only set ROI if it's reasonably sized
                if (w > 10 && h > 10) {
                    roi_rect_ = cv::Rect(x, y, w, h);
                    vision_pipeline_->updateROI(roi_rect_);
                    std::cout << "ROI set: " << roi_rect_.x << "," << roi_rect_.y 
                             << " " << roi_rect_.width << "x" << roi_rect_.height << std::endl;
                }
            }
            
            return;
        }
        
        // POLYGON DRAWING MODE (normal)
        // Left click to add point
        if (ImGui::IsMouseClicked(0)) {
            current_polygon_.push_back(img_point);
            is_drawing_ = true;
        }
        
        // Right click or Enter to finish polygon
        if ((ImGui::IsMouseClicked(1) || ImGui::IsKeyPressed(ImGuiKey_Enter)) && current_polygon_.size() >= 3) {
            Polygon poly;
            poly.points = current_polygon_;
            poly.is_good_sample = current_is_good_;
            poly.color = current_is_good_ ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            polygons_.push_back(poly);
            
            current_polygon_.clear();
            is_drawing_ = false;
        }
        
        // Escape to cancel
        if (ImGui::IsKeyPressed(ImGuiKey_Escape) && is_drawing_) {
            current_polygon_.clear();
            is_drawing_ = false;
        }
    }
    
    cv::Point2f screenToImageCoords(ImVec2 screen_pos) {
        // Direct mapping from screen to image coordinates
        float x = (screen_pos.x - image_display_pos_.x) / image_display_size_.x * current_image_.cols;
        float y = (screen_pos.y - image_display_pos_.y) / image_display_size_.y * current_image_.rows;
        // No flip needed - coordinates map directly
        return cv::Point2f(x, y);
    }
    
    void drawROIOnImage(cv::Mat& img) {
        if (drawing_roi_) {
            // Draw live ROI while dragging
            int x = std::min(roi_start_.x, roi_end_.x);
            int y = std::min(roi_start_.y, roi_end_.y);
            int w = std::abs(roi_end_.x - roi_start_.x);
            int h = std::abs(roi_end_.y - roi_start_.y);
            cv::Rect temp_roi(x, y, w, h);
            cv::rectangle(img, temp_roi, cv::Scalar(255, 255, 0), 3);
            cv::putText(img, "ROI", cv::Point(x+5, y+25),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        } else if (roi_rect_.width > 0 && roi_rect_.height > 0) {
            // Draw finalized ROI
            cv::rectangle(img, roi_rect_, cv::Scalar(0, 255, 255), 3);
            cv::putText(img, "ROI", cv::Point(roi_rect_.x+5, roi_rect_.y+25),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        }
    }
    
    cv::Mat drawPolygonsOnImage() {
        cv::Mat display = current_image_.clone();
        
        // Draw ROI rectangle if enabled
        if (enable_roi_) {
            if (drawing_roi_) {
                // Draw live ROI while dragging
                int x = std::min(roi_start_.x, roi_end_.x);
                int y = std::min(roi_start_.y, roi_end_.y);
                int w = std::abs(roi_end_.x - roi_start_.x);
                int h = std::abs(roi_end_.y - roi_start_.y);
                cv::Rect temp_roi(x, y, w, h);
                cv::rectangle(display, temp_roi, cv::Scalar(255, 255, 0), 3);
                cv::putText(display, "ROI", cv::Point(x+5, y+25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
            } else if (roi_rect_.width > 0 && roi_rect_.height > 0) {
                // Draw finalized ROI
                cv::rectangle(display, roi_rect_, cv::Scalar(0, 255, 255), 3);
                cv::putText(display, "ROI", cv::Point(roi_rect_.x+5, roi_rect_.y+25),
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
            }
        }
        
        // Draw completed polygons with semi-transparent fill
        for (const auto& poly : polygons_) {
            std::vector<cv::Point> pts;
            for (const auto& pt : poly.points) {
                pts.push_back(cv::Point(pt.x, pt.y));
            }
            
            // Draw filled polygon with transparency
            cv::Mat overlay = display.clone();
            std::vector<std::vector<cv::Point>> contours = {pts};
            cv::fillPoly(overlay, contours, poly.color);
            cv::addWeighted(display, 0.6, overlay, 0.4, 0, display);
            
            // Draw polygon outline
            cv::polylines(display, pts, true, poly.color, 3);
            
            // Draw corner points
            for (const auto& pt : pts) {
                cv::circle(display, pt, 4, poly.color, -1);
                cv::circle(display, pt, 5, cv::Scalar(255, 255, 255), 1);
            }
        }
        
        // Draw current polygon being drawn (LIVE OVERLAY)
        if (is_drawing_ && !current_polygon_.empty()) {
            std::vector<cv::Point> pts;
            for (const auto& pt : current_polygon_) {
                pts.push_back(cv::Point(pt.x, pt.y));
            }
            
            // Draw live polygon fill (semi-transparent yellow)
            if (pts.size() >= 3) {
                cv::Mat overlay = display.clone();
                std::vector<std::vector<cv::Point>> contours = {pts};
                cv::fillPoly(overlay, contours, cv::Scalar(0, 255, 255));
                cv::addWeighted(display, 0.7, overlay, 0.3, 0, display);
            }
            
            // Draw lines connecting points
            if (pts.size() > 1) {
                cv::polylines(display, pts, false, cv::Scalar(255, 255, 0), 3);
            }
            
            // Draw each point with labels
            for (size_t i = 0; i < pts.size(); i++) {
                // Big yellow circle for point
                cv::circle(display, pts[i], 8, cv::Scalar(0, 255, 255), -1);
                cv::circle(display, pts[i], 9, cv::Scalar(255, 255, 255), 2);
                
                // Point number
                std::string label = std::to_string(i + 1);
                cv::putText(display, label, cv::Point(pts[i].x + 12, pts[i].y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            }
            
            // Draw preview line to show where next point would connect
            if (pts.size() >= 1) {
                cv::Point last_pt = pts.back();
                // Note: We can't get current mouse pos here, but points are visible
            }
        }
        
        return display;
    }
    
    void updateTexture(const cv::Mat& img, GLuint texture) {
        if (img.empty()) return;
        
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        // Don't flip - keep original orientation
        
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        #ifndef GL_CLAMP_TO_EDGE
        #define GL_CLAMP_TO_EDGE 0x812F
        #endif
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        // Use GL_UNPACK_ALIGNMENT for proper row alignment
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 
                     0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    }
    
    void loadImage() {
        std::string path;
#ifdef _WIN32
        // Windows: Use native file dialog
        OPENFILENAMEA ofn;
        char szFile[260] = {0};
        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = nullptr;
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        ofn.lpstrFilter = "Image Files\0*.jpg;*.jpeg;*.png;*.bmp\0All Files\0*.*\0";
        ofn.nFilterIndex = 1;
        ofn.lpstrFileTitle = nullptr;
        ofn.nMaxFileTitle = 0;
        ofn.lpstrInitialDir = nullptr;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
        
        if (GetOpenFileNameA(&ofn) == TRUE) {
            path = szFile;
        } else {
            return;  // User cancelled
        }
#else
        std::string command = "zenity --file-selection --title='Select Image' --file-filter='Images | *.jpg *.jpeg *.png *.bmp'";
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) return;
        
        char buffer[1024];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            path = buffer;
            if (!path.empty() && path[path.length()-1] == '\n') {
                path.erase(path.length()-1);
            }
        }
        pclose(pipe);
#endif
        
        if (!path.empty()) {
            // Stop any ongoing video
            stopVideo();
            current_image_ = cv::imread(path);
            if (!current_image_.empty()) {
                current_image_path_ = path;  // Save path for session persistence
                has_image_ = true;
                has_results_ = false;
                polygons_.clear();
                current_polygon_.clear();
                is_drawing_ = false;
                std::cout << "Loaded: " << path << " (" << current_image_.cols << "x" << current_image_.rows << ")" << std::endl;
            }
        }
    }
    
    void loadVideo() {
        std::string path;
#ifdef _WIN32
        // Windows: Use native file dialog
        OPENFILENAMEA ofn;
        char szFile[260] = {0};
        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = nullptr;
        ofn.lpstrFile = szFile;
        ofn.nMaxFile = sizeof(szFile);
        ofn.lpstrFilter = "Video Files\0*.mp4;*.mov;*.avi;*.mkv\0All Files\0*.*\0";
        ofn.nFilterIndex = 1;
        ofn.lpstrFileTitle = nullptr;
        ofn.nMaxFileTitle = 0;
        ofn.lpstrInitialDir = nullptr;
        ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
        
        if (GetOpenFileNameA(&ofn) == TRUE) {
            path = szFile;
        } else {
            return;  // User cancelled
        }
#else
        std::string command = "zenity --file-selection --title='Select Video' --file-filter='Videos | *.mp4 *.mov *.avi *.mkv'";
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) return;
        
        char buffer[1024];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            path = buffer;
            if (!path.empty() && path[path.length()-1] == '\n') {
                path.erase(path.length()-1);
            }
        }
        pclose(pipe);
#endif
        
        if (!path.empty()) {
            // Release any previous capture
            stopVideo();
            video_path_ = path;
            if (video_cap_.open(video_path_)) {
                double fps = video_cap_.get(cv::CAP_PROP_FPS);
                video_frame_interval_ = fps > 0 ? 1.0 / fps : 0.0;
                has_video_ = true;
                video_loaded_ = true;
                video_playing_ = true;
                video_paused_ = false;
                video_last_time_ = glfwGetTime();
                video_status_ = "Loaded";
                std::cout << "Loaded video: " << video_path_ << " @ " << fps << " FPS" << std::endl;
            } else {
                has_video_ = false;
                video_loaded_ = false;
                video_playing_ = false;
                video_status_ = "Failed to open video";
                std::cerr << "Failed to open video: " << video_path_ << std::endl;
            }
        }
    }
    
    void stopVideo() {
        if (video_cap_.isOpened()) {
            video_cap_.release();
        }
        has_video_ = false;
        video_loaded_ = false;
        video_playing_ = false;
        video_paused_ = false;
        video_last_time_ = 0.0;
        video_frame_interval_ = 0.0;
    }
    
    void learnFromPolygons() {
        if (polygons_.empty()) return;
        
        // Collect ALL pixel samples from good polygons for robust statistics
        std::vector<std::vector<double>> h_values, s_values, v_values;
        std::vector<double> good_areas;
        
        cv::Mat hsv;
        cv::cvtColor(current_image_, hsv, cv::COLOR_BGR2HSV);
        
        for (const auto& poly : polygons_) {
            if (poly.is_good_sample) {
                // Create mask for this polygon
                cv::Mat mask = cv::Mat::zeros(current_image_.size(), CV_8UC1);
                std::vector<cv::Point> pts;
                for (const auto& pt : poly.points) {
                    pts.push_back(cv::Point(pt.x, pt.y));
                }
                std::vector<std::vector<cv::Point>> contours = {pts};
                cv::fillPoly(mask, contours, cv::Scalar(255));
                
                // Sample ALL pixels inside this polygon
                std::vector<double> h_poly, s_poly, v_poly;
                for (int y = 0; y < hsv.rows; y++) {
                    for (int x = 0; x < hsv.cols; x++) {
                        if (mask.at<uchar>(y, x) > 0) {
                            cv::Vec3b pixel = hsv.at<cv::Vec3b>(y, x);
                            h_poly.push_back(pixel[0]);
                            s_poly.push_back(pixel[1]);
                            v_poly.push_back(pixel[2]);
                        }
                    }
                }
                
                h_values.push_back(h_poly);
                s_values.push_back(s_poly);
                v_values.push_back(v_poly);
                
                // Get area
                double area = cv::contourArea(pts);
                good_areas.push_back(area);
            }
        }
        
        if (!h_values.empty()) {
            // Flatten all samples from all polygons
            std::vector<double> all_h, all_s, all_v;
            for (const auto& vec : h_values) all_h.insert(all_h.end(), vec.begin(), vec.end());
            for (const auto& vec : s_values) all_s.insert(all_s.end(), vec.begin(), vec.end());
            for (const auto& vec : v_values) all_v.insert(all_v.end(), vec.begin(), vec.end());
            
            // Sort for percentile calculation
            std::sort(all_h.begin(), all_h.end());
            std::sort(all_s.begin(), all_s.end());
            std::sort(all_v.begin(), all_v.end());
            
            // Use 10th and 90th percentiles (tighter - matches Java version)
            auto percentile = [](const std::vector<double>& sorted, double p) {
                int idx = std::max(0, std::min((int)(sorted.size() * p), (int)sorted.size() - 1));
                return sorted[idx];
            };
            
            double h_lower = percentile(all_h, 0.10);
            double h_upper = percentile(all_h, 0.90);
            double s_lower = percentile(all_s, 0.10);
            double s_upper = percentile(all_s, 0.90);
            double v_lower = percentile(all_v, 0.10);
            double v_upper = percentile(all_v, 0.90);
            
            std::cout << "  Raw percentile ranges - H:[" << h_lower << "-" << h_upper 
                      << "] S:[" << s_lower << "-" << s_upper 
                      << "] V:[" << v_lower << "-" << v_upper << "]" << std::endl;
            
            // Fixed tolerances (MATCHES JAVA EXACTLY): H±15, S±50, V±60
            double h_tol = 15.0;
            double s_tol = 50.0;
            double v_tol = 60.0;
            
            cv::Scalar lower, upper;
            lower[0] = std::max(0.0, h_lower - h_tol);
            upper[0] = std::min(180.0, h_upper + h_tol);
            lower[1] = std::max(0.0, s_lower - s_tol);
            upper[1] = std::min(255.0, s_upper + s_tol);
            lower[2] = std::max(0.0, v_lower - v_tol);
            upper[2] = std::min(255.0, v_upper + v_tol);
            
            // Update color range (ROI not used in teach mode)
            vision_pipeline_->updateColorRange(lower, upper);
            
            // Calculate area rules from samples with very generous tolerance
            double min_area = *std::min_element(good_areas.begin(), good_areas.end());
            double max_area = *std::max_element(good_areas.begin(), good_areas.end());
            
            // Very generous area tolerance for real-world variation
            min_area = std::max(100.0, min_area * 0.25);  // 75% smaller OK, minimum 100px
            max_area = max_area * 4.0;  // 4x bigger OK
            
            // Update detection rules - VERY LENIENT for maximum recall
            DetectionRules rules;
            rules.min_area = min_area;
            rules.max_area = max_area;
            rules.min_circularity = 0.0;  // Accept any shape
            rules.max_circularity = 1.0;
            rules.min_aspect_ratio = 0.0;  // Accept any aspect ratio
            rules.max_aspect_ratio = 100.0;
            rules.expected_count = 0;
            rules.enforce_count = false;
            
            vision_pipeline_->updateDetectionRules(rules);
            
            std::cout << "\n===== LEARNED PARAMETERS =====" << std::endl;
            std::cout << "Polygons analyzed: " << h_values.size() << std::endl;
            std::cout << "Total pixels sampled: " << all_h.size() << std::endl;
            std::cout << "\nHSV Ranges (5th-95th percentile + margin):" << std::endl;
            std::cout << "  Hue:        " << lower[0] << " - " << upper[0] << " (0-180)" << std::endl;
            std::cout << "  Saturation: " << lower[1] << " - " << upper[1] << " (0-255)" << std::endl;
            std::cout << "  Value:      " << lower[2] << " - " << upper[2] << " (0-255)" << std::endl;
            std::cout << "\nArea Range: " << (int)min_area << " - " << (int)max_area << " pixels" << std::endl;
            std::cout << "Shape Rules: ANY (no circularity/aspect ratio limits)" << std::endl;
            std::cout << "==============================\n" << std::endl;
            
            has_results_ = true;
        }
    }
    
    void runInference() {
        if (!has_image_) return;
        
        // Validate ROI before applying
        if (enable_roi_ && roi_rect_.width > 0 && roi_rect_.height > 0) {
            // Clamp ROI to image bounds
            int x = std::max(0, std::min(roi_rect_.x, current_image_.cols - 1));
            int y = std::max(0, std::min(roi_rect_.y, current_image_.rows - 1));
            int w = std::min(roi_rect_.width, current_image_.cols - x);
            int h = std::min(roi_rect_.height, current_image_.rows - y);
            
            if (w > 10 && h > 10) {
                cv::Rect safe_roi(x, y, w, h);
                vision_pipeline_->updateROI(safe_roi);
                std::cout << "Inference with ROI: " << safe_roi.x << "," << safe_roi.y 
                         << " " << safe_roi.width << "x" << safe_roi.height << std::endl;
            } else {
                vision_pipeline_->updateROI(cv::Rect(0, 0, 0, 0));
                std::cout << "ROI too small, using full image" << std::endl;
            }
        } else {
            vision_pipeline_->updateROI(cv::Rect(0, 0, 0, 0));
            std::cout << "Inference on full image (no ROI)" << std::endl;
        }
        
        try {
            last_result_ = vision_pipeline_->processFrame(current_image_);
        } catch (const std::exception& e) {
            std::cerr << "Error in processFrame: " << e.what() << std::endl;
            return;
        }
        
        // Create enhanced visualization with mask overlay
        result_image_ = current_image_.clone();
        
        // Get the segmentation mask
        cv::Mat mask;
        try {
            mask = vision_pipeline_->getSegmentedMask();
        } catch (const std::exception& e) {
            std::cerr << "Error getting mask: " << e.what() << std::endl;
            has_results_ = true;
            return;
        }
        
        // Draw mask overlay if enabled
        if (show_mask_overlay_ && !mask.empty() && 
            mask.rows == result_image_.rows && mask.cols == result_image_.cols) {
            // Create colored mask overlay (cyan for detected regions)
            cv::Mat mask_colored = cv::Mat::zeros(result_image_.size(), result_image_.type());
            
            // Convert grayscale mask to BGR
            cv::Mat mask_bgr;
            cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);
            
            // Apply cyan color where mask is white
            mask_bgr.setTo(cv::Scalar(255, 255, 0), mask);
            
            // Blend mask with original image (only where mask size matches)
            cv::addWeighted(result_image_, 0.6, mask_bgr, 0.4, 0, result_image_);
        }
        
        // Draw detection results (contours and bounding boxes)
        for (size_t i = 0; i < last_result_.contours.size(); i++) {
            // Safety check
            if (last_result_.contours[i].empty()) continue;
            if (i >= last_result_.bounding_boxes.size()) continue;
            
            // ROI-aware drawing: only show overlays inside ROI if enabled
            cv::Rect active_roi = vision_pipeline_->getROI();
            bool roi_enabled = (active_roi.width > 0 && active_roi.height > 0);
            
            // Draw contour outline (green) if enabled
            if (show_contours_) {
                try {
                    if (roi_enabled) {
                        // Draw contour to overlay then blend only within ROI
                        cv::Mat overlay = cv::Mat::zeros(result_image_.size(), result_image_.type());
                        cv::drawContours(overlay, last_result_.contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
                        cv::Mat dst_roi = result_image_(active_roi);
                        cv::Mat overlay_roi = overlay(active_roi);
                        cv::addWeighted(dst_roi, 1.0, overlay_roi, 1.0, 0.0, dst_roi);
                    } else {
                        cv::drawContours(result_image_, last_result_.contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
                    }
                } catch (...) {
                    std::cerr << "Error drawing contour " << i << std::endl;
                }
            }
            
            // Draw bounding box (green if area passes, red if fails) if enabled - thick and prominent
            if (show_bounding_boxes_) {
                try {
                    cv::Rect bbox = last_result_.bounding_boxes[i];
                    
                    // Determine box color based on area threshold (if enabled)
                    cv::Scalar box_color = cv::Scalar(0, 0, 255); // Default: red
                    bool area_passes = true;
                    
                    if (i < last_result_.measurements.size()) {
                        const auto& meas = last_result_.measurements[i];
                        
                        // Check area threshold if enabled
                        if (quality_thresholds_.enable_area_check) {
                            if (quality_thresholds_.min_area > 0 && meas.area_pixels < quality_thresholds_.min_area) {
                                area_passes = false; // Area too small
                            } else if (quality_thresholds_.max_area > 0 && meas.area_pixels > quality_thresholds_.max_area) {
                                area_passes = false; // Area too large
                            } else {
                                area_passes = true; // Area within range
                            }
                            
                            // Green if area passes, red if fails
                            box_color = area_passes ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                        } else {
                            // If area check is not enabled, check if it meets all specs
                            box_color = meas.meets_specs ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                        }
                    }
                    
                    if (roi_enabled) {
                        cv::Rect clipped = bbox & active_roi;
                        if (clipped.area() > 0) {
                            cv::rectangle(result_image_, clipped, box_color, 4);
                            bbox = clipped; // Use clipped for labels positioning
                        } else {
                            continue; // Box fully outside ROI; skip all for this detection
                        }
                    } else {
                        cv::rectangle(result_image_, bbox, box_color, 4);
                    }
                } catch (...) {
                    std::cerr << "Error drawing bbox " << i << std::endl;
                }
            }
            
            // Draw center point (yellow)
            if (i < last_result_.centers.size()) {
                try {
                    const cv::Point2f& c = last_result_.centers[i];
                    if (!roi_enabled || active_roi.contains(c)) {
                        cv::circle(result_image_, c, 8, cv::Scalar(0, 255, 255), -1);
                        cv::circle(result_image_, c, 9, cv::Scalar(255, 255, 255), 2);
                    }
                } catch (...) {
                    std::cerr << "Error drawing center " << i << std::endl;
                }
            }
            
            // Draw label with detection number and dimensions (only if bounding boxes are shown)
            if (show_bounding_boxes_ && i < last_result_.measurements.size()) {
                try {
                    cv::Rect bbox = last_result_.bounding_boxes[i];
                    if (roi_enabled) {
                        bbox = bbox & active_roi;
                        if (bbox.area() <= 0) continue;
                    }
                    const auto& meas = last_result_.measurements[i];
                    
                    // Determine label color based on area threshold (if enabled)
                    cv::Scalar label_bg_color = cv::Scalar(0, 0, 255); // Default: red
                    bool area_passes = true;
                    
                    if (quality_thresholds_.enable_area_check) {
                        if (quality_thresholds_.min_area > 0 && meas.area_pixels < quality_thresholds_.min_area) {
                            area_passes = false; // Area too small
                        } else if (quality_thresholds_.max_area > 0 && meas.area_pixels > quality_thresholds_.max_area) {
                            area_passes = false; // Area too large
                        } else {
                            area_passes = true; // Area within range
                        }
                        
                        // Green if area passes, red if fails
                        label_bg_color = area_passes ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                    } else {
                        // If area check is not enabled, use meets_specs status
                        label_bg_color = meas.meets_specs ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                    }
                    
                    // ID label at top (matches bounding box color)
                    std::string id_label = "#" + std::to_string(i + 1);
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(id_label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
                    cv::rectangle(result_image_, 
                                 cv::Point(bbox.x, bbox.y - text_size.height - 8),
                                 cv::Point(bbox.x + text_size.width + 8, bbox.y),
                                 label_bg_color, -1);
                    cv::putText(result_image_, id_label, 
                               cv::Point(bbox.x + 4, bbox.y - 4),
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                    
                    // Width label (horizontal, bottom center)
                    int w_px = (int)meas.width_pixels;
                    float w_mm = w_px / pixels_per_mm_;
                    std::string width_label = std::to_string(w_px) + "px (" + 
                                             std::to_string((int)w_mm) + "mm)";
                    cv::Size w_size = cv::getTextSize(width_label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    int w_x = bbox.x + (bbox.width - w_size.width) / 2;
                    int w_y = bbox.y + bbox.height + w_size.height + 5;
                    
                    // Background for width
                    cv::rectangle(result_image_,
                                 cv::Point(w_x - 3, w_y - w_size.height - 2),
                                 cv::Point(w_x + w_size.width + 3, w_y + 2),
                                 cv::Scalar(255, 255, 0), -1);
                    cv::putText(result_image_, width_label, cv::Point(w_x, w_y),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                    
                    // Height label (vertical, right center)
                    int h_px = (int)meas.height_pixels;
                    float h_mm = h_px / pixels_per_mm_;
                    std::string height_label = std::to_string(h_px) + "px (" + 
                                              std::to_string((int)h_mm) + "mm)";
                    cv::Size h_size = cv::getTextSize(height_label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    int h_x = bbox.x + bbox.width + 8;
                    int h_y = bbox.y + (bbox.height + h_size.height) / 2;
                    
                    // Background for height
                    cv::rectangle(result_image_,
                                 cv::Point(h_x - 3, h_y - h_size.height - 2),
                                 cv::Point(h_x + h_size.width + 3, h_y + 2),
                                 cv::Scalar(255, 255, 0), -1);
                    cv::putText(result_image_, height_label, cv::Point(h_x, h_y),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
                    
                    // Area label (center of bounding box)
                    float area_px2 = meas.area_pixels;
                    float area_mm2 = area_px2 / (pixels_per_mm_ * pixels_per_mm_);
                    std::string area_label = std::to_string((int)area_px2) + "px² (" + 
                                            std::to_string((int)area_mm2) + "mm²)";
                    cv::Size a_size = cv::getTextSize(area_label, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
                    int a_x = bbox.x + (bbox.width - a_size.width) / 2;
                    int a_y = bbox.y + bbox.height / 2;
                    
                    // Background for area (cyan to distinguish from width/height)
                    cv::rectangle(result_image_,
                                 cv::Point(a_x - 3, a_y - a_size.height - 2),
                                 cv::Point(a_x + a_size.width + 3, a_y + 2),
                                 cv::Scalar(255, 200, 0), -1);
                    cv::putText(result_image_, area_label, cv::Point(a_x, a_y),
                               cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
                    
                } catch (...) {
                    std::cerr << "Error drawing labels " << i << std::endl;
                }
            }
        }
        
        // Add summary info
        std::string summary = "Detected: " + std::to_string(last_result_.dough_count) + " pieces";
        cv::putText(result_image_, summary, cv::Point(20, 40),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 3);
        cv::putText(result_image_, summary, cv::Point(20, 40),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        
        has_results_ = true;
    }
    
    void saveResultImage() {
        if (!has_results_ || result_image_.empty()) return;
        
        std::string filename = "result_" + std::to_string(time(nullptr)) + ".jpg";
        cv::imwrite(filename, result_image_);
        std::cout << "Saved: " << filename << std::endl;
    }
    
    void saveAnnotations() {
        std::cout << "Annotations saved (TODO: implement JSON export)" << std::endl;
    }
    
    void loadAnnotations() {
        std::cout << "Load annotations (TODO: implement JSON import)" << std::endl;
    }
    
    // Recipe management methods
    void refreshRecipeList() {
        recipe_names_ = recipe_manager_->getRecipeNames();
    }
    
    void loadRecipe(const std::string& name) {
        if (recipe_manager_->setActiveRecipe(name)) {
            const Recipe& recipe = recipe_manager_->getActiveRecipe();
            recipe_manager_->applyRecipeToPipeline(vision_pipeline_.get(), recipe);
            
            // Update current index
            for (size_t i = 0; i < recipe_names_.size(); i++) {
                if (recipe_names_[i] == name) {
                    current_recipe_index_ = i;
                    break;
                }
            }
            
            // Update quality thresholds
            quality_thresholds_ = recipe.quality_thresholds;
            
            std::cout << "Loaded recipe: " << name << std::endl;
            
            // Re-run detection if in inference mode with image
            if (!teach_mode_ && has_image_) {
                runInference();
            }
        }
    }
    
    void renderNewRecipeDialog() {
        ImGui::SetNextWindowSize(ImVec2(500, 300));
        ImGui::Begin("Create New Recipe", &show_new_recipe_dialog_, ImGuiWindowFlags_NoResize);
        
        ImGui::TextWrapped("Create a new recipe with current inspection settings.");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        ImGui::Text("Recipe Name:");
        ImGui::InputText("##recipe_name", new_recipe_name_, sizeof(new_recipe_name_));
        
        ImGui::Spacing();
        ImGui::Text("Description:");
        ImGui::InputTextMultiline("##recipe_desc", new_recipe_desc_, sizeof(new_recipe_desc_), ImVec2(-1, 80));
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        if (ImGui::Button("Create", ImVec2(120, 0))) {
            if (strlen(new_recipe_name_) > 0) {
                Recipe recipe;
                recipe.name = new_recipe_name_;
                recipe.description = new_recipe_desc_;
                recipe.quality_thresholds = quality_thresholds_;
                recipe.roi = cv::Rect(0, 0, 640, 480);
                recipe.morph_kernel_size = 5;
                recipe.enable_preprocessing = true;
                
                // Set default values
                recipe.detection_rules.min_area = 500;
                recipe.detection_rules.max_area = 50000;
                recipe.detection_rules.min_circularity = 0.3;
                recipe.detection_rules.max_circularity = 1.0;
                recipe.hsv_lower = cv::Scalar(20, 50, 50);
                recipe.hsv_upper = cv::Scalar(40, 255, 255);
                
                if (recipe_manager_->createRecipe(recipe)) {
                    refreshRecipeList();
                    loadRecipe(recipe.name);
                    show_new_recipe_dialog_ = false;
                    
                    // Clear inputs
                    memset(new_recipe_name_, 0, sizeof(new_recipe_name_));
                    memset(new_recipe_desc_, 0, sizeof(new_recipe_desc_));
                }
            }
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            show_new_recipe_dialog_ = false;
        }
        
        ImGui::End();
    }
    
    void renderRecipeManagerDialog() {
        ImGui::SetNextWindowSize(ImVec2(900, 700));
        ImGui::Begin("Recipe Manager", &show_recipe_dialog_, ImGuiWindowFlags_NoResize);
        
        ImGui::Columns(2);
        ImGui::SetColumnWidth(0, 250);
        
        // Left column: Recipe list
        ImGui::BeginChild("RecipeList", ImVec2(0, -35), true);
        ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "Available Recipes:");
        ImGui::Separator();
        
        for (size_t i = 0; i < recipe_names_.size(); i++) {
            bool is_selected = (current_recipe_index_ == (int)i);
            if (ImGui::Selectable(recipe_names_[i].c_str(), is_selected)) {
                current_recipe_index_ = i;
                if (editing_recipe_) {
                    // Cancel editing when selecting different recipe
                    editing_recipe_ = false;
                }
            }
        }
        
        ImGui::EndChild();
        
        // Buttons below list
        if (ImGui::Button("Load", ImVec2(75, 0))) {
            if (current_recipe_index_ >= 0 && current_recipe_index_ < (int)recipe_names_.size()) {
                editing_recipe_ = false;
                loadRecipe(recipe_names_[current_recipe_index_]);
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Edit", ImVec2(75, 0))) {
            if (current_recipe_index_ >= 0 && current_recipe_index_ < (int)recipe_names_.size()) {
                // Load recipe for editing
                Recipe recipe;
                if (recipe_manager_->loadRecipe(recipe_names_[current_recipe_index_], recipe)) {
                    edited_recipe_ = recipe;
                    editing_recipe_ = true;
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Delete", ImVec2(75, 0))) {
            if (current_recipe_index_ >= 0 && current_recipe_index_ < (int)recipe_names_.size()) {
                recipe_manager_->deleteRecipe(recipe_names_[current_recipe_index_]);
                refreshRecipeList();
                current_recipe_index_ = -1;
                editing_recipe_ = false;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Refresh", ImVec2(75, 0))) {
            refreshRecipeList();
        }
        
        ImGui::NextColumn();
        
        // Right column: Recipe details or editor
        ImGui::BeginChild("RecipeDetails", ImVec2(0, -35), true, ImGuiWindowFlags_AlwaysVerticalScrollbar);
        
        if (editing_recipe_) {
            // EDIT MODE - Full configuration editor
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "EDITING RECIPE");
            ImGui::Separator();
            ImGui::Spacing();
            
            // Recipe name and description
            ImGui::Text("Recipe Name:");
            char name_buf[256];
            strncpy(name_buf, edited_recipe_.name.c_str(), sizeof(name_buf));
            name_buf[255] = 0;
            if (ImGui::InputText("##edit_name", name_buf, sizeof(name_buf))) {
                edited_recipe_.name = name_buf;
            }
            
            ImGui::Spacing();
            ImGui::Text("Description:");
            char desc_buf[512];
            strncpy(desc_buf, edited_recipe_.description.c_str(), sizeof(desc_buf));
            desc_buf[511] = 0;
            if (ImGui::InputTextMultiline("##edit_desc", desc_buf, sizeof(desc_buf), ImVec2(-1, 60))) {
                edited_recipe_.description = desc_buf;
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            // HSV Color Range
            if (ImGui::CollapsingHeader("HSV Color Range", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Lower HSV (H, S, V):");
                float hsv_lower[3] = {
                    static_cast<float>(edited_recipe_.hsv_lower[0]), 
                    static_cast<float>(edited_recipe_.hsv_lower[1]), 
                    static_cast<float>(edited_recipe_.hsv_lower[2])
                };
                if (ImGui::InputFloat3("##hsv_lower", hsv_lower)) {
                    edited_recipe_.hsv_lower = cv::Scalar(hsv_lower[0], hsv_lower[1], hsv_lower[2]);
                }
                ImGui::Text("Upper HSV (H, S, V):");
                float hsv_upper[3] = {
                    static_cast<float>(edited_recipe_.hsv_upper[0]), 
                    static_cast<float>(edited_recipe_.hsv_upper[1]), 
                    static_cast<float>(edited_recipe_.hsv_upper[2])
                };
                if (ImGui::InputFloat3("##hsv_upper", hsv_upper)) {
                    edited_recipe_.hsv_upper = cv::Scalar(hsv_upper[0], hsv_upper[1], hsv_upper[2]);
                }
                ImGui::Spacing();
            }
            
            // Detection Rules
            if (ImGui::CollapsingHeader("Detection Rules", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("Area Range (pixels):");
                ImGui::InputDouble("Min Area##det", &edited_recipe_.detection_rules.min_area, 10.0, 100.0, "%.0f");
                ImGui::InputDouble("Max Area##det", &edited_recipe_.detection_rules.max_area, 100.0, 1000.0, "%.0f");
                
                ImGui::Spacing();
                ImGui::Text("Circularity Range:");
                ImGui::InputDouble("Min Circularity##det", &edited_recipe_.detection_rules.min_circularity, 0.01, 0.1, "%.2f");
                ImGui::InputDouble("Max Circularity##det", &edited_recipe_.detection_rules.max_circularity, 0.01, 0.1, "%.2f");
                
                ImGui::Spacing();
                ImGui::Text("Aspect Ratio Range:");
                ImGui::InputDouble("Min Aspect Ratio##det", &edited_recipe_.detection_rules.min_aspect_ratio, 0.1, 1.0, "%.2f");
                ImGui::InputDouble("Max Aspect Ratio##det", &edited_recipe_.detection_rules.max_aspect_ratio, 0.1, 1.0, "%.2f");
                
                ImGui::Spacing();
                ImGui::InputInt("Expected Count##det", &edited_recipe_.detection_rules.expected_count);
                ImGui::Checkbox("Enforce Count##det", &edited_recipe_.detection_rules.enforce_count);
                ImGui::Spacing();
            }
            
            // Quality Thresholds - Size & Tolerances
            if (ImGui::CollapsingHeader("Quality Thresholds - Sizes & Tolerances", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "Select Thresholds to Monitor:");
                ImGui::Checkbox("✓ Area Check##recipe", &edited_recipe_.quality_thresholds.enable_area_check);
                ImGui::Checkbox("✓ Width Check##recipe", &edited_recipe_.quality_thresholds.enable_width_check);
                ImGui::Checkbox("✓ Length Check##recipe", &edited_recipe_.quality_thresholds.enable_height_check);
                ImGui::Checkbox("✓ Aspect Ratio Check##recipe", &edited_recipe_.quality_thresholds.enable_aspect_ratio_check);
                ImGui::Checkbox("✓ Circularity Check##recipe", &edited_recipe_.quality_thresholds.enable_circularity_check);
                ImGui::Checkbox("✓ Count Check##recipe", &edited_recipe_.quality_thresholds.enable_count_check);
                
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();
                
                // Count Validation (shown if enabled)
                if (edited_recipe_.quality_thresholds.enable_count_check) {
                    ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "Count Validation:");
                    ImGui::InputInt("Expected Count", &edited_recipe_.quality_thresholds.expected_count);
                    ImGui::Checkbox("Enforce Exact Count", &edited_recipe_.quality_thresholds.enforce_exact_count);
                    if (!edited_recipe_.quality_thresholds.enforce_exact_count) {
                        ImGui::InputInt("Min Count", &edited_recipe_.quality_thresholds.min_count);
                        ImGui::InputInt("Max Count", &edited_recipe_.quality_thresholds.max_count);
                    }
                    ImGui::Spacing();
                    ImGui::Separator();
                    ImGui::Spacing();
                }
                
                // Size Validation (shown if enabled)
                if (edited_recipe_.quality_thresholds.enable_area_check || 
                    edited_recipe_.quality_thresholds.enable_width_check || 
                    edited_recipe_.quality_thresholds.enable_height_check) {
                    ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "Size Validation (pixels):");
                    
                    if (edited_recipe_.quality_thresholds.enable_area_check) {
                        ImGui::Text("Area:");
                        ImGui::InputDouble("Min Area##recipe", &edited_recipe_.quality_thresholds.min_area, 10.0, 100.0, "%.0f");
                        ImGui::InputDouble("Max Area##recipe", &edited_recipe_.quality_thresholds.max_area, 100.0, 1000.0, "%.0f");
                        ImGui::Spacing();
                    }
                    
                    if (edited_recipe_.quality_thresholds.enable_width_check) {
                        ImGui::Text("Width:");
                        ImGui::InputDouble("Min Width##recipe", &edited_recipe_.quality_thresholds.min_width, 1.0, 10.0, "%.0f");
                        ImGui::InputDouble("Max Width##recipe", &edited_recipe_.quality_thresholds.max_width, 1.0, 10.0, "%.0f");
                        ImGui::Spacing();
                    }
                    
                    if (edited_recipe_.quality_thresholds.enable_height_check) {
                        ImGui::Text("Height:");
                        ImGui::InputDouble("Min Height##recipe", &edited_recipe_.quality_thresholds.min_height, 1.0, 10.0, "%.0f");
                        ImGui::InputDouble("Max Height##recipe", &edited_recipe_.quality_thresholds.max_height, 1.0, 10.0, "%.0f");
                        ImGui::Spacing();
                    }
                    
                    ImGui::Separator();
                    ImGui::Spacing();
                }
                
                // Shape Validation (shown if enabled)
                if (edited_recipe_.quality_thresholds.enable_aspect_ratio_check || 
                    edited_recipe_.quality_thresholds.enable_circularity_check) {
                    ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "Shape Validation:");
                    
                    if (edited_recipe_.quality_thresholds.enable_aspect_ratio_check) {
                        ImGui::Text("Aspect Ratio:");
                        ImGui::InputDouble("Min Aspect Ratio##qual", &edited_recipe_.quality_thresholds.min_aspect_ratio, 0.1, 1.0, "%.2f");
                        ImGui::InputDouble("Max Aspect Ratio##qual", &edited_recipe_.quality_thresholds.max_aspect_ratio, 0.1, 1.0, "%.2f");
                        ImGui::Spacing();
                    }
                    
                    if (edited_recipe_.quality_thresholds.enable_circularity_check) {
                        ImGui::Text("Circularity:");
                        ImGui::InputDouble("Min Circularity##qual", &edited_recipe_.quality_thresholds.min_circularity, 0.01, 0.1, "%.2f");
                        ImGui::InputDouble("Max Circularity##qual", &edited_recipe_.quality_thresholds.max_circularity, 0.01, 0.1, "%.2f");
                        ImGui::Spacing();
                    }
                    
                    ImGui::Separator();
                    ImGui::Spacing();
                }
                
                ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "Fault Triggers:");
                ImGui::Checkbox("Fail on Undersized", &edited_recipe_.quality_thresholds.fail_on_undersized);
                ImGui::Checkbox("Fail on Oversized", &edited_recipe_.quality_thresholds.fail_on_oversized);
                ImGui::Checkbox("Fail on Count Mismatch", &edited_recipe_.quality_thresholds.fail_on_count_mismatch);
                ImGui::Checkbox("Fail on Shape Defects", &edited_recipe_.quality_thresholds.fail_on_shape_defects);
                ImGui::Spacing();
            }
            
            // ROI Settings
            if (ImGui::CollapsingHeader("Region of Interest (ROI)")) {
                ImGui::Text("ROI Position & Size:");
                int roi_x = edited_recipe_.roi.x;
                int roi_y = edited_recipe_.roi.y;
                int roi_w = edited_recipe_.roi.width;
                int roi_h = edited_recipe_.roi.height;
                
                if (ImGui::InputInt("X##roi", &roi_x)) edited_recipe_.roi.x = roi_x;
                if (ImGui::InputInt("Y##roi", &roi_y)) edited_recipe_.roi.y = roi_y;
                if (ImGui::InputInt("Width##roi", &roi_w)) edited_recipe_.roi.width = roi_w;
                if (ImGui::InputInt("Height##roi", &roi_h)) edited_recipe_.roi.height = roi_h;
                
                ImGui::Spacing();
            }
            
            // Processing Parameters
            if (ImGui::CollapsingHeader("Processing Parameters")) {
                ImGui::InputInt("Morphological Kernel Size", &edited_recipe_.morph_kernel_size);
                if (edited_recipe_.morph_kernel_size < 1) edited_recipe_.morph_kernel_size = 1;
                if (edited_recipe_.morph_kernel_size % 2 == 0) edited_recipe_.morph_kernel_size++;
                ImGui::Checkbox("Enable Preprocessing", &edited_recipe_.enable_preprocessing);
                ImGui::Spacing();
            }
            
        } else if (recipe_manager_->hasActiveRecipe()) {
            // VIEW MODE - Display only
            const Recipe& recipe = recipe_manager_->getActiveRecipe();
            
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Active Recipe");
            ImGui::Separator();
            ImGui::Spacing();
            
            ImGui::Text("Name: %s", recipe.name.c_str());
            ImGui::Text("Description: %s", recipe.description.c_str());
            ImGui::Spacing();
            
            if (ImGui::CollapsingHeader("HSV Range", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("  Lower: [%.0f, %.0f, %.0f]", recipe.hsv_lower[0], recipe.hsv_lower[1], recipe.hsv_lower[2]);
                ImGui::Text("  Upper: [%.0f, %.0f, %.0f]", recipe.hsv_upper[0], recipe.hsv_upper[1], recipe.hsv_upper[2]);
            }
            
            if (ImGui::CollapsingHeader("Detection Rules", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("  Area: %.0f - %.0f", recipe.detection_rules.min_area, recipe.detection_rules.max_area);
                ImGui::Text("  Circularity: %.2f - %.2f", recipe.detection_rules.min_circularity, recipe.detection_rules.max_circularity);
                ImGui::Text("  Aspect Ratio: %.2f - %.2f", recipe.detection_rules.min_aspect_ratio, recipe.detection_rules.max_aspect_ratio);
                ImGui::Text("  Expected Count: %d", recipe.detection_rules.expected_count);
            }
            
            if (ImGui::CollapsingHeader("Quality Thresholds", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("  Expected Count: %d", recipe.quality_thresholds.expected_count);
                ImGui::Text("  Count Range: %d - %d", recipe.quality_thresholds.min_count, recipe.quality_thresholds.max_count);
                ImGui::Text("  Area Range: %.0f - %.0f px", recipe.quality_thresholds.min_area, recipe.quality_thresholds.max_area);
                ImGui::Text("  Width Range: %.0f - %.0f px", recipe.quality_thresholds.min_width, recipe.quality_thresholds.max_width);
                ImGui::Text("  Height Range: %.0f - %.0f px", recipe.quality_thresholds.min_height, recipe.quality_thresholds.max_height);
                ImGui::Text("  Aspect Ratio: %.2f - %.2f", recipe.quality_thresholds.min_aspect_ratio, recipe.quality_thresholds.max_aspect_ratio);
                ImGui::Text("  Circularity: %.2f - %.2f", recipe.quality_thresholds.min_circularity, recipe.quality_thresholds.max_circularity);
            }
            
            ImGui::Spacing();
            if (!recipe.created_date.empty()) {
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Created: %s", recipe.created_date.c_str());
            }
            if (!recipe.modified_date.empty()) {
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Modified: %s", recipe.modified_date.c_str());
            }
        } else {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No recipe loaded.");
            ImGui::Text("Select a recipe from the list and click 'Load' or 'Edit'.");
        }
        
        ImGui::EndChild();
        
        // Action buttons at bottom
        if (editing_recipe_) {
            if (ImGui::Button("Save Changes", ImVec2(150, 0))) {
                // Update modified timestamp
                auto now = std::chrono::system_clock::now();
                auto time = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
                edited_recipe_.modified_date = ss.str();
                
                if (recipe_manager_->saveRecipe(edited_recipe_)) {
                    refreshRecipeList();
                    loadRecipe(edited_recipe_.name);
                    editing_recipe_ = false;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(150, 0))) {
                editing_recipe_ = false;
            }
            ImGui::SameLine();
        }
        
        if (ImGui::Button("Close", ImVec2(120, 0))) {
            if (editing_recipe_) {
                editing_recipe_ = false;
            } else {
                show_recipe_dialog_ = false;
            }
        }
        
        ImGui::Columns(1);
        ImGui::End();
    }
    
    void saveSession() {
        try {
            json session;
            
            // Application state
            session["teach_mode"] = teach_mode_;
            session["show_help"] = show_help_;
            session["current_image_path"] = current_image_path_;
            
            // ROI settings
            session["roi"]["enabled"] = enable_roi_;
            session["roi"]["x"] = roi_rect_.x;
            session["roi"]["y"] = roi_rect_.y;
            session["roi"]["width"] = roi_rect_.width;
            session["roi"]["height"] = roi_rect_.height;
            
            // Display options
            session["display"]["show_bounding_boxes"] = show_bounding_boxes_;
            session["display"]["show_contours"] = show_contours_;
            session["display"]["show_mask_overlay"] = show_mask_overlay_;
            session["display"]["show_measurements"] = show_measurements_;
            
            // Calibration
            session["calibration"]["pixels_per_mm"] = pixels_per_mm_;
            
            // Quality thresholds
            // Save all quality threshold values
            session["quality"]["enable_area_check"] = quality_thresholds_.enable_area_check;
            session["quality"]["enable_width_check"] = quality_thresholds_.enable_width_check;
            session["quality"]["enable_height_check"] = quality_thresholds_.enable_height_check;
            session["quality"]["enable_aspect_ratio_check"] = quality_thresholds_.enable_aspect_ratio_check;
            session["quality"]["enable_circularity_check"] = quality_thresholds_.enable_circularity_check;
            session["quality"]["enable_count_check"] = quality_thresholds_.enable_count_check;
            
            session["quality"]["min_area"] = quality_thresholds_.min_area;
            session["quality"]["max_area"] = quality_thresholds_.max_area;
            session["quality"]["min_width"] = quality_thresholds_.min_width;
            session["quality"]["max_width"] = quality_thresholds_.max_width;
            session["quality"]["min_height"] = quality_thresholds_.min_height;
            session["quality"]["max_height"] = quality_thresholds_.max_height;
            session["quality"]["min_aspect_ratio"] = quality_thresholds_.min_aspect_ratio;
            session["quality"]["max_aspect_ratio"] = quality_thresholds_.max_aspect_ratio;
            session["quality"]["min_circularity"] = quality_thresholds_.min_circularity;
            session["quality"]["max_circularity"] = quality_thresholds_.max_circularity;
            
            session["quality"]["expected_count"] = quality_thresholds_.expected_count;
            session["quality"]["enforce_exact_count"] = quality_thresholds_.enforce_exact_count;
            session["quality"]["min_count"] = quality_thresholds_.min_count;
            session["quality"]["max_count"] = quality_thresholds_.max_count;
            
            session["quality"]["fail_on_undersized"] = quality_thresholds_.fail_on_undersized;
            session["quality"]["fail_on_oversized"] = quality_thresholds_.fail_on_oversized;
            session["quality"]["fail_on_count_mismatch"] = quality_thresholds_.fail_on_count_mismatch;
            session["quality"]["fail_on_shape_defects"] = quality_thresholds_.fail_on_shape_defects;
            
            // Current recipe
            if (recipe_manager_->hasActiveRecipe()) {
                session["active_recipe"] = recipe_manager_->getActiveRecipeName();
            }
            
            // Save polygons (teach mode data)
            session["polygons"] = json::array();
            for (const auto& poly : polygons_) {
                json poly_json;
                poly_json["is_good_sample"] = poly.is_good_sample;
                poly_json["points"] = json::array();
                for (const auto& pt : poly.points) {
                    poly_json["points"].push_back({pt.x, pt.y});
                }
                session["polygons"].push_back(poly_json);
            }
            
            // Write to file
            std::ofstream file("session.json");
            if (file.is_open()) {
                file << session.dump(2);
                file.close();
                std::cout << "Session saved to session.json" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to save session: " << e.what() << std::endl;
        }
    }
    
    void loadConfigIntoUI() {
        try {
            // Load from config/default_config.json to populate UI with defaults if session doesn't have values
            std::ifstream config_file("config/default_config.json");
            if (!config_file.is_open()) {
                std::cout << "Config file not found, using defaults" << std::endl;
                return;
            }
            
            json config;
            config_file >> config;
            config_file.close();
            
            // Load detection thresholds from config (these apply to quality thresholds as defaults)
            if (config.contains("detection")) {
                auto& det = config["detection"];
                // Only set if not already loaded from session
                if (quality_thresholds_.min_area == 0) {
                    quality_thresholds_.min_area = det.value("min_area", 0.0);
                }
                if (quality_thresholds_.max_area == 0) {
                    quality_thresholds_.max_area = det.value("max_area", 100000.0);
                }
                if (quality_thresholds_.min_circularity == 0) {
                    quality_thresholds_.min_circularity = det.value("min_circularity", 0.0);
                }
                if (quality_thresholds_.max_circularity == 0) {
                    quality_thresholds_.max_circularity = det.value("max_circularity", 1.0);
                }
            }
            
            // Load ROI from config if not set
            if (config.contains("roi") && roi_rect_.width == 0 && roi_rect_.height == 0) {
                roi_rect_.x = config["roi"].value("x", 0);
                roi_rect_.y = config["roi"].value("y", 0);
                roi_rect_.width = config["roi"].value("width", 640);
                roi_rect_.height = config["roi"].value("height", 480);
            }
            
            // Apply loaded thresholds to vision pipeline
            vision_pipeline_->updateQualityThresholds(quality_thresholds_);
            
            std::cout << "Config loaded from default_config.json" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load config: " << e.what() << std::endl;
        }
    }
    
    void loadSession() {
        try {
            std::ifstream file("session.json");
            if (!file.is_open()) {
                // No session file exists, that's OK
                return;
            }
            
            json session;
            file >> session;
            file.close();
            
            // Load application state
            if (session.contains("teach_mode")) {
                teach_mode_ = session["teach_mode"];
            }
            if (session.contains("show_help")) {
                show_help_ = session["show_help"];
            }
            
            // Load image if path exists and file is accessible
            if (session.contains("current_image_path")) {
                std::string img_path = session["current_image_path"];
                if (!img_path.empty()) {
                    current_image_ = cv::imread(img_path);
                    if (!current_image_.empty()) {
                        current_image_path_ = img_path;
                        has_image_ = true;
                        std::cout << "Restored image from session: " << img_path << std::endl;
                    }
                }
            }
            
            // Load ROI
            if (session.contains("roi")) {
                enable_roi_ = session["roi"].value("enabled", false);
                roi_rect_.x = session["roi"].value("x", 0);
                roi_rect_.y = session["roi"].value("y", 0);
                roi_rect_.width = session["roi"].value("width", 0);
                roi_rect_.height = session["roi"].value("height", 0);
                if (roi_rect_.width > 0 && roi_rect_.height > 0) {
                    vision_pipeline_->updateROI(roi_rect_);
                }
            }
            
            // Load display options
            if (session.contains("display")) {
                show_bounding_boxes_ = session["display"].value("show_bounding_boxes", true);
                show_contours_ = session["display"].value("show_contours", true);
                show_mask_overlay_ = session["display"].value("show_mask_overlay", true);
                show_measurements_ = session["display"].value("show_measurements", true);
            }
            
            // Load calibration
            if (session.contains("calibration")) {
                pixels_per_mm_ = session["calibration"].value("pixels_per_mm", 1.0f);
            }
            
            // Load quality thresholds - load ALL threshold values and enable flags
            if (session.contains("quality")) {
                auto& q = session["quality"];
                
                // Load enable flags
                quality_thresholds_.enable_area_check = q.value("enable_area_check", false);
                quality_thresholds_.enable_width_check = q.value("enable_width_check", false);
                quality_thresholds_.enable_height_check = q.value("enable_height_check", false);
                quality_thresholds_.enable_aspect_ratio_check = q.value("enable_aspect_ratio_check", false);
                quality_thresholds_.enable_circularity_check = q.value("enable_circularity_check", false);
                quality_thresholds_.enable_count_check = q.value("enable_count_check", false);
                
                // Load threshold values
                quality_thresholds_.min_area = q.value("min_area", quality_thresholds_.min_area);
                quality_thresholds_.max_area = q.value("max_area", quality_thresholds_.max_area);
                quality_thresholds_.min_width = q.value("min_width", quality_thresholds_.min_width);
                quality_thresholds_.max_width = q.value("max_width", quality_thresholds_.max_width);
                quality_thresholds_.min_height = q.value("min_height", quality_thresholds_.min_height);
                quality_thresholds_.max_height = q.value("max_height", quality_thresholds_.max_height);
                quality_thresholds_.min_aspect_ratio = q.value("min_aspect_ratio", quality_thresholds_.min_aspect_ratio);
                quality_thresholds_.max_aspect_ratio = q.value("max_aspect_ratio", quality_thresholds_.max_aspect_ratio);
                quality_thresholds_.min_circularity = q.value("min_circularity", quality_thresholds_.min_circularity);
                quality_thresholds_.max_circularity = q.value("max_circularity", quality_thresholds_.max_circularity);
                
                // Load count settings
                quality_thresholds_.expected_count = q.value("expected_count", quality_thresholds_.expected_count);
                quality_thresholds_.enforce_exact_count = q.value("enforce_exact_count", quality_thresholds_.enforce_exact_count);
                quality_thresholds_.min_count = q.value("min_count", quality_thresholds_.min_count);
                quality_thresholds_.max_count = q.value("max_count", quality_thresholds_.max_count);
                
                // Load fault triggers
                quality_thresholds_.fail_on_undersized = q.value("fail_on_undersized", quality_thresholds_.fail_on_undersized);
                quality_thresholds_.fail_on_oversized = q.value("fail_on_oversized", quality_thresholds_.fail_on_oversized);
                quality_thresholds_.fail_on_count_mismatch = q.value("fail_on_count_mismatch", quality_thresholds_.fail_on_count_mismatch);
                quality_thresholds_.fail_on_shape_defects = q.value("fail_on_shape_defects", quality_thresholds_.fail_on_shape_defects);
                
                // Apply loaded thresholds to vision pipeline
                vision_pipeline_->updateQualityThresholds(quality_thresholds_);
            }
            
            // Also load from config file if session doesn't have values
            loadConfigIntoUI();
            
            // Load active recipe
            if (session.contains("active_recipe")) {
                std::string recipe_name = session["active_recipe"];
                if (!recipe_name.empty()) {
                    loadRecipe(recipe_name);
                }
            }
            
            // Load polygons (teach mode data)
            if (session.contains("polygons") && session["polygons"].is_array()) {
                polygons_.clear();
                for (const auto& poly_json : session["polygons"]) {
                    Polygon poly;
                    poly.is_good_sample = poly_json.value("is_good_sample", true);
                    poly.color = poly.is_good_sample ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                    
                    if (poly_json.contains("points") && poly_json["points"].is_array()) {
                        for (const auto& pt_json : poly_json["points"]) {
                            if (pt_json.is_array() && pt_json.size() >= 2) {
                                cv::Point2f pt(pt_json[0], pt_json[1]);
                                poly.points.push_back(pt);
                            }
                        }
                    }
                    
                    if (!poly.points.empty()) {
                        polygons_.push_back(poly);
                    }
                }
                std::cout << "Restored " << polygons_.size() << " polygons from session" << std::endl;
            }
            
            std::cout << "Session restored from session.json" << std::endl;
            session_loaded_ = true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load session: " << e.what() << std::endl;
            // Even if session load fails, mark as loaded so config can be used
            session_loaded_ = true;
        }
    }
};

} // namespace country_style

int main() {
    std::cout << "Country Style Dough Inspector - Polygon Teaching Edition" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    country_style::PolygonTeachingApp app;
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize" << std::endl;
        return -1;
    }
    
    app.run();
    app.shutdown();
    
    return 0;
}
