#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include "vision_pipeline.h"

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
          image_offset_y_(0.0f) {
        
        vision_pipeline_ = std::make_unique<VisionPipeline>();
        vision_pipeline_->initialize("config/default_config.json");
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
        
        // Larger font for better readability
        io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16.0f);
        
        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 6.0f;
        style.FrameRounding = 3.0f;
        style.WindowPadding = ImVec2(10, 10);
        style.FramePadding = ImVec2(8, 4);
        style.ItemSpacing = ImVec2(8, 8);
        
        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init("#version 330");
        
        glGenTextures(1, &image_texture_);
        glGenTextures(1, &result_texture_);
        
        return true;
    }
    
    void run() {
        while (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();
            
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            
            renderUI();
            
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window_, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.12f, 0.12f, 0.12f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            
            glfwSwapBuffers(window_);
        }
    }
    
    void shutdown() {
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
    
    bool has_image_;
    bool has_results_;
    bool is_drawing_;
    bool show_help_;
    bool teach_mode_;
    
    float image_scale_;
    float image_offset_x_;
    float image_offset_y_;
    
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
    cv::Point2f calib_end_;
    bool drawing_calib_line_ = false;
    
    std::unique_ptr<VisionPipeline> vision_pipeline_;
    
    ImVec2 image_display_pos_;
    ImVec2 image_display_size_;
    
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
            ImGui::EndMenuBar();
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
        
        if (!has_image_) {
            if (ImGui::Button("Load Test Image", ImVec2(-1, 50))) {
                loadImage();
            }
        } else {
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
            
            // Quality thresholds
            ImGui::TextColored(ImVec4(1, 0.8, 0.4, 1), "Thresholds (pixels):");
            
            // Width tolerance
            static int target_width = 200;
            static int width_tolerance = 50;
            ImGui::InputInt("Target Width (px)", &target_width);
            ImGui::InputInt("Width Tolerance (±)", &width_tolerance);
            
            // Length tolerance
            static int target_length = 300;
            static int length_tolerance = 50;
            ImGui::InputInt("Target Length (px)", &target_length);
            ImGui::InputInt("Length Tolerance (±)", &length_tolerance);
            
            // Update thresholds
            quality_thresholds_.min_width = std::max(1, target_width - width_tolerance);
            quality_thresholds_.max_width = target_width + width_tolerance;
            quality_thresholds_.min_height = std::max(1, target_length - length_tolerance);
            quality_thresholds_.max_height = target_length + length_tolerance;
            quality_thresholds_.fail_on_undersized = true;
            quality_thresholds_.fail_on_oversized = true;
            
            ImGui::Text("Valid range:");
            ImGui::Text(" Width: %d - %d px", (int)quality_thresholds_.min_width, (int)quality_thresholds_.max_width);
            ImGui::Text(" Length: %d - %d px", (int)quality_thresholds_.min_height, (int)quality_thresholds_.max_height);
            
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
                    
                    if (ImGui::BeginTable("Measurements", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY, ImVec2(0, 150))) {
                        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 30);
                        ImGui::TableSetupColumn("W(px)", ImGuiTableColumnFlags_WidthFixed, 50);
                        ImGui::TableSetupColumn("W(mm)", ImGuiTableColumnFlags_WidthFixed, 50);
                        ImGui::TableSetupColumn("L(px)", ImGuiTableColumnFlags_WidthFixed, 50);
                        ImGui::TableSetupColumn("L(mm)", ImGuiTableColumnFlags_WidthFixed, 50);
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
                            if (quality_thresholds_.min_width > 0 && 
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
                            if (quality_thresholds_.min_height > 0 && 
                                (meas.height_pixels < quality_thresholds_.min_height || meas.height_pixels > quality_thresholds_.max_height)) {
                                ImGui::TextColored(ImVec4(1, 0, 0, 1), "%d", (int)meas.height_pixels);
                            } else {
                                ImGui::Text("%d", (int)meas.height_pixels);
                            }
                            
                            // Length (mm)
                            ImGui::TableSetColumnIndex(4);
                            float h_mm = meas.height_pixels / pixels_per_mm_;
                            ImGui::Text("%d", (int)h_mm);
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
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        
        // Use GL_UNPACK_ALIGNMENT for proper row alignment
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 
                     0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    }
    
    void loadImage() {
        std::string command = "zenity --file-selection --title='Select Image' --file-filter='Images | *.jpg *.jpeg *.png *.bmp'";
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) return;
        
        char buffer[1024];
        std::string path;
        if (fgets(buffer, sizeof(buffer), pipe)) {
            path = buffer;
            if (!path.empty() && path[path.length()-1] == '\n') {
                path.erase(path.length()-1);
            }
        }
        pclose(pipe);
        
        if (!path.empty()) {
            current_image_ = cv::imread(path);
            if (!current_image_.empty()) {
                has_image_ = true;
                has_results_ = false;
                polygons_.clear();
                current_polygon_.clear();
                is_drawing_ = false;
                std::cout << "Loaded: " << path << " (" << current_image_.cols << "x" << current_image_.rows << ")" << std::endl;
            }
        }
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
            
            // Draw contour outline (green) if enabled
            if (show_contours_) {
                try {
                    cv::drawContours(result_image_, last_result_.contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
                } catch (...) {
                    std::cerr << "Error drawing contour " << i << std::endl;
                }
            }
            
            // Draw bounding box (bright red) if enabled - thick and prominent
            if (show_bounding_boxes_) {
                try {
                    cv::rectangle(result_image_, last_result_.bounding_boxes[i], 
                                 cv::Scalar(0, 0, 255), 4);
                } catch (...) {
                    std::cerr << "Error drawing bbox " << i << std::endl;
                }
            }
            
            // Draw center point (yellow)
            if (i < last_result_.centers.size()) {
                try {
                    cv::circle(result_image_, last_result_.centers[i], 8, 
                              cv::Scalar(0, 255, 255), -1);
                    cv::circle(result_image_, last_result_.centers[i], 9, 
                              cv::Scalar(255, 255, 255), 2);
                } catch (...) {
                    std::cerr << "Error drawing center " << i << std::endl;
                }
            }
            
            // Draw label with detection number and dimensions (only if bounding boxes are shown)
            if (show_bounding_boxes_ && i < last_result_.measurements.size()) {
                try {
                    cv::Rect bbox = last_result_.bounding_boxes[i];
                    const auto& meas = last_result_.measurements[i];
                    
                    // ID label at top
                    std::string id_label = "#" + std::to_string(i + 1);
                    int baseline = 0;
                    cv::Size text_size = cv::getTextSize(id_label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
                    cv::rectangle(result_image_, 
                                 cv::Point(bbox.x, bbox.y - text_size.height - 8),
                                 cv::Point(bbox.x + text_size.width + 8, bbox.y),
                                 cv::Scalar(0, 0, 255), -1);
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
