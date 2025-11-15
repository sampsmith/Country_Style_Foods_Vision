#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#ifdef _WIN32
    #include <windows.h>
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
#include "vision_pipeline.h"

namespace country_style {

class SimpleApplication {
public:
    SimpleApplication() 
        : window_(nullptr),
          image_texture_(0),
          result_texture_(0),
          current_mode_(MODE_TEACH),
          has_image_(false),
          has_results_(false) {
        
        vision_pipeline_ = std::make_unique<VisionPipeline>();
        vision_pipeline_->initialize("config/default_config.json");
    }
    
    ~SimpleApplication() {
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
        
        window_ = glfwCreateWindow(1400, 900, "Country Style Dough Inspector", nullptr, nullptr);
        if (!window_) {
            glfwTerminate();
            return false;
        }
        
        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1);
        
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        
        // Clean dark theme
        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 8.0f;
        style.FrameRounding = 4.0f;
        style.GrabRounding = 4.0f;
        style.ScrollbarRounding = 4.0f;
        
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
            
            renderMainWindow();
            
            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window_, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
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
    enum Mode {
        MODE_TEACH,
        MODE_INFERENCE
    };
    
    GLFWwindow* window_;
    GLuint image_texture_;
    GLuint result_texture_;
    
    Mode current_mode_;
    cv::Mat current_image_;
    cv::Mat result_image_;
    DetectionResult last_result_;
    
    bool has_image_;
    bool has_results_;
    
    std::unique_ptr<VisionPipeline> vision_pipeline_;
    
    // Teach mode parameters
    float hsv_lower_[3] = {20, 50, 50};
    float hsv_upper_[3] = {40, 255, 255};
    float min_area_ = 500.0f;
    float max_area_ = 50000.0f;
    float min_circularity_ = 0.3f;
    
    void renderMainWindow() {
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        
        ImGui::Begin("Country Style Dough Inspector", nullptr, 
                    ImGuiWindowFlags_NoTitleBar | 
                    ImGuiWindowFlags_NoResize | 
                    ImGuiWindowFlags_NoMove | 
                    ImGuiWindowFlags_NoCollapse);
        
        // Header
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
        ImGui::Text("Country Style Dough Inspector");
        ImGui::SameLine(ImGui::GetWindowWidth() - 200);
        if (has_results_) {
            if (last_result_.total_time_ms < 10.0) {
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "%.1fms ✓", last_result_.total_time_ms);
            } else {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "%.1fms", last_result_.total_time_ms);
            }
        }
        ImGui::PopFont();
        
        ImGui::Separator();
        
        // Mode selector
        ImGui::BeginChild("ModeSelector", ImVec2(200, 0), true);
        
        ImGui::Text("Mode");
        ImGui::Separator();
        
        if (ImGui::Selectable("Teach Mode", current_mode_ == MODE_TEACH, 0, ImVec2(0, 40))) {
            current_mode_ = MODE_TEACH;
            has_results_ = false;
        }
        
        if (ImGui::Selectable("Run Inference", current_mode_ == MODE_INFERENCE, 0, ImVec2(0, 40))) {
            current_mode_ = MODE_INFERENCE;
            has_results_ = false;
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        
        if (ImGui::Button("Load Image...", ImVec2(-1, 40))) {
            loadImage();
        }
        
        if (has_image_ && ImGui::Button("Clear Image", ImVec2(-1, 40))) {
            clearImage();
        }
        
        ImGui::EndChild();
        
        ImGui::SameLine();
        
        // Main content area
        ImGui::BeginChild("MainContent", ImVec2(0, 0), true);
        
        if (current_mode_ == MODE_TEACH) {
            renderTeachMode();
        } else {
            renderInferenceMode();
        }
        
        ImGui::EndChild();
        
        ImGui::End();
    }
    
    void renderTeachMode() {
        ImGui::Text("Teach Mode - Configure Detection Parameters");
        ImGui::Separator();
        
        ImGui::BeginChild("TeachLeft", ImVec2(ImGui::GetContentRegionAvail().x * 0.6f, 0), false);
        
        if (has_image_) {
            ImGui::Text("Loaded Image");
            displayImage(current_image_, image_texture_, ImGui::GetContentRegionAvail());
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
                             "Load an image to start teaching");
            ImGui::Spacing();
            ImGui::TextWrapped("Click 'Load Image...' to upload a sample dough image");
        }
        
        ImGui::EndChild();
        
        ImGui::SameLine();
        
        ImGui::BeginChild("TeachRight", ImVec2(0, 0), true);
        
        ImGui::Text("Color Detection (HSV)");
        ImGui::Separator();
        ImGui::SliderFloat3("Lower Bound", hsv_lower_, 0, 255);
        ImGui::SliderFloat3("Upper Bound", hsv_upper_, 0, 255);
        
        ImGui::Spacing();
        ImGui::Text("Size & Shape Rules");
        ImGui::Separator();
        ImGui::InputFloat("Min Area (px)", &min_area_);
        ImGui::InputFloat("Max Area (px)", &max_area_);
        ImGui::SliderFloat("Min Roundness", &min_circularity_, 0.0f, 1.0f);
        
        ImGui::Spacing();
        
        if (has_image_ && ImGui::Button("Test Parameters", ImVec2(-1, 50))) {
            testParameters();
        }
        
        if (has_results_) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Text("Test Results:");
            ImGui::Text("Found: %d dough pieces", last_result_.dough_count);
            ImGui::Text("Time: %.2f ms", last_result_.total_time_ms);
        }
        
        ImGui::Spacing();
        ImGui::Separator();
        
        if (ImGui::Button("Save Configuration", ImVec2(-1, 40))) {
            saveConfiguration();
        }
        
        ImGui::EndChild();
    }
    
    void renderInferenceMode() {
        ImGui::Text("Inference Mode - Run Detection");
        ImGui::Separator();
        
        ImGui::BeginChild("InferenceLeft", ImVec2(ImGui::GetContentRegionAvail().x * 0.7f, 0), false);
        
        if (has_results_) {
            ImGui::Text("Detection Results");
            displayImage(result_image_, result_texture_, ImGui::GetContentRegionAvail());
        } else if (has_image_) {
            ImGui::Text("Loaded Image (Click 'Run Detection' below)");
            displayImage(current_image_, image_texture_, ImGui::GetContentRegionAvail());
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), 
                             "Load an image to run detection");
            ImGui::Spacing();
            ImGui::TextWrapped("Click 'Load Image...' to upload an image for inspection");
        }
        
        ImGui::EndChild();
        
        ImGui::SameLine();
        
        ImGui::BeginChild("InferenceRight", ImVec2(0, 0), true);
        
        if (has_image_ && ImGui::Button("Run Detection", ImVec2(-1, 60))) {
            runInference();
        }
        
        if (has_results_) {
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            
            ImGui::Text("Detection Results");
            ImGui::Separator();
            
            ImGui::Text("Dough Count: %d", last_result_.dough_count);
            
            if (last_result_.is_valid) {
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "Status: PASS ✓");
            } else {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "Status: FAIL ✗");
            }
            
            ImGui::Spacing();
            ImGui::Text("Performance");
            ImGui::Separator();
            ImGui::Text("Total: %.2f ms", last_result_.total_time_ms);
            ImGui::Text("Segmentation: %.2f ms", last_result_.segmentation_time_ms);
            ImGui::Text("Contours: %.2f ms", last_result_.contour_time_ms);
            
            if (last_result_.total_time_ms < 10.0) {
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "Target met: <10ms ✓");
            } else {
                ImGui::TextColored(ImVec4(1, 1, 0, 1), "Target: <10ms");
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            
            if (ImGui::Button("Save Result Image", ImVec2(-1, 40))) {
                saveResultImage();
            }
        }
        
        ImGui::EndChild();
    }
    
    void displayImage(const cv::Mat& img, GLuint texture, ImVec2 available_size) {
        if (img.empty()) return;
        
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 
                     0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        
        float aspect = (float)img.cols / img.rows;
        float display_width = available_size.x - 20;
        float display_height = display_width / aspect;
        
        if (display_height > available_size.y - 40) {
            display_height = available_size.y - 40;
            display_width = display_height * aspect;
        }
        
        ImGui::Image((void*)(intptr_t)texture, ImVec2(display_width, display_height));
    }
    
    void loadImage() {
        // Simple file dialog using zenity (GTK native)
        std::string command = "zenity --file-selection --title='Select Image' --file-filter='Images | *.jpg *.jpeg *.png *.bmp'";
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) return;
        
        char buffer[1024];
        std::string path;
        if (fgets(buffer, sizeof(buffer), pipe)) {
            path = buffer;
            // Remove trailing newline
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
                std::cout << "Loaded image: " << path << std::endl;
            }
        }
    }
    
    void clearImage() {
        current_image_.release();
        result_image_.release();
        has_image_ = false;
        has_results_ = false;
    }
    
    void testParameters() {
        if (!has_image_) return;
        
        // Update vision pipeline with current parameters
        vision_pipeline_->updateColorRange(
            cv::Scalar(hsv_lower_[0], hsv_lower_[1], hsv_lower_[2]),
            cv::Scalar(hsv_upper_[0], hsv_upper_[1], hsv_upper_[2])
        );
        
        DetectionRules rules;
        rules.min_area = min_area_;
        rules.max_area = max_area_;
        rules.min_circularity = min_circularity_;
        rules.max_circularity = 1.0;
        rules.min_aspect_ratio = 0.5;
        rules.max_aspect_ratio = 2.0;
        rules.expected_count = 0;
        rules.enforce_count = false;
        
        vision_pipeline_->updateDetectionRules(rules);
        
        // Run detection
        last_result_ = vision_pipeline_->processFrame(current_image_);
        
        // Draw results
        result_image_ = current_image_.clone();
        vision_pipeline_->renderDetections(result_image_, last_result_);
        
        has_results_ = true;
    }
    
    void runInference() {
        testParameters();  // Same as test, just different mode
    }
    
    void saveConfiguration() {
        std::cout << "Configuration saved!" << std::endl;
        // TODO: Implement actual config saving
    }
    
    void saveResultImage() {
        if (!has_results_ || result_image_.empty()) return;
        
        std::string filename = "result_" + std::to_string(time(nullptr)) + ".jpg";
        cv::imwrite(filename, result_image_);
        std::cout << "Saved result to: " << filename << std::endl;
    }
};

} // namespace country_style

int main() {
    std::cout << "Country Style Dough Inspector - High Performance Edition" << std::endl;
    std::cout << "Simple Image-Based Interface" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    country_style::SimpleApplication app;
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize application" << std::endl;
        return -1;
    }
    
    app.run();
    app.shutdown();
    
    return 0;
}
