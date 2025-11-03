#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include <string>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

namespace country_style {

struct VisionConfig {
    // Color segmentation parameters
    cv::Scalar color_lower;
    cv::Scalar color_upper;
    
    // Region of interest
    cv::Rect roi;
    
    // Detection rules
    double min_area;
    double max_area;
    double min_circularity;
    double max_circularity;
    
    // Camera settings
    int camera_index;
    int frame_width;
    int frame_height;
    int fps;
    
    // Processing settings
    int morph_kernel_size;
    bool enable_preprocessing;
};

class ConfigManager {
public:
    ConfigManager();
    ~ConfigManager();

    // Load configuration from JSON file
    bool loadConfig(const std::string& file_path);
    
    // Save configuration to JSON file
    bool saveConfig(const std::string& file_path);
    
    // Get/Set configuration
    VisionConfig getConfig() const;
    void setConfig(const VisionConfig& config);
    
    // Update specific parameters
    void updateColorRange(const cv::Scalar& lower, const cv::Scalar& upper);
    void updateROI(const cv::Rect& roi);

private:
    VisionConfig config_;
    std::string config_path_;
    
    nlohmann::json configToJson();
    VisionConfig jsonToConfig(const nlohmann::json& j);
};

} // namespace country_style

#endif // CONFIG_MANAGER_H
