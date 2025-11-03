#include "config_manager.h"
#include <fstream>
#include <iostream>

namespace country_style {

ConfigManager::ConfigManager() {
    // Set default configuration
    config_.color_lower = cv::Scalar(20, 50, 50);
    config_.color_upper = cv::Scalar(40, 255, 255);
    config_.roi = cv::Rect(0, 0, 640, 480);
    config_.min_area = 500.0;
    config_.max_area = 50000.0;
    config_.min_circularity = 0.3;
    config_.max_circularity = 1.0;
    config_.camera_index = 0;
    config_.frame_width = 640;
    config_.frame_height = 480;
    config_.fps = 30;
    config_.morph_kernel_size = 5;
    config_.enable_preprocessing = true;
}

ConfigManager::~ConfigManager() {}

bool ConfigManager::loadConfig(const std::string& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Could not open config file: " << file_path << std::endl;
            return false;
        }
        
        nlohmann::json j;
        file >> j;
        
        config_ = jsonToConfig(j);
        config_path_ = file_path;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigManager::saveConfig(const std::string& file_path) {
    try {
        std::ofstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Could not open file for writing: " << file_path << std::endl;
            return false;
        }
        
        nlohmann::json j = configToJson();
        file << j.dump(4); // Pretty print with 4 spaces
        
        config_path_ = file_path;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving config: " << e.what() << std::endl;
        return false;
    }
}

VisionConfig ConfigManager::getConfig() const {
    return config_;
}

void ConfigManager::setConfig(const VisionConfig& config) {
    config_ = config;
}

void ConfigManager::updateColorRange(const cv::Scalar& lower, const cv::Scalar& upper) {
    config_.color_lower = lower;
    config_.color_upper = upper;
}

void ConfigManager::updateROI(const cv::Rect& roi) {
    config_.roi = roi;
}

nlohmann::json ConfigManager::configToJson() {
    nlohmann::json j;
    
    j["color_segmentation"]["lower"] = {
        config_.color_lower[0], config_.color_lower[1], config_.color_lower[2]
    };
    j["color_segmentation"]["upper"] = {
        config_.color_upper[0], config_.color_upper[1], config_.color_upper[2]
    };
    
    j["roi"]["x"] = config_.roi.x;
    j["roi"]["y"] = config_.roi.y;
    j["roi"]["width"] = config_.roi.width;
    j["roi"]["height"] = config_.roi.height;
    
    j["detection"]["min_area"] = config_.min_area;
    j["detection"]["max_area"] = config_.max_area;
    j["detection"]["min_circularity"] = config_.min_circularity;
    j["detection"]["max_circularity"] = config_.max_circularity;
    
    j["camera"]["index"] = config_.camera_index;
    j["camera"]["width"] = config_.frame_width;
    j["camera"]["height"] = config_.frame_height;
    j["camera"]["fps"] = config_.fps;
    
    j["processing"]["morph_kernel_size"] = config_.morph_kernel_size;
    j["processing"]["enable_preprocessing"] = config_.enable_preprocessing;
    
    return j;
}

VisionConfig ConfigManager::jsonToConfig(const nlohmann::json& j) {
    VisionConfig cfg;
    
    if (j.contains("color_segmentation")) {
        auto lower = j["color_segmentation"]["lower"];
        auto upper = j["color_segmentation"]["upper"];
        cfg.color_lower = cv::Scalar(lower[0], lower[1], lower[2]);
        cfg.color_upper = cv::Scalar(upper[0], upper[1], upper[2]);
    }
    
    if (j.contains("roi")) {
        cfg.roi = cv::Rect(
            j["roi"]["x"], j["roi"]["y"],
            j["roi"]["width"], j["roi"]["height"]
        );
    }
    
    if (j.contains("detection")) {
        cfg.min_area = j["detection"]["min_area"];
        cfg.max_area = j["detection"]["max_area"];
        cfg.min_circularity = j["detection"]["min_circularity"];
        cfg.max_circularity = j["detection"]["max_circularity"];
    }
    
    if (j.contains("camera")) {
        cfg.camera_index = j["camera"]["index"];
        cfg.frame_width = j["camera"]["width"];
        cfg.frame_height = j["camera"]["height"];
        cfg.fps = j["camera"]["fps"];
    }
    
    if (j.contains("processing")) {
        cfg.morph_kernel_size = j["processing"]["morph_kernel_size"];
        cfg.enable_preprocessing = j["processing"]["enable_preprocessing"];
    }
    
    return cfg;
}

} // namespace country_style
