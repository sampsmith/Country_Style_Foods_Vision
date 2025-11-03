#ifndef VISION_PIPELINE_H
#define VISION_PIPELINE_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <chrono>
#include "fast_color_segmentation.h"
#include "contour_detector.h"
#include "rule_engine.h"

namespace country_style {

struct DetectionResult {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Rect> bounding_boxes;
    std::vector<cv::Point2f> centers;
    int dough_count;
    bool is_valid;
    double confidence;
    std::string message;
    
    // Performance metrics
    double segmentation_time_ms;
    double contour_time_ms;
    double rule_time_ms;
    double total_time_ms;
};

class VisionPipeline {
public:
    VisionPipeline();
    ~VisionPipeline();
    
    // Initialize with configuration
    bool initialize(const std::string& config_path);
    
    // Process a single frame (target: <10ms total)
    DetectionResult processFrame(const cv::Mat& frame);
    
    // Update configuration parameters
    void updateColorRange(const cv::Scalar& lower, const cv::Scalar& upper);
    void updateROI(const cv::Rect& roi);
    void updateDetectionRules(const DetectionRules& rules);
    
    // Get intermediate processing results
    const cv::Mat& getSegmentedMask() const { return segmented_mask_; }
    const cv::Mat& getHsvFrame() const { return hsv_frame_; }
    
    // Render detection overlay on frame
    void renderDetections(cv::Mat& frame, const DetectionResult& result);
    
    // Performance monitoring
    struct PerformanceStats {
        double avg_total_ms;
        double avg_segmentation_ms;
        double avg_contour_ms;
        double min_total_ms;
        double max_total_ms;
        int frame_count;
    };
    
    PerformanceStats getPerformanceStats() const;
    void resetPerformanceStats();
    
private:
    // Vision components
    std::unique_ptr<FastColorSegmentation> color_segmenter_;
    std::unique_ptr<ContourDetector> contour_detector_;
    std::unique_ptr<RuleEngine> rule_engine_;
    
    // Processing state
    cv::Mat segmented_mask_;
    cv::Mat hsv_frame_;
    cv::Rect roi_;
    bool is_initialized_;
    
    // Pre-allocated buffers for zero-copy operations
    cv::Mat roi_frame_;
    std::vector<std::vector<cv::Point>> temp_contours_;
    
    // Performance tracking
    std::vector<double> frame_times_;
    std::vector<double> segmentation_times_;
    std::vector<double> contour_times_;
    
    // Helper for timing
    class Timer {
    public:
        Timer() : start_(std::chrono::high_resolution_clock::now()) {}
        double elapsedMs() const {
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start_).count();
        }
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };
};

} // namespace country_style

#endif // VISION_PIPELINE_H
