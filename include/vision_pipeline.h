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

// Individual detection measurements
struct DetectionMeasurement {
    int id;
    double area_pixels;
    double width_pixels;
    double height_pixels;
    double aspect_ratio;
    double circularity;
    cv::Point2f center;
    cv::Rect bbox;
    bool meets_specs;  // Individual pass/fail
    std::string fault_reason;
};

// Quality thresholds for fault detection
struct QualityThresholds {
    // Count validation
    int expected_count;
    bool enforce_exact_count;
    int min_count;
    int max_count;
    
    // Size validation (pixels)
    double min_area;
    double max_area;
    double min_width;
    double max_width;
    double min_height;
    double max_height;
    
    // Shape validation
    double min_aspect_ratio;
    double max_aspect_ratio;
    double min_circularity;
    double max_circularity;
    
    // Fault triggers
    bool fail_on_undersized;
    bool fail_on_oversized;
    bool fail_on_count_mismatch;
    bool fail_on_shape_defects;
};

struct DetectionResult {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Rect> bounding_boxes;
    std::vector<cv::Point2f> centers;
    std::vector<DetectionMeasurement> measurements;  // Detailed per-detection data
    
    int dough_count;
    bool is_valid;  // Overall pass/fail
    double confidence;
    std::string message;
    
    // Fault flags
    bool fault_count_low;
    bool fault_count_high;
    bool fault_undersized;
    bool fault_oversized;
    bool fault_shape_defect;
    std::vector<std::string> fault_messages;
    
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
    void updateQualityThresholds(const QualityThresholds& thresholds);
    
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
    QualityThresholds quality_thresholds_;
    
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
