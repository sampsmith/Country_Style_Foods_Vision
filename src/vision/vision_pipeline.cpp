#include "vision_pipeline.h"
#include "config_manager.h"
#include <iostream>

namespace country_style {

VisionPipeline::VisionPipeline()
    : is_initialized_(false) {
    color_segmenter_ = std::make_unique<FastColorSegmentation>();
    contour_detector_ = std::make_unique<ContourDetector>();
    rule_engine_ = std::make_unique<RuleEngine>();
}

VisionPipeline::~VisionPipeline() {}

bool VisionPipeline::initialize(const std::string& config_path) {
    ConfigManager config_mgr;
    if (!config_mgr.loadConfig(config_path)) {
        std::cerr << "Warning: Could not load config, using defaults" << std::endl;
        // Set default values
        color_segmenter_->setColorRange(
            cv::Scalar(20, 50, 50),
            cv::Scalar(40, 255, 255)
        );
        roi_ = cv::Rect(0, 0, 640, 480);
    } else {
        VisionConfig cfg = config_mgr.getConfig();
        color_segmenter_->setColorRange(cfg.color_lower, cfg.color_upper);
        roi_ = cfg.roi;
        
        DetectionRules rules;
        rules.min_area = cfg.min_area;
        rules.max_area = cfg.max_area;
        rules.min_circularity = cfg.min_circularity;
        rules.max_circularity = cfg.max_circularity;
        rules.min_aspect_ratio = 0.5;
        rules.max_aspect_ratio = 2.0;
        rules.expected_count = 0;
        rules.enforce_count = false;
        
        rule_engine_->setRules(rules);
    }
    
    is_initialized_ = true;
    return true;
}

DetectionResult VisionPipeline::processFrame(const cv::Mat& frame) {
    Timer total_timer;
    DetectionResult result;
    result.dough_count = 0;
    result.is_valid = false;
    result.confidence = 0.0;
    
    if (frame.empty() || !is_initialized_) {
        result.message = "Invalid frame or not initialized";
        return result;
    }
    
    // Apply ROI if set (zero-copy view)
    cv::Mat roi_frame = frame;
    if (roi_.width > 0 && roi_.height > 0 && 
        roi_.x >= 0 && roi_.y >= 0 &&
        roi_.x + roi_.width <= frame.cols &&
        roi_.y + roi_.height <= frame.rows) {
        roi_frame = frame(roi_);
    }
    
    // Color segmentation with timing
    Timer seg_timer;
    color_segmenter_->segment(roi_frame, segmented_mask_);
    result.segmentation_time_ms = seg_timer.elapsedMs();
    
    // Find and extract contours with timing
    Timer contour_timer;
    std::vector<std::vector<cv::Point>> contours = 
        contour_detector_->findContours(segmented_mask_);
    std::vector<ContourFeatures> features = 
        contour_detector_->extractFeatures(contours);
    result.contour_time_ms = contour_timer.elapsedMs();
    
    // Apply rules to filter valid dough pieces
    Timer rule_timer;
    std::vector<std::vector<cv::Point>> valid_contours;
    std::vector<cv::Rect> bounding_boxes;
    std::vector<cv::Point2f> centers;
    
    for (size_t i = 0; i < features.size(); i++) {
        if (rule_engine_->validateContour(features[i])) {
            valid_contours.push_back(contours[i]);
            bounding_boxes.push_back(features[i].bounding_box);
            centers.push_back(features[i].center);
        }
    }
    result.rule_time_ms = rule_timer.elapsedMs();
    
    result.contours = valid_contours;
    result.bounding_boxes = bounding_boxes;
    result.centers = centers;
    result.dough_count = static_cast<int>(valid_contours.size());
    result.is_valid = rule_engine_->applyRules(features);
    result.message = rule_engine_->getValidationMessage();
    result.confidence = result.dough_count > 0 ? 0.85 : 0.0;
    result.total_time_ms = total_timer.elapsedMs();
    
    // Track performance stats
    frame_times_.push_back(result.total_time_ms);
    segmentation_times_.push_back(result.segmentation_time_ms);
    contour_times_.push_back(result.contour_time_ms);
    
    // Keep only last 100 frames for stats
    if (frame_times_.size() > 100) {
        frame_times_.erase(frame_times_.begin());
        segmentation_times_.erase(segmentation_times_.begin());
        contour_times_.erase(contour_times_.begin());
    }
    
    return result;
}

void VisionPipeline::renderDetections(cv::Mat& frame, const DetectionResult& result) {
    // Draw ROI rectangle
    if (roi_.width > 0 && roi_.height > 0) {
        cv::rectangle(frame, roi_, cv::Scalar(255, 255, 0), 2);
    }
    
    // Get ROI view for drawing
    cv::Mat roi_view = frame;
    if (roi_.width > 0 && roi_.height > 0) {
        roi_view = frame(roi_);
    }
    
    // Draw contours and bounding boxes
    for (size_t i = 0; i < result.contours.size(); i++) {
        cv::drawContours(roi_view, result.contours, i, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(roi_view, result.bounding_boxes[i], cv::Scalar(255, 0, 0), 2);
        
        // Draw center point
        cv::circle(roi_view, result.centers[i], 5, cv::Scalar(0, 0, 255), -1);
        
        // Draw count label
        std::string label = std::to_string(i + 1);
        cv::putText(roi_view, label, 
                   cv::Point(result.bounding_boxes[i].x, result.bounding_boxes[i].y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
    }
    
    // Draw performance stats
    std::string perf_text = "Frame: " + 
        std::to_string(static_cast<int>(result.total_time_ms)) + "ms | " +
        "Seg: " + std::to_string(static_cast<int>(result.segmentation_time_ms)) + "ms";
    cv::putText(frame, perf_text, cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    
    // Draw count
    std::string count_text = "Count: " + std::to_string(result.dough_count);
    cv::putText(frame, count_text, cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
}

void VisionPipeline::updateColorRange(const cv::Scalar& lower, const cv::Scalar& upper) {
    color_segmenter_->setColorRange(lower, upper);
}

void VisionPipeline::updateROI(const cv::Rect& roi) {
    roi_ = roi;
}

void VisionPipeline::updateDetectionRules(const DetectionRules& rules) {
    rule_engine_->setRules(rules);
}

VisionPipeline::PerformanceStats VisionPipeline::getPerformanceStats() const {
    PerformanceStats stats;
    stats.frame_count = frame_times_.size();
    
    if (frame_times_.empty()) {
        stats.avg_total_ms = 0.0;
        stats.avg_segmentation_ms = 0.0;
        stats.avg_contour_ms = 0.0;
        stats.min_total_ms = 0.0;
        stats.max_total_ms = 0.0;
        return stats;
    }
    
    double sum_total = 0.0, sum_seg = 0.0, sum_contour = 0.0;
    double min_val = frame_times_[0], max_val = frame_times_[0];
    
    for (size_t i = 0; i < frame_times_.size(); i++) {
        sum_total += frame_times_[i];
        sum_seg += segmentation_times_[i];
        sum_contour += contour_times_[i];
        
        if (frame_times_[i] < min_val) min_val = frame_times_[i];
        if (frame_times_[i] > max_val) max_val = frame_times_[i];
    }
    
    stats.avg_total_ms = sum_total / frame_times_.size();
    stats.avg_segmentation_ms = sum_seg / segmentation_times_.size();
    stats.avg_contour_ms = sum_contour / contour_times_.size();
    stats.min_total_ms = min_val;
    stats.max_total_ms = max_val;
    
    return stats;
}

void VisionPipeline::resetPerformanceStats() {
    frame_times_.clear();
    segmentation_times_.clear();
    contour_times_.clear();
}

} // namespace country_style
