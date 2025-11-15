#include "vision_pipeline.h"
#include "config_manager.h"
#include <iostream>

namespace country_style {

VisionPipeline::VisionPipeline()
    : is_initialized_(false) {
    color_segmenter_ = std::make_unique<FastColorSegmentation>();
    contour_detector_ = std::make_unique<ContourDetector>();
    rule_engine_ = std::make_unique<RuleEngine>();
    
    // Initialize quality thresholds to disabled
    quality_thresholds_.enable_area_check = false;
    quality_thresholds_.enable_width_check = false;
    quality_thresholds_.enable_height_check = false;
    quality_thresholds_.enable_aspect_ratio_check = false;
    quality_thresholds_.enable_circularity_check = false;
    quality_thresholds_.enable_count_check = false;
    
    quality_thresholds_.expected_count = 0;
    quality_thresholds_.enforce_exact_count = false;
    quality_thresholds_.min_count = 0;
    quality_thresholds_.max_count = 0;
    quality_thresholds_.min_area = 0;
    quality_thresholds_.max_area = 0;
    quality_thresholds_.min_width = 0;
    quality_thresholds_.max_width = 0;
    quality_thresholds_.min_height = 0;
    quality_thresholds_.max_height = 0;
    quality_thresholds_.min_aspect_ratio = 0;
    quality_thresholds_.max_aspect_ratio = 0;
    quality_thresholds_.min_circularity = 0;
    quality_thresholds_.max_circularity = 0;
    quality_thresholds_.fail_on_undersized = false;
    quality_thresholds_.fail_on_oversized = false;
    quality_thresholds_.fail_on_count_mismatch = false;
    quality_thresholds_.fail_on_shape_defects = false;
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
    
    // Always segment on full frame - no ROI cropping
    Timer seg_timer;
    color_segmenter_->segment(frame, segmented_mask_);
    
    // If ROI is set, zero out segmentation outside ROI so no detections appear there
    if (roi_.width > 0 && roi_.height > 0) {
        cv::Rect safe_roi = roi_ & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safe_roi.width > 0 && safe_roi.height > 0) {
            cv::Mat masked = cv::Mat::zeros(segmented_mask_.size(), segmented_mask_.type());
            segmented_mask_(safe_roi).copyTo(masked(safe_roi));
            segmented_mask_ = masked;
        } else {
            // ROI is out of bounds; clear entire mask
            segmented_mask_ = cv::Mat::zeros(segmented_mask_.size(), segmented_mask_.type());
        }
    }
    result.segmentation_time_ms = seg_timer.elapsedMs();
    
    // Find and extract contours with timing
    Timer contour_timer;
    std::vector<std::vector<cv::Point>> contours = 
        contour_detector_->findContours(segmented_mask_);
    std::vector<ContourFeatures> features = 
        contour_detector_->extractFeatures(contours);
    result.contour_time_ms = contour_timer.elapsedMs();
    
    // Apply rules to filter valid dough pieces and calculate measurements
    Timer rule_timer;
    std::vector<std::vector<cv::Point>> valid_contours;
    std::vector<cv::Rect> bounding_boxes;
    std::vector<cv::Point2f> centers;
    std::vector<DetectionMeasurement> measurements;
    
    // Check if ROI filtering is enabled
    bool use_roi_filter = (roi_.width > 0 && roi_.height > 0);
    
    int detection_id = 1;
    for (size_t i = 0; i < features.size(); i++) {
        if (rule_engine_->validateContour(features[i])) {
            // If ROI is set, only keep detections whose center is inside ROI
            if (use_roi_filter) {
                if (!roi_.contains(features[i].center)) {
                    continue;  // Skip detections outside ROI
                }
            }
            
            // Create detailed measurement for this detection
            DetectionMeasurement meas;
            meas.id = detection_id++;
            meas.area_pixels = features[i].area;
            meas.width_pixels = features[i].bounding_box.width;
            meas.height_pixels = features[i].bounding_box.height;
            meas.aspect_ratio = features[i].aspect_ratio;
            meas.circularity = features[i].circularity;
            meas.center = features[i].center;
            meas.bbox = features[i].bounding_box;
            meas.meets_specs = true;  // Will be updated in fault detection
            
            // Individual threshold checks - respect enable flags
            bool width_ok = true;
            bool length_ok = true;
            bool area_ok = true;
            bool aspect_ratio_ok = true;
            bool circularity_ok = true;
            
            // Area check (if enabled)
            if (quality_thresholds_.enable_area_check) {
                if (quality_thresholds_.min_area > 0 && meas.area_pixels < quality_thresholds_.min_area) {
                    meas.meets_specs = false;
                    area_ok = false;
                    if (!meas.fault_reason.empty()) meas.fault_reason += ", ";
                    meas.fault_reason += "Area too small (" + std::to_string((int)meas.area_pixels) + "px²)";
                }
                if (quality_thresholds_.max_area > 0 && meas.area_pixels > quality_thresholds_.max_area) {
                    meas.meets_specs = false;
                    area_ok = false;
                    if (!meas.fault_reason.empty()) meas.fault_reason += ", ";
                    meas.fault_reason += "Area too large (" + std::to_string((int)meas.area_pixels) + "px²)";
                }
            }
            
            // Width check (if enabled)
            if (quality_thresholds_.enable_width_check) {
                if (quality_thresholds_.min_width > 0 && meas.width_pixels < quality_thresholds_.min_width) {
                    meas.meets_specs = false;
                    width_ok = false;
                    if (!meas.fault_reason.empty()) meas.fault_reason += ", ";
                    meas.fault_reason += "Width too small (" + std::to_string((int)meas.width_pixels) + "px)";
                }
                if (quality_thresholds_.max_width > 0 && meas.width_pixels > quality_thresholds_.max_width) {
                    meas.meets_specs = false;
                    width_ok = false;
                    if (!meas.fault_reason.empty()) meas.fault_reason += ", ";
                    meas.fault_reason += "Width too large (" + std::to_string((int)meas.width_pixels) + "px)";
                }
            }
            
            // Height/Length check (if enabled)
            if (quality_thresholds_.enable_height_check) {
                if (quality_thresholds_.min_height > 0 && meas.height_pixels < quality_thresholds_.min_height) {
                    meas.meets_specs = false;
                    length_ok = false;
                    if (!meas.fault_reason.empty()) meas.fault_reason += ", ";
                    meas.fault_reason += "Length too small (" + std::to_string((int)meas.height_pixels) + "px)";
                }
                if (quality_thresholds_.max_height > 0 && meas.height_pixels > quality_thresholds_.max_height) {
                    meas.meets_specs = false;
                    length_ok = false;
                    if (!meas.fault_reason.empty()) meas.fault_reason += ", ";
                    meas.fault_reason += "Length too large (" + std::to_string((int)meas.height_pixels) + "px)";
                }
            }
            
            // Aspect ratio check (if enabled)
            if (quality_thresholds_.enable_aspect_ratio_check) {
                if (quality_thresholds_.min_aspect_ratio > 0 && meas.aspect_ratio < quality_thresholds_.min_aspect_ratio) {
                    meas.meets_specs = false;
                    aspect_ratio_ok = false;
                    if (!meas.fault_reason.empty()) meas.fault_reason += ", ";
                    meas.fault_reason += "Aspect ratio too low (" + std::to_string(meas.aspect_ratio) + ")";
                }
                if (quality_thresholds_.max_aspect_ratio > 0 && meas.aspect_ratio > quality_thresholds_.max_aspect_ratio) {
                    meas.meets_specs = false;
                    aspect_ratio_ok = false;
                    if (!meas.fault_reason.empty()) meas.fault_reason += ", ";
                    meas.fault_reason += "Aspect ratio too high (" + std::to_string(meas.aspect_ratio) + ")";
                }
            }
            
            // Circularity check (if enabled)
            if (quality_thresholds_.enable_circularity_check) {
                if (quality_thresholds_.min_circularity > 0 && meas.circularity < quality_thresholds_.min_circularity) {
                    meas.meets_specs = false;
                    circularity_ok = false;
                    if (!meas.fault_reason.empty()) meas.fault_reason += ", ";
                    meas.fault_reason += "Circularity too low (" + std::to_string(meas.circularity) + ")";
                }
                if (quality_thresholds_.max_circularity > 0 && meas.circularity > quality_thresholds_.max_circularity) {
                    meas.meets_specs = false;
                    circularity_ok = false;
                    if (!meas.fault_reason.empty()) meas.fault_reason += ", ";
                    meas.fault_reason += "Circularity too high (" + std::to_string(meas.circularity) + ")";
                }
            }
            
            valid_contours.push_back(contours[i]);
            bounding_boxes.push_back(features[i].bounding_box);
            centers.push_back(features[i].center);
            measurements.push_back(meas);
        }
    }
    result.rule_time_ms = rule_timer.elapsedMs();
    
    result.contours = valid_contours;
    result.bounding_boxes = bounding_boxes;
    result.centers = centers;
    result.measurements = measurements;
    result.dough_count = static_cast<int>(valid_contours.size());
    
    // Initialize fault flags
    result.fault_count_low = false;
    result.fault_count_high = false;
    result.fault_undersized = false;
    result.fault_oversized = false;
    result.fault_shape_defect = false;
    result.fault_messages.clear();
    
    // Count validation (if enabled)
    if (quality_thresholds_.enable_count_check) {
        if (quality_thresholds_.enforce_exact_count && result.dough_count != quality_thresholds_.expected_count) {
            result.fault_count_low = result.dough_count < quality_thresholds_.expected_count;
            result.fault_count_high = result.dough_count > quality_thresholds_.expected_count;
            if (result.fault_count_low) {
                result.fault_messages.push_back("COUNT TOO LOW: " + std::to_string(result.dough_count) + 
                                               " (expected " + std::to_string(quality_thresholds_.expected_count) + ")");
            }
            if (result.fault_count_high) {
                result.fault_messages.push_back("COUNT TOO HIGH: " + std::to_string(result.dough_count) + 
                                               " (expected " + std::to_string(quality_thresholds_.expected_count) + ")");
            }
        } else if (quality_thresholds_.min_count > 0 && result.dough_count < quality_thresholds_.min_count) {
            result.fault_count_low = true;
            result.fault_messages.push_back("COUNT TOO LOW: " + std::to_string(result.dough_count) + 
                                           " (min " + std::to_string(quality_thresholds_.min_count) + ")");
        } else if (quality_thresholds_.max_count > 0 && result.dough_count > quality_thresholds_.max_count) {
            result.fault_count_high = true;
            result.fault_messages.push_back("COUNT TOO HIGH: " + std::to_string(result.dough_count) + 
                                           " (max " + std::to_string(quality_thresholds_.max_count) + ")");
        }
    }
    
    // Individual detection faults
    for (const auto& meas : measurements) {
        if (!meas.meets_specs) {
            if (meas.fault_reason.find("Undersized") != std::string::npos) {
                result.fault_undersized = true;
            }
            if (meas.fault_reason.find("Oversized") != std::string::npos) {
                result.fault_oversized = true;
            }
            if (meas.fault_reason.find("Shape") != std::string::npos) {
                result.fault_shape_defect = true;
            }
            result.fault_messages.push_back("Detection #" + std::to_string(meas.id) + ": " + meas.fault_reason);
        }
    }
    
    // Overall validation based on fault triggers
    result.is_valid = true;
    if (quality_thresholds_.fail_on_count_mismatch && (result.fault_count_low || result.fault_count_high)) {
        result.is_valid = false;
    }
    if (quality_thresholds_.fail_on_undersized && result.fault_undersized) {
        result.is_valid = false;
    }
    if (quality_thresholds_.fail_on_oversized && result.fault_oversized) {
        result.is_valid = false;
    }
    if (quality_thresholds_.fail_on_shape_defects && result.fault_shape_defect) {
        result.is_valid = false;
    }
    
    // Set message
    if (result.is_valid) {
        result.message = "PASS";
    } else {
        result.message = "FAIL: " + std::to_string(result.fault_messages.size()) + " fault(s)";
    }
    
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
    
    bool roi_enabled = (roi_.width > 0 && roi_.height > 0);
    
    // Draw contours and bounding boxes with ROI-aware clipping
    for (size_t i = 0; i < result.contours.size(); i++) {
        const cv::Rect& bbox = result.bounding_boxes[i];
        
        if (roi_enabled) {
            // Clip bbox to ROI
            cv::Rect clipped = bbox & roi_;
            if (clipped.area() > 0) {
                // Draw clipped bbox
                cv::rectangle(frame, clipped, cv::Scalar(255, 0, 0), 2);
            }
            
            // Draw contour only inside ROI using overlay + ROI copy
            cv::Mat overlay = cv::Mat::zeros(frame.size(), frame.type());
            cv::drawContours(overlay, result.contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
            cv::Mat frame_roi = frame(roi_);
            cv::Mat overlay_roi = overlay(roi_);
            cv::addWeighted(frame_roi, 1.0, overlay_roi, 1.0, 0.0, frame_roi);
            
            // Draw center point only if inside ROI
            if (roi_.contains(result.centers[i])) {
                cv::circle(frame, result.centers[i], 5, cv::Scalar(0, 0, 255), -1);
            }
            
            // Draw label at clipped top-left if visible
            if (clipped.area() > 0) {
                std::string label = std::to_string(i + 1);
                cv::Point label_pos(clipped.x, std::max(0, clipped.y - 5));
                cv::putText(frame, label, label_pos,
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
            }
        } else {
            // Normal drawing when no ROI
            cv::drawContours(frame, result.contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
            cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2);
            cv::circle(frame, result.centers[i], 5, cv::Scalar(0, 0, 255), -1);
            std::string label = std::to_string(i + 1);
            cv::putText(frame, label, 
                       cv::Point(bbox.x, bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
        }
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

void VisionPipeline::updateQualityThresholds(const QualityThresholds& thresholds) {
    quality_thresholds_ = thresholds;
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
