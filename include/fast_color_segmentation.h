#ifndef FAST_COLOR_SEGMENTATION_H
#define FAST_COLOR_SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <memory>
#include "simd_hsv_convert.h"

namespace country_style {

class FastColorSegmentation {
public:
    FastColorSegmentation();
    ~FastColorSegmentation();
    
    // Set HSV color range for dough detection
    void setColorRange(const cv::Scalar& lower, const cv::Scalar& upper);
    
    // High-performance segmentation (target: <5ms for 640x480)
    // Returns binary mask in pre-allocated buffer
    void segment(const cv::Mat& frame, cv::Mat& mask);
    
    // Apply morphological operations (optimized single-pass)
    void cleanMask(cv::Mat& mask);
    
    // Get current color range
    void getColorRange(cv::Scalar& lower, cv::Scalar& upper) const;
    
    // Performance statistics
    double getLastProcessingTimeMs() const { return last_processing_time_ms_; }
    
private:
    // SIMD-accelerated HSV converter
    std::unique_ptr<SimdHsvConverter> hsv_converter_;
    
    // Color range bounds
    cv::Scalar lower_bound_;
    cv::Scalar upper_bound_;
    
    // Pre-allocated buffers to avoid memory allocation overhead
    cv::Mat hsv_buffer_;
    cv::Mat morph_kernel_;
    
    // Morphology settings
    int morph_kernel_size_;
    
    // Performance tracking
    double last_processing_time_ms_;
    
    // SIMD-optimized inRange operation
    void inRangeSIMD(const cv::Mat& hsv, cv::Mat& mask);
};

} // namespace country_style

#endif // FAST_COLOR_SEGMENTATION_H
