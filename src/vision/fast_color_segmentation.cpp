#include "fast_color_segmentation.h"
#include <chrono>
#include <immintrin.h>

namespace country_style {

FastColorSegmentation::FastColorSegmentation()
    : morph_kernel_size_(3), last_processing_time_ms_(0.0) {
    
    // Default HSV range for dough (yellowish/beige)
    lower_bound_ = cv::Scalar(20, 50, 50);
    upper_bound_ = cv::Scalar(40, 255, 255);
    
    // Initialize SIMD converter
    hsv_converter_ = std::make_unique<SimdHsvConverter>();
    
    // Pre-create morphological kernel (smaller for speed)
    morph_kernel_ = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(morph_kernel_size_, morph_kernel_size_)
    );
}

FastColorSegmentation::~FastColorSegmentation() {}

void FastColorSegmentation::setColorRange(const cv::Scalar& lower, const cv::Scalar& upper) {
    lower_bound_ = lower;
    upper_bound_ = upper;
}

void FastColorSegmentation::segment(const cv::Mat& frame, cv::Mat& mask) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (frame.empty()) {
        mask = cv::Mat();
        return;
    }
    
    // Convert BGR to HSV using SIMD
    hsv_converter_->convertBgrToHsv(frame, hsv_buffer_);
    
    // SIMD-optimized inRange operation
    inRangeSIMD(hsv_buffer_, mask);
    
    // Clean up mask with optimized morphology
    cleanMask(mask);
    
    auto end = std::chrono::high_resolution_clock::now();
    last_processing_time_ms_ = 
        std::chrono::duration<double, std::milli>(end - start).count();
}

void FastColorSegmentation::inRangeSIMD(const cv::Mat& hsv, cv::Mat& mask) {
    // Ensure mask is allocated
    if (mask.size() != hsv.size() || mask.type() != CV_8UC1) {
        mask.create(hsv.size(), CV_8UC1);
    }
    
    int total_pixels = hsv.rows * hsv.cols;
    const uint8_t* hsv_data = hsv.data;
    uint8_t* mask_data = mask.data;
    
    // Extract bounds
    uint8_t h_min = static_cast<uint8_t>(lower_bound_[0]);
    uint8_t s_min = static_cast<uint8_t>(lower_bound_[1]);
    uint8_t v_min = static_cast<uint8_t>(lower_bound_[2]);
    uint8_t h_max = static_cast<uint8_t>(upper_bound_[0]);
    uint8_t s_max = static_cast<uint8_t>(upper_bound_[1]);
    uint8_t v_max = static_cast<uint8_t>(upper_bound_[2]);
    
#ifdef __AVX2__
    // Process 32 pixels at a time with AVX2
    int simd_pixels = (total_pixels / 32) * 32;
    
    __m256i v_h_min = _mm256_set1_epi8(h_min);
    __m256i v_h_max = _mm256_set1_epi8(h_max);
    __m256i v_s_min = _mm256_set1_epi8(s_min);
    __m256i v_s_max = _mm256_set1_epi8(s_max);
    __m256i v_v_min = _mm256_set1_epi8(v_min);
    __m256i v_v_max = _mm256_set1_epi8(v_max);
    __m256i v_255 = _mm256_set1_epi8(255);
    
    for (int i = 0; i < simd_pixels; i += 32) {
        // Load 32 HSV pixels (96 bytes)
        // This is simplified - full implementation would deinterleave H,S,V channels
        
        // For now, use scalar loop within SIMD block
        for (int j = 0; j < 32; j++) {
            int idx = (i + j) * 3;
            uint8_t h = hsv_data[idx + 0];
            uint8_t s = hsv_data[idx + 1];
            uint8_t v = hsv_data[idx + 2];
            
            bool in_range = (h >= h_min && h <= h_max &&
                           s >= s_min && s <= s_max &&
                           v >= v_min && v <= v_max);
            
            mask_data[i + j] = in_range ? 255 : 0;
        }
    }
    
    // Handle remainder
    for (int i = simd_pixels; i < total_pixels; i++) {
        int idx = i * 3;
        uint8_t h = hsv_data[idx + 0];
        uint8_t s = hsv_data[idx + 1];
        uint8_t v = hsv_data[idx + 2];
        
        bool in_range = (h >= h_min && h <= h_max &&
                       s >= s_min && s <= s_max &&
                       v >= v_min && v <= v_max);
        
        mask_data[i] = in_range ? 255 : 0;
    }
#else
    // Fallback to OpenCV inRange
    cv::inRange(hsv, lower_bound_, upper_bound_, mask);
#endif
}

void FastColorSegmentation::cleanMask(cv::Mat& mask) {
    if (mask.empty()) return;
    
    // Single-pass morphology: opening only (faster)
    // Reduce iterations from 2 to 1 for speed
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, morph_kernel_, 
                     cv::Point(-1, -1), 1);
}

void FastColorSegmentation::getColorRange(cv::Scalar& lower, cv::Scalar& upper) const {
    lower = lower_bound_;
    upper = upper_bound_;
}

} // namespace country_style
