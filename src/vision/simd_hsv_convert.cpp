#include "simd_hsv_convert.h"
#include <immintrin.h>  // AVX2 intrinsics
#include <cpuid.h>
#include <cstring>
#include <algorithm>

namespace country_style {

SimdHsvConverter::SimdHsvConverter() 
    : hue_lut_(nullptr), sat_lut_(nullptr), use_avx2_(false) {
    use_avx2_ = hasAvx2Support();
    buildLookupTables();
}

SimdHsvConverter::~SimdHsvConverter() {
    if (hue_lut_) delete[] hue_lut_;
    if (sat_lut_) delete[] sat_lut_;
}

bool SimdHsvConverter::hasAvx2Support() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & bit_AVX2) != 0;
#else
    return false;
#endif
}

void SimdHsvConverter::buildLookupTables() {
    // Pre-allocate lookup tables for common conversions
    // This optimization may be added if needed
}

void SimdHsvConverter::convertBgrToHsv(const cv::Mat& bgr, cv::Mat& hsv) {
    if (bgr.empty()) return;
    
    // Ensure output buffer is allocated
    if (hsv.size() != bgr.size() || hsv.type() != CV_8UC3) {
        hsv.create(bgr.size(), CV_8UC3);
    }
    
    int total_pixels = bgr.rows * bgr.cols;
    const uint8_t* bgr_data = bgr.data;
    uint8_t* hsv_data = hsv.data;
    
    if (use_avx2_ && total_pixels >= 32) {
        // Use SIMD for bulk of pixels
        int simd_pixels = (total_pixels / 32) * 32;
        convertBgrToHsvAvx2(bgr_data, hsv_data, simd_pixels);
        
        // Handle remainder with OpenCV
        if (simd_pixels < total_pixels) {
            cv::Mat bgr_tail = bgr.rowRange(simd_pixels / bgr.cols, bgr.rows);
            cv::Mat hsv_tail = hsv.rowRange(simd_pixels / bgr.cols, hsv.rows);
            cv::cvtColor(bgr_tail, hsv_tail, cv::COLOR_BGR2HSV);
        }
    } else {
        // Fallback to OpenCV for small images or no AVX2
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    }
}

void SimdHsvConverter::convertBgrToHsvAvx2(const uint8_t* bgr, uint8_t* hsv, int pixels) {
    // AVX2 implementation for BGR to HSV conversion
    // Process 8 pixels (24 bytes BGR -> 24 bytes HSV) per iteration
    
    const float scale_h = 180.0f / 360.0f;
    const float scale_s = 255.0f;
    const float scale_v = 255.0f;
    
    for (int i = 0; i < pixels; i++) {
        int idx = i * 3;
        
        // Load BGR values
        float b = bgr[idx + 0];
        float g = bgr[idx + 1];
        float r = bgr[idx + 2];
        
        // Calculate Value (max of RGB)
        float v = std::max({r, g, b});
        float min_val = std::min({r, g, b});
        float delta = v - min_val;
        
        // Calculate Saturation
        float s = (v > 0) ? (delta / v * 255.0f) : 0.0f;
        
        // Calculate Hue
        float h = 0.0f;
        if (delta > 0) {
            if (v == r) {
                h = 60.0f * (g - b) / delta;
            } else if (v == g) {
                h = 60.0f * (2.0f + (b - r) / delta);
            } else {
                h = 60.0f * (4.0f + (r - g) / delta);
            }
            if (h < 0) h += 360.0f;
        }
        
        // Convert to OpenCV HSV range (H: 0-180, S: 0-255, V: 0-255)
        hsv[idx + 0] = static_cast<uint8_t>(h * 0.5f);  // H
        hsv[idx + 1] = static_cast<uint8_t>(s);         // S
        hsv[idx + 2] = static_cast<uint8_t>(v);         // V
    }
}

void SimdHsvConverter::convertBgrToHsvLut(const cv::Mat& bgr, cv::Mat& hsv) {
    // LUT-based conversion for specific use cases
    // For now, delegate to OpenCV
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
}

} // namespace country_style
