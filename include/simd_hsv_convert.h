#ifndef SIMD_HSV_CONVERT_H
#define SIMD_HSV_CONVERT_H

#include <opencv2/opencv.hpp>
#include <cstdint>

namespace country_style {

// SIMD-optimized BGR to HSV conversion using AVX2
class SimdHsvConverter {
public:
    SimdHsvConverter();
    ~SimdHsvConverter();
    
    // Convert BGR to HSV with SIMD acceleration
    void convertBgrToHsv(const cv::Mat& bgr, cv::Mat& hsv);
    
    // Build lookup tables for faster conversion (call once at startup)
    void buildLookupTables();
    
    // Check if AVX2 is available
    static bool hasAvx2Support();
    
private:
    // LUT-based conversion (fallback or hybrid approach)
    void convertBgrToHsvLut(const cv::Mat& bgr, cv::Mat& hsv);
    
    // Pure SIMD implementation
    void convertBgrToHsvAvx2(const uint8_t* bgr, uint8_t* hsv, int pixels);
    
    // Lookup tables for fast conversion
    uint8_t* hue_lut_;
    uint8_t* sat_lut_;
    bool use_avx2_;
};

} // namespace country_style

#endif // SIMD_HSV_CONVERT_H
