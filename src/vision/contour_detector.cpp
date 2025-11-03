#include "contour_detector.h"

namespace country_style {

ContourDetector::ContourDetector() 
    : retrieval_mode_(cv::RETR_EXTERNAL),
      approximation_method_(cv::CHAIN_APPROX_SIMPLE) {}

ContourDetector::~ContourDetector() {}

std::vector<std::vector<cv::Point>> ContourDetector::findContours(const cv::Mat& mask) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    if (mask.empty()) {
        return contours;
    }
    
    cv::findContours(mask.clone(), contours, hierarchy, 
                     retrieval_mode_, approximation_method_);
    
    return contours;
}

std::vector<std::vector<cv::Point>> ContourDetector::filterByArea(
    const std::vector<std::vector<cv::Point>>& contours,
    double min_area, double max_area) {
    
    std::vector<std::vector<cv::Point>> filtered;
    
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area >= min_area && area <= max_area) {
            filtered.push_back(contour);
        }
    }
    
    return filtered;
}

std::vector<ContourFeatures> ContourDetector::extractFeatures(
    const std::vector<std::vector<cv::Point>>& contours) {
    
    std::vector<ContourFeatures> features;
    
    for (const auto& contour : contours) {
        ContourFeatures feat;
        
        // Calculate area
        feat.area = cv::contourArea(contour);
        
        // Skip very small contours
        if (feat.area < 100) {
            continue;
        }
        
        // Calculate perimeter
        feat.perimeter = cv::arcLength(contour, true);
        
        // Calculate circularity: 4*PI*area / perimeter^2
        // Perfect circle = 1.0, less circular shapes < 1.0
        if (feat.perimeter > 0) {
            feat.circularity = (4.0 * CV_PI * feat.area) / (feat.perimeter * feat.perimeter);
        } else {
            feat.circularity = 0.0;
        }
        
        // Get bounding box
        feat.bounding_box = cv::boundingRect(contour);
        
        // Calculate aspect ratio
        if (feat.bounding_box.height > 0) {
            feat.aspect_ratio = static_cast<double>(feat.bounding_box.width) / 
                               static_cast<double>(feat.bounding_box.height);
        } else {
            feat.aspect_ratio = 0.0;
        }
        
        // Calculate center
        cv::Moments m = cv::moments(contour);
        if (m.m00 != 0) {
            feat.center = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
        } else {
            feat.center = cv::Point2f(0, 0);
        }
        
        features.push_back(feat);
    }
    
    return features;
}

cv::Mat ContourDetector::drawContours(const cv::Mat& frame,
                                      const std::vector<std::vector<cv::Point>>& contours,
                                      const cv::Scalar& color) {
    cv::Mat output = frame.clone();
    
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(output, contours, static_cast<int>(i), color, 2);
    }
    
    return output;
}

} // namespace country_style
