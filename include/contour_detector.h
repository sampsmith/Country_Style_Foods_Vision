#ifndef CONTOUR_DETECTOR_H
#define CONTOUR_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace country_style {

struct ContourFeatures {
    double area;
    double perimeter;
    double circularity;
    double aspect_ratio;
    cv::Rect bounding_box;
    cv::Point2f center;
};

class ContourDetector {
public:
    ContourDetector();
    ~ContourDetector();

    // Find contours in binary mask
    std::vector<std::vector<cv::Point>> findContours(const cv::Mat& mask);
    
    // Filter contours based on area constraints
    std::vector<std::vector<cv::Point>> filterByArea(
        const std::vector<std::vector<cv::Point>>& contours,
        double min_area, double max_area);
    
    // Extract features from contours
    std::vector<ContourFeatures> extractFeatures(
        const std::vector<std::vector<cv::Point>>& contours);
    
    // Draw contours on image
    cv::Mat drawContours(const cv::Mat& frame,
                         const std::vector<std::vector<cv::Point>>& contours,
                         const cv::Scalar& color);

private:
    int retrieval_mode_;
    int approximation_method_;
};

} // namespace country_style

#endif // CONTOUR_DETECTOR_H
