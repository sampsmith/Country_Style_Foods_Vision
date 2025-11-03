#ifndef RULE_ENGINE_H
#define RULE_ENGINE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "contour_detector.h"

namespace country_style {

struct DetectionRules {
    double min_area;
    double max_area;
    double min_circularity;
    double max_circularity;
    double min_aspect_ratio;
    double max_aspect_ratio;
    int expected_count;
    bool enforce_count;
};

class RuleEngine {
public:
    RuleEngine();
    ~RuleEngine();

    // Load rules from configuration
    bool loadRules(const std::string& config_path);
    
    // Set rules programmatically
    void setRules(const DetectionRules& rules);
    
    // Apply rules to contour features
    bool applyRules(const std::vector<ContourFeatures>& features);
    
    // Validate individual contour
    bool validateContour(const ContourFeatures& feature);
    
    // Get rule validation results
    std::string getValidationMessage() const;
    
    // Get current rules
    DetectionRules getRules() const;

private:
    DetectionRules rules_;
    std::string validation_message_;
    
    bool validateArea(double area);
    bool validateCircularity(double circularity);
    bool validateAspectRatio(double ratio);
};

} // namespace country_style

#endif // RULE_ENGINE_H
