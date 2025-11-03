#include "rule_engine.h"
#include <sstream>

namespace country_style {

RuleEngine::RuleEngine() {
    // Set default rules
    rules_.min_area = 500.0;
    rules_.max_area = 50000.0;
    rules_.min_circularity = 0.3;
    rules_.max_circularity = 1.0;
    rules_.min_aspect_ratio = 0.3;
    rules_.max_aspect_ratio = 3.0;
    rules_.expected_count = 0;
    rules_.enforce_count = false;
}

RuleEngine::~RuleEngine() {}

bool RuleEngine::loadRules(const std::string& config_path) {
    // TODO: Implement JSON loading
    return false;
}

void RuleEngine::setRules(const DetectionRules& rules) {
    rules_ = rules;
}

bool RuleEngine::applyRules(const std::vector<ContourFeatures>& features) {
    validation_message_.clear();
    
    int valid_count = 0;
    for (const auto& feature : features) {
        if (validateContour(feature)) {
            valid_count++;
        }
    }
    
    // Check if count matches expected
    if (rules_.enforce_count && valid_count != rules_.expected_count) {
        std::ostringstream oss;
        oss << "Expected " << rules_.expected_count << " dough pieces, found " << valid_count;
        validation_message_ = oss.str();
        return false;
    }
    
    if (valid_count == 0) {
        validation_message_ = "No valid dough pieces detected";
        return false;
    }
    
    validation_message_ = "Detection OK";
    return true;
}

bool RuleEngine::validateContour(const ContourFeatures& feature) {
    // Validate area
    if (!validateArea(feature.area)) {
        return false;
    }
    
    // Validate circularity
    if (!validateCircularity(feature.circularity)) {
        return false;
    }
    
    // Validate aspect ratio
    if (!validateAspectRatio(feature.aspect_ratio)) {
        return false;
    }
    
    return true;
}

std::string RuleEngine::getValidationMessage() const {
    return validation_message_;
}

DetectionRules RuleEngine::getRules() const {
    return rules_;
}

bool RuleEngine::validateArea(double area) {
    return area >= rules_.min_area && area <= rules_.max_area;
}

bool RuleEngine::validateCircularity(double circularity) {
    return circularity >= rules_.min_circularity && circularity <= rules_.max_circularity;
}

bool RuleEngine::validateAspectRatio(double ratio) {
    return ratio >= rules_.min_aspect_ratio && ratio <= rules_.max_aspect_ratio;
}

} // namespace country_style
