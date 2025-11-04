#ifndef RECIPE_MANAGER_H
#define RECIPE_MANAGER_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <nlohmann/json_fwd.hpp>
#include "vision_pipeline.h"
#include "rule_engine.h"

namespace country_style {

// Complete recipe definition with all inspection parameters
struct Recipe {
    std::string name;
    std::string description;
    
    // HSV color range
    cv::Scalar hsv_lower;
    cv::Scalar hsv_upper;
    
    // Region of interest
    cv::Rect roi;
    
    // Detection rules
    DetectionRules detection_rules;
    
    // Quality thresholds
    QualityThresholds quality_thresholds;
    
    // Processing parameters
    int morph_kernel_size;
    bool enable_preprocessing;
    
    // Metadata
    std::string created_date;
    std::string modified_date;
    std::string created_by;
    
    Recipe() : morph_kernel_size(5), enable_preprocessing(true) {}
};

// Manages loading, saving, and switching between recipes
class RecipeManager {
public:
    RecipeManager();
    ~RecipeManager();
    
    // Initialize with recipe directory
    bool initialize(const std::string& recipe_dir = "config/recipes");
    
    // Recipe CRUD operations
    bool createRecipe(const Recipe& recipe);
    bool saveRecipe(const Recipe& recipe);
    bool loadRecipe(const std::string& name, Recipe& recipe);
    bool deleteRecipe(const std::string& name);
    bool renameRecipe(const std::string& old_name, const std::string& new_name);
    
    // List available recipes
    std::vector<std::string> getRecipeNames() const;
    bool recipeExists(const std::string& name) const;
    
    // Active recipe management
    bool setActiveRecipe(const std::string& name);
    const Recipe& getActiveRecipe() const { return active_recipe_; }
    std::string getActiveRecipeName() const { return active_recipe_name_; }
    bool hasActiveRecipe() const { return !active_recipe_name_.empty(); }
    
    // Apply recipe to vision pipeline
    void applyRecipeToPipeline(VisionPipeline* pipeline, const Recipe& recipe);
    
    // Import/export
    bool exportRecipe(const std::string& name, const std::string& export_path);
    bool importRecipe(const std::string& import_path, const std::string& new_name = "");
    
    // Create default recipe from current pipeline state
    Recipe createRecipeFromPipeline(const std::string& name, VisionPipeline* pipeline);
    
    // Get recipe directory
    const std::string& getRecipeDirectory() const { return recipe_dir_; }
    
private:
    std::string recipe_dir_;
    Recipe active_recipe_;
    std::string active_recipe_name_;
    
    // Helper methods
    std::string getRecipePath(const std::string& name) const;
    bool ensureRecipeDirectory();
    std::string getCurrentTimestamp() const;
    
    // JSON serialization
    bool recipeToJson(const Recipe& recipe, nlohmann::json& json);
    bool jsonToRecipe(const nlohmann::json& json, Recipe& recipe);
};

} // namespace country_style

#endif // RECIPE_MANAGER_H
