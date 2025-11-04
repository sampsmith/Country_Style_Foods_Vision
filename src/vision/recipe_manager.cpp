#include "recipe_manager.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace country_style {

RecipeManager::RecipeManager() {}

RecipeManager::~RecipeManager() {}

bool RecipeManager::initialize(const std::string& recipe_dir) {
    recipe_dir_ = recipe_dir;
    return ensureRecipeDirectory();
}

bool RecipeManager::ensureRecipeDirectory() {
    try {
        if (!fs::exists(recipe_dir_)) {
            fs::create_directories(recipe_dir_);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create recipe directory: " << e.what() << std::endl;
        return false;
    }
}

std::string RecipeManager::getRecipePath(const std::string& name) const {
    return recipe_dir_ + "/" + name + ".json";
}

std::string RecipeManager::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

bool RecipeManager::recipeToJson(const Recipe& recipe, json& j) {
    try {
        j["name"] = recipe.name;
        j["description"] = recipe.description;
        
        // HSV range
        j["hsv_lower"] = {recipe.hsv_lower[0], recipe.hsv_lower[1], recipe.hsv_lower[2]};
        j["hsv_upper"] = {recipe.hsv_upper[0], recipe.hsv_upper[1], recipe.hsv_upper[2]};
        
        // ROI
        j["roi"]["x"] = recipe.roi.x;
        j["roi"]["y"] = recipe.roi.y;
        j["roi"]["width"] = recipe.roi.width;
        j["roi"]["height"] = recipe.roi.height;
        
        // Detection rules
        j["detection_rules"]["min_area"] = recipe.detection_rules.min_area;
        j["detection_rules"]["max_area"] = recipe.detection_rules.max_area;
        j["detection_rules"]["min_circularity"] = recipe.detection_rules.min_circularity;
        j["detection_rules"]["max_circularity"] = recipe.detection_rules.max_circularity;
        j["detection_rules"]["min_aspect_ratio"] = recipe.detection_rules.min_aspect_ratio;
        j["detection_rules"]["max_aspect_ratio"] = recipe.detection_rules.max_aspect_ratio;
        
        // Quality thresholds
        j["quality"]["expected_count"] = recipe.quality_thresholds.expected_count;
        j["quality"]["enforce_exact_count"] = recipe.quality_thresholds.enforce_exact_count;
        j["quality"]["min_count"] = recipe.quality_thresholds.min_count;
        j["quality"]["max_count"] = recipe.quality_thresholds.max_count;
        j["quality"]["min_area"] = recipe.quality_thresholds.min_area;
        j["quality"]["max_area"] = recipe.quality_thresholds.max_area;
        j["quality"]["min_width"] = recipe.quality_thresholds.min_width;
        j["quality"]["max_width"] = recipe.quality_thresholds.max_width;
        j["quality"]["min_height"] = recipe.quality_thresholds.min_height;
        j["quality"]["max_height"] = recipe.quality_thresholds.max_height;
        j["quality"]["min_aspect_ratio"] = recipe.quality_thresholds.min_aspect_ratio;
        j["quality"]["max_aspect_ratio"] = recipe.quality_thresholds.max_aspect_ratio;
        j["quality"]["min_circularity"] = recipe.quality_thresholds.min_circularity;
        j["quality"]["max_circularity"] = recipe.quality_thresholds.max_circularity;
        j["quality"]["fail_on_undersized"] = recipe.quality_thresholds.fail_on_undersized;
        j["quality"]["fail_on_oversized"] = recipe.quality_thresholds.fail_on_oversized;
        j["quality"]["fail_on_count_mismatch"] = recipe.quality_thresholds.fail_on_count_mismatch;
        j["quality"]["fail_on_shape_defects"] = recipe.quality_thresholds.fail_on_shape_defects;
        
        // Processing parameters
        j["processing"]["morph_kernel_size"] = recipe.morph_kernel_size;
        j["processing"]["enable_preprocessing"] = recipe.enable_preprocessing;
        
        // Metadata
        j["metadata"]["created_date"] = recipe.created_date;
        j["metadata"]["modified_date"] = recipe.modified_date;
        j["metadata"]["created_by"] = recipe.created_by;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to convert recipe to JSON: " << e.what() << std::endl;
        return false;
    }
}

bool RecipeManager::jsonToRecipe(const json& j, Recipe& recipe) {
    try {
        recipe.name = j.value("name", "");
        recipe.description = j.value("description", "");
        
        // HSV range
        if (j.contains("hsv_lower") && j["hsv_lower"].is_array() && j["hsv_lower"].size() >= 3) {
            recipe.hsv_lower = cv::Scalar(j["hsv_lower"][0], j["hsv_lower"][1], j["hsv_lower"][2]);
        }
        if (j.contains("hsv_upper") && j["hsv_upper"].is_array() && j["hsv_upper"].size() >= 3) {
            recipe.hsv_upper = cv::Scalar(j["hsv_upper"][0], j["hsv_upper"][1], j["hsv_upper"][2]);
        }
        
        // ROI
        if (j.contains("roi")) {
            recipe.roi.x = j["roi"].value("x", 0);
            recipe.roi.y = j["roi"].value("y", 0);
            recipe.roi.width = j["roi"].value("width", 640);
            recipe.roi.height = j["roi"].value("height", 480);
        }
        
        // Detection rules
        if (j.contains("detection_rules")) {
            auto& dr = j["detection_rules"];
            recipe.detection_rules.min_area = dr.value("min_area", 500.0);
            recipe.detection_rules.max_area = dr.value("max_area", 50000.0);
            recipe.detection_rules.min_circularity = dr.value("min_circularity", 0.3);
            recipe.detection_rules.max_circularity = dr.value("max_circularity", 1.0);
            recipe.detection_rules.min_aspect_ratio = dr.value("min_aspect_ratio", 0.0);
            recipe.detection_rules.max_aspect_ratio = dr.value("max_aspect_ratio", 10.0);
        }
        
        // Quality thresholds
        if (j.contains("quality")) {
            auto& q = j["quality"];
            recipe.quality_thresholds.expected_count = q.value("expected_count", 0);
            recipe.quality_thresholds.enforce_exact_count = q.value("enforce_exact_count", false);
            recipe.quality_thresholds.min_count = q.value("min_count", 0);
            recipe.quality_thresholds.max_count = q.value("max_count", 100);
            recipe.quality_thresholds.min_area = q.value("min_area", 0.0);
            recipe.quality_thresholds.max_area = q.value("max_area", 100000.0);
            recipe.quality_thresholds.min_width = q.value("min_width", 0.0);
            recipe.quality_thresholds.max_width = q.value("max_width", 1000.0);
            recipe.quality_thresholds.min_height = q.value("min_height", 0.0);
            recipe.quality_thresholds.max_height = q.value("max_height", 1000.0);
            recipe.quality_thresholds.min_aspect_ratio = q.value("min_aspect_ratio", 0.0);
            recipe.quality_thresholds.max_aspect_ratio = q.value("max_aspect_ratio", 10.0);
            recipe.quality_thresholds.min_circularity = q.value("min_circularity", 0.0);
            recipe.quality_thresholds.max_circularity = q.value("max_circularity", 1.0);
            recipe.quality_thresholds.fail_on_undersized = q.value("fail_on_undersized", true);
            recipe.quality_thresholds.fail_on_oversized = q.value("fail_on_oversized", true);
            recipe.quality_thresholds.fail_on_count_mismatch = q.value("fail_on_count_mismatch", true);
            recipe.quality_thresholds.fail_on_shape_defects = q.value("fail_on_shape_defects", true);
        }
        
        // Processing parameters
        if (j.contains("processing")) {
            recipe.morph_kernel_size = j["processing"].value("morph_kernel_size", 5);
            recipe.enable_preprocessing = j["processing"].value("enable_preprocessing", true);
        }
        
        // Metadata
        if (j.contains("metadata")) {
            recipe.created_date = j["metadata"].value("created_date", "");
            recipe.modified_date = j["metadata"].value("modified_date", "");
            recipe.created_by = j["metadata"].value("created_by", "");
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse recipe JSON: " << e.what() << std::endl;
        return false;
    }
}

bool RecipeManager::createRecipe(const Recipe& recipe) {
    if (recipe.name.empty()) {
        std::cerr << "Recipe name cannot be empty" << std::endl;
        return false;
    }
    
    if (recipeExists(recipe.name)) {
        std::cerr << "Recipe '" << recipe.name << "' already exists" << std::endl;
        return false;
    }
    
    return saveRecipe(recipe);
}

bool RecipeManager::saveRecipe(const Recipe& recipe) {
    if (recipe.name.empty()) {
        std::cerr << "Recipe name cannot be empty" << std::endl;
        return false;
    }
    
    ensureRecipeDirectory();
    
    json j;
    if (!recipeToJson(recipe, j)) {
        return false;
    }
    
    std::string path = getRecipePath(recipe.name);
    try {
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << path << std::endl;
            return false;
        }
        file << j.dump(2);
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save recipe: " << e.what() << std::endl;
        return false;
    }
}

bool RecipeManager::loadRecipe(const std::string& name, Recipe& recipe) {
    std::string path = getRecipePath(name);
    
    try {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::cerr << "Failed to open recipe file: " << path << std::endl;
            return false;
        }
        
        json j;
        file >> j;
        file.close();
        
        return jsonToRecipe(j, recipe);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load recipe: " << e.what() << std::endl;
        return false;
    }
}

bool RecipeManager::deleteRecipe(const std::string& name) {
    std::string path = getRecipePath(name);
    
    try {
        if (fs::exists(path)) {
            fs::remove(path);
            
            // Clear active recipe if it was deleted
            if (active_recipe_name_ == name) {
                active_recipe_name_.clear();
            }
            
            return true;
        }
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Failed to delete recipe: " << e.what() << std::endl;
        return false;
    }
}

bool RecipeManager::renameRecipe(const std::string& old_name, const std::string& new_name) {
    if (!recipeExists(old_name)) {
        std::cerr << "Recipe '" << old_name << "' does not exist" << std::endl;
        return false;
    }
    
    if (recipeExists(new_name)) {
        std::cerr << "Recipe '" << new_name << "' already exists" << std::endl;
        return false;
    }
    
    Recipe recipe;
    if (!loadRecipe(old_name, recipe)) {
        return false;
    }
    
    recipe.name = new_name;
    recipe.modified_date = getCurrentTimestamp();
    
    if (!saveRecipe(recipe)) {
        return false;
    }
    
    deleteRecipe(old_name);
    
    if (active_recipe_name_ == old_name) {
        active_recipe_name_ = new_name;
        active_recipe_ = recipe;
    }
    
    return true;
}

std::vector<std::string> RecipeManager::getRecipeNames() const {
    std::vector<std::string> names;
    
    try {
        if (!fs::exists(recipe_dir_)) {
            return names;
        }
        
        for (const auto& entry : fs::directory_iterator(recipe_dir_)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                names.push_back(entry.path().stem().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to list recipes: " << e.what() << std::endl;
    }
    
    return names;
}

bool RecipeManager::recipeExists(const std::string& name) const {
    return fs::exists(getRecipePath(name));
}

bool RecipeManager::setActiveRecipe(const std::string& name) {
    Recipe recipe;
    if (!loadRecipe(name, recipe)) {
        return false;
    }
    
    active_recipe_ = recipe;
    active_recipe_name_ = name;
    return true;
}

void RecipeManager::applyRecipeToPipeline(VisionPipeline* pipeline, const Recipe& recipe) {
    if (!pipeline) return;
    
    pipeline->updateColorRange(recipe.hsv_lower, recipe.hsv_upper);
    pipeline->updateROI(recipe.roi);
    pipeline->updateDetectionRules(recipe.detection_rules);
    pipeline->updateQualityThresholds(recipe.quality_thresholds);
}

bool RecipeManager::exportRecipe(const std::string& name, const std::string& export_path) {
    Recipe recipe;
    if (!loadRecipe(name, recipe)) {
        return false;
    }
    
    json j;
    if (!recipeToJson(recipe, j)) {
        return false;
    }
    
    try {
        std::ofstream file(export_path);
        if (!file.is_open()) {
            return false;
        }
        file << j.dump(2);
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to export recipe: " << e.what() << std::endl;
        return false;
    }
}

bool RecipeManager::importRecipe(const std::string& import_path, const std::string& new_name) {
    try {
        std::ifstream file(import_path);
        if (!file.is_open()) {
            return false;
        }
        
        json j;
        file >> j;
        file.close();
        
        Recipe recipe;
        if (!jsonToRecipe(j, recipe)) {
            return false;
        }
        
        // Use provided name or keep original
        if (!new_name.empty()) {
            recipe.name = new_name;
        }
        
        recipe.modified_date = getCurrentTimestamp();
        
        return saveRecipe(recipe);
    } catch (const std::exception& e) {
        std::cerr << "Failed to import recipe: " << e.what() << std::endl;
        return false;
    }
}

Recipe RecipeManager::createRecipeFromPipeline(const std::string& name, VisionPipeline* pipeline) {
    Recipe recipe;
    recipe.name = name;
    recipe.created_date = getCurrentTimestamp();
    recipe.modified_date = getCurrentTimestamp();
    
    // Would need to extract current settings from pipeline
    // For now, return default recipe
    return recipe;
}

} // namespace country_style
