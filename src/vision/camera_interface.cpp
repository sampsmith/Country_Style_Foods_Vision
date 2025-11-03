#include "camera_interface.h"
#include <iostream>

namespace country_style {

CameraInterface::CameraInterface() 
    : is_initialized_(false), width_(640), height_(480), fps_(30) {
    capture_ = std::make_unique<cv::VideoCapture>();
}

CameraInterface::~CameraInterface() {
    release();
}

bool CameraInterface::initialize(int camera_index) {
    capture_->open(camera_index);
    
    if (!capture_->isOpened()) {
        std::cerr << "Error: Could not open camera " << camera_index << std::endl;
        return false;
    }
    
    // Set default resolution
    capture_->set(cv::CAP_PROP_FRAME_WIDTH, width_);
    capture_->set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    capture_->set(cv::CAP_PROP_FPS, fps_);
    
    // Read actual values (may differ from requested)
    width_ = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = static_cast<int>(capture_->get(cv::CAP_PROP_FPS));
    
    is_initialized_ = true;
    
    std::cout << "Camera initialized: " << width_ << "x" << height_ 
              << " @ " << fps_ << " FPS" << std::endl;
    
    return true;
}

bool CameraInterface::initializeFromFile(const std::string& video_path) {
    capture_->open(video_path);
    
    if (!capture_->isOpened()) {
        std::cerr << "Error: Could not open video file " << video_path << std::endl;
        return false;
    }
    
    width_ = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = static_cast<int>(capture_->get(cv::CAP_PROP_FPS));
    
    is_initialized_ = true;
    
    std::cout << "Video file opened: " << width_ << "x" << height_ 
              << " @ " << fps_ << " FPS" << std::endl;
    
    return true;
}

bool CameraInterface::captureFrame(cv::Mat& frame) {
    if (!is_initialized_ || !capture_->isOpened()) {
        return false;
    }
    
    return capture_->read(frame);
}

void CameraInterface::setResolution(int width, int height) {
    width_ = width;
    height_ = height;
    
    if (capture_->isOpened()) {
        capture_->set(cv::CAP_PROP_FRAME_WIDTH, width);
        capture_->set(cv::CAP_PROP_FRAME_HEIGHT, height);
    }
}

void CameraInterface::setFPS(int fps) {
    fps_ = fps;
    
    if (capture_->isOpened()) {
        capture_->set(cv::CAP_PROP_FPS, fps);
    }
}

void CameraInterface::setBrightness(double brightness) {
    if (capture_->isOpened()) {
        capture_->set(cv::CAP_PROP_BRIGHTNESS, brightness);
    }
}

void CameraInterface::setContrast(double contrast) {
    if (capture_->isOpened()) {
        capture_->set(cv::CAP_PROP_CONTRAST, contrast);
    }
}

int CameraInterface::getWidth() const {
    return width_;
}

int CameraInterface::getHeight() const {
    return height_;
}

int CameraInterface::getFPS() const {
    return fps_;
}

bool CameraInterface::isOpened() const {
    return capture_ && capture_->isOpened();
}

bool CameraInterface::open(int camera_index, int width, int height, int fps) {
    width_ = width;
    height_ = height;
    fps_ = fps;
    return initialize(camera_index);
}

bool CameraInterface::isOpen() const {
    return isOpened();
}

void CameraInterface::release() {
    if (capture_ && capture_->isOpened()) {
        capture_->release();
    }
    is_initialized_ = false;
}

} // namespace country_style
