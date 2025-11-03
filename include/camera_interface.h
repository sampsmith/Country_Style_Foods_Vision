#ifndef CAMERA_INTERFACE_H
#define CAMERA_INTERFACE_H

#include <opencv2/opencv.hpp>
#include <memory>

namespace country_style {

class CameraInterface {
public:
    CameraInterface();
    ~CameraInterface();

    // Initialize camera with index
    bool initialize(int camera_index = 0);
    
    // Initialize with video file
    bool initializeFromFile(const std::string& video_path);
    
    // Open camera with parameters
    bool open(int camera_index, int width, int height, int fps);
    
    // Check if camera is open
    bool isOpen() const;
    
    // Capture a single frame
    bool captureFrame(cv::Mat& frame);
    
    // Set camera properties
    void setResolution(int width, int height);
    void setFPS(int fps);
    void setBrightness(double brightness);
    void setContrast(double contrast);
    
    // Get camera info
    int getWidth() const;
    int getHeight() const;
    int getFPS() const;
    bool isOpened() const;
    
    // Release camera
    void release();

private:
    std::unique_ptr<cv::VideoCapture> capture_;
    bool is_initialized_;
    int width_;
    int height_;
    int fps_;
};

} // namespace country_style

#endif // CAMERA_INTERFACE_H
