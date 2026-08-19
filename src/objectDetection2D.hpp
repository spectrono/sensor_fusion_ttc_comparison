
#ifndef objectDetection2D_hpp
#define objectDetection2D_hpp

#include <stdio.h>
#include <opencv2/core.hpp>

#include "dataStructures.h"

/**
 * @brief Struct to hold letterbox transformation parameters
 *
 * Contains all the information needed to transform bounding boxes
 * from the padded/letterboxed image back to the original image coordinates.
 */
struct LetterboxParams
{
    float ratio;        // Scaling ratio applied to the original image
    int new_width;      // Width of the resized image before padding
    int new_height;     // Height of the resized image before padding
    int pad_left;       // Left padding offset
    int pad_top;        // Top padding offset
};

/**
 * @brief Result of letterbox operation
 *
 * Contains both the padded image and the transformation parameters
 * needed to map detections back to the original image.
 */
struct LetterboxResult
{
    cv::Mat padded;         // The padded/letterboxed image
    LetterboxParams params; // Transformation parameters
};

/**
 * @brief Loads class names from a YAML configuration file (e.g., coco.yaml)
 *
 * Parses the YAML file and extracts the class names (e.g., COCO dataset classes).
 *
 * @param yaml_path Path to the YAML configuration file
 * @return std::vector<std::string> Vector of class names. Empty if file cannot be parsed.
 */
std::vector<std::string> loadClassNames(const std::string& yaml_path);


/**
 * @brief Apply letterbox transformation to an image
 *
 * Resizes the image to fit within target dimensions while maintaining aspect ratio,
 * then adds padding to reach the exact target dimensions.
 *
 * @param img Input image to transform
 * @param target_w Target width
 * @param target_h Target height
 * @param pad_color Color to use for padding (default: gray 114,114,114)
 * @return LetterboxResult containing the padded image and transformation parameters
 */
LetterboxResult letterbox(
    const cv::Mat& img,
    const int target_w,
    const int target_h,
    const cv::Scalar pad_color = cv::Scalar(114, 114, 114));

/**
 * @brief Detects objects in an image using a YOLOv7-tiny ONNX model
 *
 * Modern OpenCV 5.x version using ONNX runtime. Performs object detection
 * using the specified neural network model, confidence threshold, and NMS threshold.
 * Draws bounding boxes and class labels on detected objects when visualization
 * is enabled.
 *
 * The model is trained on the COCO dataset (80 classes listed in coco.yaml).
 * This implementation supports only ONNX format.
 *
 * @param img Input image (BGR format)
 * @param bBoxes Output vector of BoundingBox objects containing detection results
 * @param net Pre-loaded cv::dnn::Net (YOLOv7-tiny ONNX model)
 * @param onnx_input_width Fixed input width expected by the ONNX model (e.g., 640)
 * @param onnx_input_height Fixed input height expected by the ONNX model (e.g., 640)
 * @param confidence_threshold Confidence score threshold for detections (0.0-1.0)
 * @param nms_threshold NMS threshold for suppressing overlapping boxes
 * @param class_names Vector of class names for labeling (from coco.yaml)
 * @param bVis If true, displays results in a window
 */
void detectObjects(
    cv::Mat& img,
    std::vector<BoundingBox>& bBoxes,
    cv::dnn::Net& net,
    const int onnx_input_width,
    const int onnx_input_height,
    const float confidence_threshold,
    const float nms_threshold,
    const std::vector<std::string>& class_names,
    const bool bVis);


#endif // objectDetection2D_hpp
