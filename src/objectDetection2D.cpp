
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <yaml-cpp/yaml.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "objectDetection2D.hpp"


/**
 * @brief Loads class names from a YAML configuration file (e.g., coco.yaml)
 *
 * Parses the YAML file and extracts the class names (e.g., COCO dataset classes).
 *
 * @param yaml_path Path to the YAML configuration file
 * @return std::vector<std::string> Vector of class names. Empty if file cannot be parsed.
 */
std::vector<std::string> loadClassNames(const std::string& yaml_path)
{
    std::vector<std::string> class_names;
    
    try
    {
        YAML::Node config = YAML::LoadFile(yaml_path);
        
        if (config["names"])
        {
            YAML::Node names_node = config["names"];
            for (auto const& name : names_node)
            {
                class_names.push_back(name.as<std::string>());
            }
        }
    }
    catch (const YAML::Exception& e)
    {
        std::cerr << "Error parsing YAML file: " << e.what() << std::endl;
    }
    
    return class_names;
}

/**
 * @brief Apply letterbox transformation to an image
 *
 * Resizes the image to fit within target dimensions while maintaining aspect ratio,
 * then adds padding to reach the exact target dimensions.
 *
 * @param img Input image to transform
 * @param target_w Target width
 * @param target_h Target height
 * @param pad_color Color to use for padding
 * @return LetterboxResult containing the padded image and transformation parameters
 */
LetterboxResult letterbox(
    const cv::Mat& img,
    const int target_w,
    const int target_h,
    const cv::Scalar pad_color)
{
    float ratio = std::min(static_cast<float>(target_w) / img.cols, static_cast<float>(target_h) / img.rows);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), ratio, ratio, cv::INTER_LINEAR);

    const int new_w = resized.cols;
    const int new_h = resized.rows;
    const int pad_w = target_w - new_w;
    const int pad_h = target_h - new_h;
    const int pad_left = pad_w / 2;
    const int pad_top = pad_h / 2;

    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        pad_top, pad_h - pad_top,  // top, bottom
        pad_left, pad_w - pad_left,  // left, right
        cv::BORDER_CONSTANT,
        pad_color);
    
    LetterboxResult result;
    result.padded = padded;
    result.params = {ratio, new_w, new_h, pad_left, pad_top};
    return result;
}

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
 * The net is passed as a parameter to avoid reloading it for each detection,
 * reducing computational overhead.
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
    const bool bVis)
{
    // Transform input to be compatible with onnx model used
    LetterboxResult letterbox_result = letterbox(img, onnx_input_width, onnx_input_height);
    cv::Mat padded = letterbox_result.padded;
    
    // Get transformation parameters from letterbox
    const float letterbox_ratio = letterbox_result.params.ratio;
    const int pad_left = letterbox_result.params.pad_left;
    const int pad_top  = letterbox_result.params.pad_top;
    cv::Mat blob;
    cv::dnn::blobFromImage(
        padded, 
        blob, 
        1.0 / 255.0,                // Scale to 0.0 - 1.0
        cv::Size(                   // Used onnx model has a fixed input size of 640x640!
            onnx_input_width,
            onnx_input_height),  
        cv::Scalar(),               // No mean subtraction
        true,                       // swapRB: BGR -> RGB (true)
        false,                      // No center crop
        CV_32F
    );

    // Inference
    net.setInput(blob);
    cv::Mat net_output;
    net.forward(net_output, net.getUnconnectedOutLayersNames());  // Dimensions: [1 x 25200 x 85]

    if (net_output.empty())
    {
        std::cerr << "Error: The model did not deliver any output!" << std::endl;
        return;
    }
    
    // Extract pedictions
    const int net_output_rows = net_output.size[1]; // 25200
    const int net_output_cols = net_output.size[2]; // 85
    cv::Mat net_predictions(net_output_rows, net_output_cols, CV_32F, net_output.ptr<float>());

    std::vector<cv::Rect> net_boxes;
    std::vector<float>    net_confidences;
    std::vector<int>      net_class_ids;

    // Process the predictions
    for (int i = 0; i < net_predictions.rows; ++i)
    {
        // Get the objectness confidence score
        const float obj_confidence = net_predictions.at<float>(i, 4);
        
        if (obj_confidence > confidence_threshold)
        {
            // Find the class with the highest score (columns 5 to 84)
            const cv::Mat scores = net_predictions.row(i).colRange(5, net_predictions.cols);
            cv::Point class_id_point;
            double max_class_score;
            cv::minMaxLoc(scores, nullptr, &max_class_score, nullptr, &class_id_point);

            // Calculate combined confidence score
            const float total_score = static_cast<float>(max_class_score * obj_confidence);

            if (total_score > confidence_threshold)
            {
                float cx = net_predictions.at<float>(i, 0);
                float cy = net_predictions.at<float>(i, 1);
                float w  = net_predictions.at<float>(i, 2);
                float h  = net_predictions.at<float>(i, 3);

                // Transform the bounding box back to the original image
                const int left = static_cast<int>((cx - (w / 2) - pad_left) / letterbox_ratio);
                const int top  = static_cast<int>((cy - (h / 2) - pad_top) / letterbox_ratio);
                const int width = static_cast<int>(w / letterbox_ratio);
                const int height = static_cast<int>(h / letterbox_ratio);

                net_boxes.push_back(cv::Rect(left, top, width, height));
                net_confidences.push_back(total_score);
                net_class_ids.push_back(class_id_point.x);
            }
        }
    }

    // Perform native Opencv 5.x NMS (Handles images with zero detections cleanly)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(net_boxes, net_confidences, confidence_threshold, nms_threshold, indices);

    // Populate the output bounding boxes
    bBoxes.clear();
    int box_id = 0;
    for (int idx : indices)
    {
        BoundingBox bbox;
        bbox.roi = net_boxes[idx];
        bbox.classID = net_class_ids[idx];
        bbox.confidence = static_cast<double>(net_confidences[idx]);
        bbox.boxID = box_id++;
        bbox.trackID = -1;  // Will be set by tracking algorithm
        bBoxes.push_back(bbox);
    }

    if (bVis)
    {
        cv::Mat visImg = img.clone();

        for (size_t i = 0; i < bBoxes.size(); ++i)
        {
            const BoundingBox& bbox = bBoxes[i];
            const cv::Rect box = bbox.roi;
            const int class_id = bbox.classID;
            const float confidence = static_cast<float>(bbox.confidence);

            cv::rectangle(visImg, box, cv::Scalar(0, 255, 0), 2); // Green bounding box
            
            // Create label with class name and confidence
            std::string class_name = (class_id >= 0 && class_id < class_names.size()) ? class_names[class_id] : "unknown";
            std::string label = class_name + ":  " + std::to_string(confidence).substr(0, 4);
        
            // Display label at the top of the bounding box
            int base_line;
            const cv::Size label_size = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &base_line);
            const int      top  = std::max(box.y, label_size.height);
            const int      left = box.x;
            cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0, 200, 200), 1);
        }

        // Show results
        std::string window_name = "Object classification";
        cv::namedWindow(window_name, 10);
        cv::imshow(window_name, visImg);
        cv::waitKey(1);
    }
}
