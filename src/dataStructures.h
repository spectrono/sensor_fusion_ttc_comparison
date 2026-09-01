/**
 * @file dataStructures.h
 * @brief Data structures for sensor fusion and TTC computation
 */

#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

/**
 * @struct LidarPoint
 * @brief Represents a single LIDAR point in 3D space
 * 
 * Contains the 3D coordinates and reflectivity of a LIDAR point.
 */
struct LidarPoint { 
    /** @brief X coordinate in meters */
    double x; 
    /** @brief Y coordinate in meters */
    double y; 
    /** @brief Z coordinate in meters */
    double z; 
    /** @brief Point reflectivity value */
    double r; 
};

/**
 * @struct BoundingBox
 * @brief Bounding box around a classified object containing both 2D and 3D data
 * 
 * Represents a detected object with its 2D image ROI, 3D LIDAR points,
 * keypoints, and tracking information.
 */
struct BoundingBox { 
    /** @brief Unique identifier for this bounding box */
    int boxID; 
    /** @brief Unique identifier for the track to which this bounding box belongs */
    int trackID; 
    /** @brief Age of the track in frames (0 = new track) */
    int trackAge; 
    
    /** @brief 2D region-of-interest in image coordinates */
    cv::Rect roi; 
    /** @brief Class ID based on class file provided to YOLO framework */
    int classID; 
    /** @brief Classification confidence/trust score */
    double confidence; 

    /** @brief LIDAR 3D points which project into 2D image ROI */
    std::vector<LidarPoint> lidarPoints; 
    /** @brief Keypoints enclosed by 2D ROI */
    std::vector<cv::KeyPoint> keypoints; 
    /** @brief Keypoint matches enclosed by 2D ROI */
    std::vector<cv::DMatch> kptMatches; 
};

/**
 * @struct DataFrame
 * @brief Represents the available sensor information at the same time instance
 * 
 * Contains all sensor data for a single frame including camera image,
 * keypoints, LIDAR points, bounding boxes, and matches.
 */
struct DataFrame { 
    /** @brief Camera image for this frame */
    cv::Mat cameraImg; 
    
    /** @brief 2D keypoints within camera image */
    std::vector<cv::KeyPoint> keypoints; 
    /** @brief Keypoint descriptors */
    cv::Mat descriptors; 
    /** @brief Keypoint matches between previous and current frame */
    std::vector<cv::DMatch> kptMatches; 
    /** @brief LIDAR points for this frame */
    std::vector<LidarPoint> lidarPoints;

    /** @brief ROI around detected objects in 2D image coordinates */
    std::vector<BoundingBox> boundingBoxes; 
    /** @brief Bounding box matches between previous and current frame */
    std::map<int,int> bbMatches; 
};

#endif /* dataStructures_h */
