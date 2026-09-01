/**
 * @file detection.hpp
 * @brief Keypoint detection functions
 *
 * Modern implementations of various keypoint detectors.
 */

#ifndef detection_hpp
#define detection_hpp

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace kp
{

// ============================================================================
// Type Definitions and Registry
// ============================================================================

/// @brief Type-safe enumeration of all supported keypoint detector types
/// Uses uint8_t as underlying type for memory efficiency (7 values fit in 1 byte)
enum class DetectorType : uint8_t
{
    HARRIS,
    FAST,
    BRISK,
    ORB,
    AKAZE,
    SIFT,
    SHITOMASI
};

/// @brief Convert string to DetectorType
/// @param type String representation of detector type (case-sensitive)
/// @return DetectorType enum value
/// @throws std::invalid_argument if type is not recognized
inline DetectorType detectorTypeFromString(const std::string& type)
{
    static const std::unordered_map<std::string, DetectorType> typeMap =
    {
        {"HARRIS", DetectorType::HARRIS},
        {"FAST", DetectorType::FAST},
        {"BRISK", DetectorType::BRISK},
        {"ORB", DetectorType::ORB},
        {"AKAZE", DetectorType::AKAZE},
        {"SIFT", DetectorType::SIFT},
        {"SHITOMASI", DetectorType::SHITOMASI}
    };

    auto it = typeMap.find(type);
    if (it == typeMap.end())
    {
        throw std::invalid_argument("Unknown detector type: " + type);
    }
    return it->second;
}

/// @brief Convert DetectorType to string
/// @param type DetectorType enum value
/// @return String representation of detector type
inline const char* enumToString(DetectorType type)
{
    switch (type)
    {
        case DetectorType::HARRIS: return "HARRIS";
        case DetectorType::FAST: return "FAST";
        case DetectorType::BRISK: return "BRISK";
        case DetectorType::ORB: return "ORB";
        case DetectorType::AKAZE: return "AKAZE";
        case DetectorType::SIFT: return "SIFT";
        case DetectorType::SHITOMASI: return "SHITOMASI";
        default: return "UNKNOWN";
    }
}

/// @brief Function pointer type for detector functions
using DetectorFunc = void (*)(std::vector<cv::KeyPoint>&, const cv::Mat&, bool);

/// @brief Forward declarations of detector functions
inline void detectHarris(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& img, bool bVis);
inline void detectFast(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& img, bool bVis);
inline void detectBrisk(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& img, bool bVis);
inline void detectOrb(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& img, bool bVis);
inline void detectAkaze(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& img, bool bVis);
inline void detectSift(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& img, bool bVis);
inline void detectShiTomasi(std::vector<cv::KeyPoint>& keypoints, const cv::Mat& img, bool bVis);

/// @brief Get the detector registry mapping strings to function pointers
/// @return Const reference to the registry map
inline const std::unordered_map<std::string, DetectorFunc>& getDetectorRegistry()
{
    static const std::unordered_map<std::string, DetectorFunc> registry =
    {
        {"HARRIS", &detectHarris},
        {"FAST", &detectFast},
        {"BRISK", &detectBrisk},
        {"ORB", &detectOrb},
        {"AKAZE", &detectAkaze},
        {"SIFT", &detectSift},
        {"SHITOMASI", &detectShiTomasi}
    };

    return registry;
}

// ============================================================================
// Main Detection Function
// ============================================================================

/// @brief Detect keypoints using the specified detector type (string-based selection)
/// This is the primary interface for MP.2
/// @param keypoints Output vector for detected keypoints
/// @param img Input grayscale image
/// @param detectorType String specifying the detector type
/// @param visualize If true, display detection results
/// @throws std::invalid_argument if detectorType is not recognized
inline void detectKeypoints(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    const std::string& detectorType,
    bool visualize = false)
{
    keypoints.clear();

    const auto& registry = getDetectorRegistry();
    auto it = registry.find(detectorType);

    if (it == registry.end())
    {
        throw std::invalid_argument("Detector type not found: " + detectorType +
            ". Available types: HARRIS, FAST, BRISK, ORB, AKAZE, SIFT, SHITOMASI");
    }

    it->second(keypoints, img, visualize);
}

// ============================================================================
// Keypoint Filtering (MP.3)
// ============================================================================

/// @brief Remove keypoints outside a given ROI rectangle
/// Only keypoints within the rectangle are retained
/// @param keypoints Input/output vector of keypoints (modified in-place)
/// @param roi Rectangle defining the region of interest
/// @return Number of keypoints removed
inline size_t reduceKeypointsToROI(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Rect& roi)
{
    size_t originalCount = keypoints.size();
    
    // Use remove_if to filter out keypoints outside the ROI
    auto it = std::remove_if(
        keypoints.begin(),
        keypoints.end(),
        [&roi](const cv::KeyPoint& kp) 
        {
            return !roi.contains(kp.pt);
        });
    
    keypoints.erase(it, keypoints.end());
    return originalCount - keypoints.size();
}

// ============================================================================
// Individual Detector Implementations
// ============================================================================

/// @brief HARRIS corner detector
/// Uses cv::cornerHarris for corner detection at a single scale
/// @param keypoints Output vector for detected keypoints
/// @param img Input grayscale image
/// @param bVis If true, display detection results
inline void detectHarris(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    bool bVis)
{
    // Parameters
    int blockSize = 2;           // Neighborhood size for corner response
    int apertureSize = 3;        // Aperture parameter for Sobel (must be odd)
    double k = 0.04;             // Harris detector free parameter (0.04-0.06)
    int minResponse = 100;        // Minimum corner response threshold (on normalized 0-255 scale)

    // Compute Harris corner response
    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k);

    // Normalize for visualization and thresholding
    cv::Mat dstNorm;
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1);
    cv::threshold(dstNorm, dstNorm, minResponse, 255, cv::THRESH_BINARY);

    // Convert to 8-bit
    cv::Mat dstNormScaled;
    cv::convertScaleAbs(dstNorm, dstNormScaled);

    // Find non-zero points (corners)
    std::vector<cv::Point> corners;
    cv::findNonZero(dstNormScaled, corners);

    // Add to keypoints
    for (const auto& corner : corners)
    {
        cv::KeyPoint kp;
        kp.pt = cv::Point2f(corner.x, corner.y);
        kp.size = 2 * apertureSize;
        kp.response = dst.at<float>(corner.y, corner.x);
        keypoints.push_back(kp);
    }

    // Visualization
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("HARRIS Corner Detector Results", 6);
        cv::imshow("HARRIS Corner Detector Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief FAST feature detector
/// Uses cv::FAST for feature detection at a single scale
/// @param keypoints Output vector for detected keypoints
/// @param img Input grayscale image
/// @param bVis If true, display detection results
inline void detectFast(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    bool bVis)
{
    int threshold = 50;                  // Difference threshold for circle pixels
    bool nonmaxSuppression = true;       // Apply non-maximum suppression

    cv::FAST(img, keypoints, threshold, nonmaxSuppression);

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage,
            cv::Scalar::all(-1),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("FAST Detector Results", 6);
        cv::imshow("FAST Detector Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief BRISK feature detector
/// Uses cv::xfeatures2d::BRISK for feature detection
/// @param keypoints Output vector for detected keypoints
/// @param img Input grayscale image
/// @param bVis If true, display detection results
inline void detectBrisk(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::BRISK::create();
    detector->detect(img, keypoints);

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage,
            cv::Scalar::all(-1),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("BRISK Detector Results", 6);
        cv::imshow("BRISK Detector Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief ORB feature detector
/// Uses cv::ORB for feature detection
/// @param keypoints Output vector for detected keypoints
/// @param img Input grayscale image
/// @param bVis If true, display detection results
inline void detectOrb(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    detector->detect(img, keypoints);

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage,
            cv::Scalar::all(-1),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("ORB Detector Results", 6);
        cv::imshow("ORB Detector Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief AKAZE feature detector
/// Uses cv::xfeatures2d::AKAZE for feature detection
/// @param keypoints Output vector for detected keypoints
/// @param img Input grayscale image
/// @param bVis If true, display detection results
inline void detectAkaze(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::AKAZE::create();
    detector->detect(img, keypoints);

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage,
            cv::Scalar::all(-1),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("AKAZE Detector Results", 6);
        cv::imshow("AKAZE Detector Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief SIFT feature detector
/// Uses cv::xfeatures2d::SIFT for feature detection
/// @param keypoints Output vector for detected keypoints
/// @param img Input grayscale image
/// @param bVis If true, display detection results
inline void detectSift(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    detector->detect(img, keypoints);

    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage,
            cv::Scalar::all(-1),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("SIFT Detector Results", 6);
        cv::imshow("SIFT Detector Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief Shi-Tomasi corner detector
/// Uses cv::goodFeaturesToTrack for corner detection at a single scale
/// @param keypoints Output vector for detected keypoints
/// @param img Input grayscale image
/// @param bVis If true, display detection results
inline void detectShiTomasi(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    bool bVis)
{
    // Compute detector parameters based on image size
    int blockSize = 4;
    double maxOverlap = 0.0;
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / std::max(1.0, minDistance);
    double qualityLevel = 0.01;
    double k = 0.04;

    // Apply corner detection
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // Add corners to result vector
    for (const auto& corner : corners)
    {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(corner.x, corner.y);
        newKeyPoint.size = blockSize;
        // Set response to quality measure (inverse of eigenvalue)
        newKeyPoint.response = qualityLevel;
        keypoints.push_back(newKeyPoint);
    }

    // Visualization
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(
            img, keypoints, visImage,
            cv::Scalar::all(-1),
            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("Shi-Tomasi Corner Detector Results", 6);
        cv::imshow("Shi-Tomasi Corner Detector Results", visImage);
        cv::waitKey(0);
    }
}

} // namespace kp

#endif // detection_hpp
