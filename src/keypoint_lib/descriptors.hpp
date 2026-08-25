#ifndef descriptors_hpp
#define descriptors_hpp

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace desc
{

// ============================================================================
// Type Definitions and Registry
// ============================================================================

/// @brief Type-safe enumeration of all supported descriptor types
/// Uses uint8_t as underlying type for memory efficiency (5 values fit in 1 byte)
enum class DescriptorType : uint8_t
{
    BRIEF,
    ORB,
    FREAK,
    AKAZE,
    SIFT
};

/// @brief Convert string to DescriptorType
/// @param type String representation of descriptor type (case-sensitive)
/// @return DescriptorType enum value
/// @throws std::invalid_argument if type is not recognized
inline DescriptorType descriptorTypeFromString(const std::string& type)
{
    static const std::unordered_map<std::string, DescriptorType> typeMap =
    {
        {"BRIEF", DescriptorType::BRIEF},
        {"ORB", DescriptorType::ORB},
        {"FREAK", DescriptorType::FREAK},
        {"AKAZE", DescriptorType::AKAZE},
        {"SIFT", DescriptorType::SIFT}
    };

    auto it = typeMap.find(type);
    if (it == typeMap.end())
    {
        throw std::invalid_argument("Unknown descriptor type: " + type);
    }
    return it->second;
}

/// @brief Convert DescriptorType to string
/// @param type DescriptorType enum value
/// @return String representation of descriptor type
inline const char* enumToString(DescriptorType type)
{
    switch (type)
    {
        case DescriptorType::BRIEF: return "BRIEF";
        case DescriptorType::ORB: return "ORB";
        case DescriptorType::FREAK: return "FREAK";
        case DescriptorType::AKAZE: return "AKAZE";
        case DescriptorType::SIFT: return "SIFT";
        default: return "UNKNOWN";
    }
}

/// @brief Function pointer type for descriptor functions
/// Signature: (keypoints_in_out, img_input, descriptors_output, visualize)
using DescriptorFunc = void (*)(
    std::vector<cv::KeyPoint>&,
    const cv::Mat&,
    cv::Mat&,
    bool
);

/// @brief Forward declarations of descriptor functions
inline void descBrief(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    bool bVis);
inline void descOrb(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    bool bVis);
inline void descFreak(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    bool bVis);
inline void descAkaze(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    bool bVis);
inline void descSift(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    bool bVis);

/// @brief Get the descriptor registry mapping strings to function pointers
/// @return Const reference to the registry map
inline const std::unordered_map<std::string, DescriptorFunc>& getDescriptorRegistry()
{
    static const std::unordered_map<std::string, DescriptorFunc> registry =
    {
        {"BRIEF", &descBrief},
        {"ORB", &descOrb},
        {"FREAK", &descFreak},
        {"AKAZE", &descAkaze},
        {"SIFT", &descSift}
    };

    return registry;
}

// ============================================================================
// Main Descriptor Extraction Function
// ============================================================================

/// @brief Extract descriptors using the specified descriptor type (string-based selection)
/// This is the primary interface for MP.4
/// @param keypoints Input vector of keypoints (also used for orientation if applicable)
/// @param img Input grayscale image
/// @param descriptors Output matrix for computed descriptors
/// @param descriptorType String specifying the descriptor type
/// @param visualize If true, display extraction results
/// @throws std::invalid_argument if descriptorType is not recognized
inline void descKeypoints(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    const std::string& descriptorType,
    bool visualize = false)
{
    descriptors.release();

    const auto& registry = getDescriptorRegistry();
    auto it = registry.find(descriptorType);

    if (it == registry.end())
    {
        throw std::invalid_argument("Descriptor type not found: " + descriptorType +
            ". Available types: BRIEF, ORB, FREAK, AKAZE, SIFT");
    }

    it->second(keypoints, img, descriptors, visualize);
}

// ============================================================================
// Individual Descriptor Implementations
// ============================================================================

/// @brief BRIEF descriptor extraction
/// Uses cv::xfeatures2d::BriefDescriptorExtractor
/// Produces binary descriptors (CV_8U)
/// @param keypoints Input vector of keypoints
/// @param img Input grayscale image
/// @param descriptors Output matrix for descriptors
/// @param bVis If true, display extraction results
inline void descBrief(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    bool bVis)
{
    // BRIEF parameters
    int bytes = 32;              // 256 bits = 32 bytes (default)
    bool useOrientation = false; // BRIEF does not use keypoint orientation by default

    cv::Ptr<cv::DescriptorExtractor> extractor = 
        cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, useOrientation);
    
    // Perform descriptor extraction
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "BRIEF descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;

    // Visualization
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("BRIEF Descriptor Extraction Results", 6);
        cv::imshow("BRIEF Descriptor Extraction Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief ORB descriptor extraction
/// Uses cv::ORB for descriptor extraction
/// Produces binary descriptors (CV_8U)
/// Note: ORB is both a detector and descriptor
/// Note: Resets keypoint octave to 0 and ensures size >= 31.0f for compatibility
///       with keypoints from other detectors (e.g., SIFT). This is necessary because
///       SIFT keypoints have octave/scale values incompatible with ORB's patch extraction,
///       causing OpenCV resize assertion failures. Scale information is lost but enables
///       cross-detector combinations for MP.8 analysis.
/// @param keypoints Input vector of keypoints (octave and size fields may be modified)
/// @param img Input grayscale image
/// @param descriptors Output matrix for descriptors
/// @param bVis If true, display extraction results
inline void descOrb(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    bool bVis)
{
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::ORB::create();

    // ORB descriptor has specific requirements for keypoint metadata
    // SIFT detector produces keypoints with octave/scale values that are
    // incompatible with ORB's internal patch extraction (resize operation fails
    // with "inv_scale_x > 0" assertion). Reset octave and ensure valid size.
    // Note: This is a simple workaround, which loses original scale information but enables cross-detector compatibility.
    for (auto& kp : keypoints)
    {
        if (kp.octave != 0)
        {
            kp.octave = 0;  // Reset pyramid layer to base level for ORB compatibility
        }
        if (kp.size < 31.0f)  // ORB requires minimum patch size of 31
        {
            kp.size = 31.0f;  // Set to minimum meaningful size for ORB
        }
    }

    // Perform descriptor extraction
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "ORB descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;

    // Visualization
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("ORB Descriptor Extraction Results", 6);
        cv::imshow("ORB Descriptor Extraction Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief FREAK descriptor extraction
/// Uses cv::xfeatures2d::FREAK for descriptor extraction
/// Produces binary descriptors (CV_8U)
/// @param keypoints Input vector of keypoints
/// @param img Input grayscale image
/// @param descriptors Output matrix for descriptors
/// @param bVis If true, display extraction results
inline void descFreak(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    bool bVis)
{
    // FREAK parameters
    bool orientationNormalized = true;
    bool scaleNormalized = true;
    float patternScale = 22.0f;  // Pattern scale factor
    int nOctaves = 4;           // Number of octaves

    cv::Ptr<cv::DescriptorExtractor> extractor = 
        cv::xfeatures2d::FREAK::create(
            orientationNormalized, scaleNormalized, patternScale, nOctaves);

    // Perform descriptor extraction
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "FREAK descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;

    // Visualization
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("FREAK Descriptor Extraction Results", 6);
        cv::imshow("FREAK Descriptor Extraction Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief AKAZE descriptor extraction
/// Uses cv::xfeatures2d::AKAZE for descriptor extraction
/// Produces binary descriptors by default (can be configured for float)
/// @param keypoints Input vector of keypoints
/// @param img Input grayscale image
/// @param descriptors Output matrix for descriptors
/// @param bVis If true, display extraction results
inline void descAkaze(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    bool bVis)
{
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::AKAZE::create();

    // AKAZE has specific requirements for keypoint class_id when used as descriptor
    // Clear class_id to ensure compatibility with keypoints from other detectors
    for (auto& kp : keypoints)
    {
        kp.class_id = 0; // Reset class_id to avoid assertion failures
    }

    // Perform descriptor extraction
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "AKAZE descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;

    // Visualization
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("AKAZE Descriptor Extraction Results", 6);
        cv::imshow("AKAZE Descriptor Extraction Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief SIFT descriptor extraction
/// Uses cv::SIFT for descriptor extraction
/// Produces float descriptors (CV_32F)
/// @param keypoints Input vector of keypoints
/// @param img Input grayscale image
/// @param descriptors Output matrix for descriptors
/// @param bVis If true, display extraction results
inline void descSift(
    std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& img,
    cv::Mat& descriptors,
    bool bVis)
{
    cv::Ptr<cv::DescriptorExtractor> extractor = cv::SIFT::create();

    // Perform descriptor extraction
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "SIFT descriptor extraction in " << 1000 * t / 1.0 << " ms" << std::endl;

    // Visualization
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage,
                          cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow("SIFT Descriptor Extraction Results", 6);
        cv::imshow("SIFT Descriptor Extraction Results", visImage);
        cv::waitKey(0);
    }
}

/// @brief Get all available descriptor type strings
/// @return Const reference to vector of all descriptor type strings
inline const std::vector<std::string>& getAllDescriptorTypes()
{
    static const std::vector<std::string> types =
    {
        "BRIEF",
        "ORB",
        "FREAK",
        "AKAZE",
        "SIFT"
    };

    return types;
}

} // namespace desc

#endif // descriptors_hpp
