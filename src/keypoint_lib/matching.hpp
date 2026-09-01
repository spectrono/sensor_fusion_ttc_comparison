/**
 * @file matching.hpp
 * @brief Keypoint matching functions
 *
 * Modern implementations of various keypoint matchers.
 */

#ifndef matching_hpp
#define matching_hpp

#include <numeric>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "detection.hpp"
#include "descriptors.hpp"


// ============================================================================
// Matching Namespace - Header-only library for descriptor matching
// ============================================================================

namespace match
{

/// @brief Find best matches for keypoints in two camera images based on several matching methods
/// Implements both Brute-Force and FLANN matching with NN and KNN selection
/// @param kPtsSource Source keypoints
/// @param kPtsRef Reference keypoints
/// @param descSource Source descriptors
/// @param descRef Reference descriptors
/// @param matches Output vector for matches
/// @param descriptorType Descriptor type: "DES_BINARY" or "DES_HOG"
/// @param matcherType Matcher type: "MAT_BF" (Brute-Force) or "MAT_FLANN" (FLANN)
/// @param selectorType Selector type: "SEL_NN" (Nearest Neighbor) or "SEL_KNN" (k-Nearest Neighbor)
/// @param ratioThreshold Descriptor distance ratio threshold for Lowe's ratio test (default: 0.8, only used with SEL_KNN)
/// @throws std::invalid_argument if matcherType or selectorType is invalid
inline void matchDescriptors(
    std::vector<cv::KeyPoint> &kPtsSource,
    std::vector<cv::KeyPoint> &kPtsRef,
    cv::Mat &descSource, cv::Mat &descRef,
    std::vector<cv::DMatch> &matches,
    std::string descriptorType,
    std::string matcherType,
    std::string selectorType,
    float ratioThreshold = 0.8f)
{
    // Configure matcher based on type
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        // Brute-Force matcher: use Hamming distance for binary descriptors, L2 for float descriptors
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // FLANN-based matcher: Fast approximate nearest neighbor search
        // FLANN in OpenCV 5.0 has limitations with binary descriptors (CV_8U)
        // So we only use FLANN for float descriptors, fall back to BF for binary
        if (descSource.type() == CV_32F && descRef.type() == CV_32F)
        {
            // Use FLANN for float descriptors
            matcher = cv::FlannBasedMatcher::create();
        }
        else
        {
            // For binary descriptors, fall back to Brute-Force with Hamming distance
            std::cout << "Note: FLANN not optimal for binary descriptors, using Brute-Force instead" << std::endl;
            int normType = cv::NORM_HAMMING; // Use Hamming for binary descriptors
            matcher = cv::BFMatcher::create(normType, crossCheck);
        }
    }
    else
    {
        throw std::invalid_argument("Invalid matcher type: " + matcherType + ". Use MAT_BF or MAT_FLANN");
    }

    // Perform matching task based on selector type
    if (selectorType.compare("SEL_NN") == 0)
    { // Nearest Neighbor: find single best match for each descriptor
        matcher->match(descSource, descRef, matches);
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k-Nearest Neighbor: find k best matches for each descriptor
        std::vector<std::vector<cv::DMatch>> knnMatches;
        int k = 2; // Find 2 nearest neighbors
        matcher->knnMatch(descSource, descRef, knnMatches, k);
        
        // Apply Lowe's ratio test: Only keep matches where the ratio between best and second-best is below threshold
        // The ratio test: if distance(best) / distance(second_best) < ratioThreshold, keep the match
        matches.clear();
        for (const auto& matchList : knnMatches)
        {
            if (matchList.size() >= 2)
            { // We need at least 2 matches to apply ratio test
                const cv::DMatch& bestMatch = matchList[0];
                const cv::DMatch& secondBestMatch = matchList[1];
                
                float distanceRatio = bestMatch.distance / secondBestMatch.distance;
                if (distanceRatio < ratioThreshold)
                { // Passed ratio test - this is a good match
                    matches.push_back(bestMatch);
                }
                // If ratio >= threshold, discard the match (both could be wrong)
            }
            else if (matchList.size() == 1)
            { // Only one match found, can't apply ratio test
                matches.push_back(matchList[0]);
            }
            // If empty match list, skip
        }
        std::cout << "Note: Applied descriptor distance ratio test with threshold " << ratioThreshold 
                  << " - filtered to " << matches.size() << " matches from " 
                  << knnMatches.size() << " keypoints" << std::endl;
    }
    else
    {
        throw std::invalid_argument("Invalid selector type: " + selectorType + ". Use SEL_NN or SEL_KNN");
    }
}

} // namespace match


// ============================================================================
// Modern implementations are available via:
//   - kp::detectKeypoints() for keypoint detection
//   - desc::descKeypoints() for descriptor extraction  
//   - match::matchDescriptors() for descriptor matching
// ============================================================================

#endif /* matching_hpp */
