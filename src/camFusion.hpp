
#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include "dataStructures.h"


// TTC computation methods for Lidar-based TTC calculation
enum class TTCMethod
{
    UNFILTERED,        // Raw mean of all X-coordinates
    PERCENTILE_MEAN,   // Remove first and last 10% of sorted X values, then use mean
    PERCENTILE_MEDIAN  // Remove first and last 10% of sorted X values, then use median
};

void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);
// Cluster all keypoint matches to bounding boxes, handling overlaps
// Returns vector of (boxID, matchesBefore, matchesAfter) for statistics
std::vector<std::tuple<int, int, int>> clusterAllKptMatchesWithROI(
    std::vector<BoundingBox> &boundingBoxes,
    std::vector<cv::KeyPoint> &kptsPrev,
    std::vector<cv::KeyPoint> &kptsCurr,
    std::vector<cv::DMatch> &kptMatches);
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame);

// Helper function for computing keypoint match statistics
std::tuple<int, int, double, double, double, double> 
computeKptMatchStats(const std::vector<cv::DMatch> &matchesBefore,
                     const std::vector<cv::DMatch> &matchesAfter,
                     const std::vector<cv::KeyPoint> &kptsPrev,
                     const std::vector<cv::KeyPoint> &kptsCurr);

// Helper function: Find the bounding box representing the preceding vehicle
int findPrecedingVehicleBox(const std::vector<BoundingBox> &boundingBoxes, const cv::Mat &cameraImg);

// Helper function: Assign track IDs and ages, and find the tracked preceding vehicle
int assignTrackIDsAndFindPreceding(
    std::vector<BoundingBox> &currBoundingBoxes,
    const std::map<int, int> &bbBestMatches,
    const std::vector<BoundingBox> &prevBoundingBoxes,
    const cv::Mat &cameraImg,
    int &trackedPrecedingVehicleBoxID,
    int &trackedPrecedingVehicleTrackID,
    std::map<int, int> &trackIDMap,
    std::map<int, int> &trackAgeMap);
void printBBMatchInfo(const std::map<int, int> &bbBestMatches, const DataFrame &prevFrame, const DataFrame &currFrame, const std::vector<cv::DMatch> &matches);

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait=true, int trackedPrecedingVehicleTrackID = -1);

void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg=nullptr);

// Main Lidar TTC computation with method selection
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC,
                     TTCMethod method = TTCMethod::PERCENTILE_MEDIAN);

// Helper function for percentile filtering
std::vector<double> filterPercentiles(
    const std::vector<double>& values, 
    double lowerPercentile,
    double upperPercentile);

#endif /* camFusion_hpp */
