/**
 * @file camFusion.hpp
 * @brief Function declarations for camera-LIDAR fusion and TTC computation
 */

#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include "dataStructures.h"


/**
 * @enum TTCMethod
 * @brief TTC computation methods for LIDAR-based TTC calculation
 */
enum class TTCMethod
{
    UNFILTERED,        /**< Raw mean of all X-coordinates (no filtering) */
    PERCENTILE_MEAN,   /**< Remove first and last 10% of sorted X values, then use mean */
    PERCENTILE_MEDIAN  /**< Remove first and last 10% of sorted X values, then use median */
};

/**
 * @brief Clusters LIDAR points whose projection falls into bounding box ROIs
 * @param boundingBoxes Vector of bounding boxes to associate points with
 * @param lidarPoints Vector of LIDAR points to cluster
 * @param shrinkFactor Factor to shrink bounding box ROI (0.0-1.0)
 * @param P_rect_xx Camera projection matrix (rectified)
 * @param R_rect_xx Rectification matrix
 * @param RT Rotation-translation matrix
 */
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);

/**
 * @brief Computes the median of a vector of doubles
 * @param values Vector of double values
 * @return Median value, or 0.0 if empty
 */
double computeMedian(std::vector<double> values);

/**
 * @brief Filters matches by Euclidean distance percentile
 * @param matches Vector of keypoint matches to filter
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @return Filtered vector of matches
 */
std::vector<cv::DMatch> filterMatchesByDistance(
    const std::vector<cv::DMatch> &matches,
    const std::vector<cv::KeyPoint> &kptsPrev,
    const std::vector<cv::KeyPoint> &kptsCurr);

/**
 * @brief Filters keypoint matches by maximum displacement threshold
 * @param matches Vector of keypoint matches to filter
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @param boundingBox Optional bounding box for threshold calculation
 * @param maxDisplacementThreshold Optional explicit threshold in pixels (overrides box-based threshold)
 * @return Filtered vector of matches with displacement <= threshold
 */
std::vector<cv::DMatch> filterMatchesByDisplacement(
    const std::vector<cv::DMatch> &matches,
    const std::vector<cv::KeyPoint> &kptsPrev,
    const std::vector<cv::KeyPoint> &kptsCurr,
    const BoundingBox *boundingBox = nullptr,
    double maxDisplacementThreshold = -1.0);

/**
 * @brief Associates keypoint matches with a single bounding box
 * @param boundingBox Bounding box to assign matches to
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @param kptMatches All keypoint matches
 */
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);

/**
 * @brief Clusters all keypoint matches to bounding boxes, handling overlaps
 * @param boundingBoxes Vector of all bounding boxes
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @param kptMatches All keypoint matches
 * @return Vector of tuples (boxID, matchesBefore, matchesAfter) for statistics
 */
std::vector<std::tuple<int, int, int>> clusterAllKptMatchesWithROI(
    std::vector<BoundingBox> &boundingBoxes,
    std::vector<cv::KeyPoint> &kptsPrev,
    std::vector<cv::KeyPoint> &kptsCurr,
    std::vector<cv::DMatch> &kptMatches);

/**
 * @brief Matches bounding boxes between previous and current frame based on keypoint correspondences
 * 
 * Each bounding box from the previous frame is matched to the bounding box in the current frame
 * with which it has the highest number of keypoint matches. Matches are unique.
 * 
 * @param matches Vector of keypoint matches between frames
 * @param bbBestMatches Map of (prevBoxID -> currBoxID) representing best matches
 * @param prevFrame Previous frame data
 * @param currFrame Current frame data
 */
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame);

/**
 * @brief Computes statistics for keypoint matches before and after filtering
 * 
 * Returns a tuple containing: (matchesBefore, matchesAfter, outliersRemovedPct, 
 * meanDistance, medianDistance, stddevDistance)
 * 
 * @param matchesBefore Matches before filtering
 * @param matchesAfter Matches after filtering
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @return Tuple of statistics
 */
std::tuple<int, int, double, double, double, double, double, double> 
computeKptMatchStats(const std::vector<cv::DMatch> &matchesBefore,
                     const std::vector<cv::DMatch> &matchesAfter,
                     const std::vector<cv::KeyPoint> &kptsPrev,
                     const std::vector<cv::KeyPoint> &kptsCurr);

/**
 * @brief Gets keypoint matches for a specific bounding box pair
 * 
 * Returns only matches where the previous keypoint is in prevBB and current keypoint is in currBB.
 * 
 * @param prevBB Previous frame bounding box
 * @param currBB Current frame bounding box
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @param kptMatches All keypoint matches
 * @return Filtered vector of matches within the bounding box pair
 */
std::vector<cv::DMatch> getKptMatchesForBBPair(
    const BoundingBox &prevBB,
    const BoundingBox &currBB,
    const std::vector<cv::KeyPoint> &kptsPrev,
    const std::vector<cv::KeyPoint> &kptsCurr,
    const std::vector<cv::DMatch> &kptMatches);

/**
 * @brief Finds the bounding box representing the preceding vehicle
 * 
 * Uses criteria: has LIDAR points and is most central on ego lane
 * (closest to image center with closest LIDAR points).
 * 
 * @param boundingBoxes Vector of all bounding boxes
 * @param cameraImg Camera image for determining center
 * @return BoxID of the preceding vehicle, or -1 if not found
 */
int findPrecedingVehicleBox(const std::vector<BoundingBox> &boundingBoxes, const cv::Mat &cameraImg);

/**
 * @brief Assigns track ID and age to a single bounding box
 * 
 * Handles three cases:
 * - prevTrackID >= 0: Continue existing track
 * - prevTrackID == -1: New detection, assign new track
 * 
 * @param currBB Current bounding box to assign track to
 * @param prevTrackID Previous track ID (or -1 for new track)
 * @param trackIDMap Map from boxID to trackID
 * @param trackAgeMap Map from trackID to age
 * @param nextTrackID Reference to next available track ID
 */
void assignTrackIDToBox(
    BoundingBox &currBB,
    int prevTrackID,
    std::map<int, int> &trackIDMap,
    std::map<int, int> &trackAgeMap,
    int &nextTrackID);

/**
 * @brief Finds trackID by boxID from a vector of bounding boxes
 * @param boundingBoxes Vector of bounding boxes to search
 * @param boxID Box ID to find track for
 * @return TrackID if found, or -1 if not found
 */
int findTrackIDForGivenBoxID(const std::vector<BoundingBox> &boundingBoxes, int boxID);

/**
 * @brief Assigns track IDs and ages to bounding boxes, and finds the tracked preceding vehicle
 * 
 * Uses keypoint matches to maintain object identity across frames.
 * The preceding vehicle is considered tracked if trackedPrecedingVehicleTrackID >= 0.
 * trackedPrecedingVehicleBoxID is cached for faster lookup.
 * 
 * @param currBoundingBoxes Current frame bounding boxes
 * @param bbBestMatches Map of best bounding box matches between frames
 * @param prevBoundingBoxes Previous frame bounding boxes
 * @param cameraImg Camera image for finding preceding vehicle
 * @param trackedPrecedingVehicleBoxID Cached boxID for preceding vehicle (in/out)
 * @param trackedPrecedingVehicleTrackID TrackID for preceding vehicle (in/out)
 * @param trackIDMap Map from boxID to trackID (out)
 * @param trackAgeMap Map from trackID to age (out)
 * @param nextTrackID Next available track ID (in/out)
 */
void assignTrackIDsAndFindPreceding(
    std::vector<BoundingBox> &currBoundingBoxes,
    const std::map<int, int> &bbBestMatches,
    const std::vector<BoundingBox> &prevBoundingBoxes,
    const cv::Mat &cameraImg,
    int &trackedPrecedingVehicleBoxID,
    int &trackedPrecedingVehicleTrackID,
    std::map<int, int> &trackIDMap,
    std::map<int, int> &trackAgeMap,
    int &nextTrackID);

/**
 * @brief Prints bounding box match information for debugging
 * @param bbBestMatches Map of best bounding box matches
 * @param prevFrame Previous frame data
 * @param currFrame Current frame data
 * @param matches Keypoint matches
 */
void printBBMatchInfo(const std::map<int, int> &bbBestMatches, const DataFrame &prevFrame, const DataFrame &currFrame, const std::vector<cv::DMatch> &matches);

/**
 * @brief Visualizes 3D objects from LIDAR points in a top-down view
 * 
 * Creates a top-view image showing LIDAR points clustered by bounding boxes.
 * The tracked preceding vehicle is highlighted in red.
 * 
 * @param boundingBoxes Vector of bounding boxes with LIDAR points
 * @param worldSize World dimensions for scaling
 * @param imageSize Output image dimensions
 * @param bWait Whether to wait for key press before continuing
 * @param trackedPrecedingVehicleTrackID Track ID of the preceding vehicle to highlight
 */
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, int trackedPrecedingVehicleTrackID, int frameIndex = -1, const std::string &dataPath = "");

/**
 * @brief Computes Time-to-Collision (TTC) based on camera keypoint correspondences
 * 
 * Uses the scale expansion principle from optical flow. For all pairs of matched keypoints,
 * computes distance ratios and uses the median for robust TTC estimation.
 * Formula: TTC = -dT / (1 - medianDistRatio), where dT = 1/frameRate.
 * 
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @param kptMatches Keypoint matches between frames
 * @param frameRate Camera frame rate in Hz
 * @param minDist Minimum distance threshold to filter noise
 * @param TTC Output: computed TTC in seconds
 * @param visImg Optional visualization image
 */
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double minDist, double &TTC, cv::Mat *visImg=nullptr);

/**
 * @brief Computes Time-to-Collision (TTC) based on LIDAR measurements
 * 
 * Calculates TTC using the formula: TTC = d1 / v_rel, where d1 is the current distance
 * and v_rel is the relative speed (d0 - d1) * frameRate.
 * 
 * @param lidarPointsPrev LIDAR points from previous frame
 * @param lidarPointsCurr LIDAR points from current frame
 * @param frameRate LIDAR frame rate in Hz
 * @param TTC Output: computed TTC in seconds
 * @param method TTC computation method (default: PERCENTILE_MEDIAN)
 */
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC,
                     TTCMethod method = TTCMethod::PERCENTILE_MEDIAN);

/**
 * @brief Filters a vector of values by percentile range
 * 
 * Removes the first and last N% of sorted values.
 * 
 * @param values Vector of double values to filter
 * @param lowerPercentile Lower percentile bound (0.0-1.0)
 * @param upperPercentile Upper percentile bound (0.0-1.0)
 * @return Filtered vector of values
 */
std::vector<double> filterPercentiles(
    const std::vector<double>& values, 
    double lowerPercentile,
    double upperPercentile);

/**
 * @brief Visualizes keypoint matches on bounding boxes and optionally saves to file
 * 
 * Draws matched keypoints between previous and current frames on the camera image,
 * highlighting keypoints within the tracked bounding box. Useful for FP.5 analysis.
 * 
 * @param img Current camera image
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @param kptMatches Keypoint matches between frames
 * @param trackedBoundingBox Bounding box of the tracked vehicle
 * @param frameIndex Current frame index for file naming
 * @param dataPath Path to save output images (empty to disable saving)
 * @param bVis Enable visualization display
 */
void showKeypointMatchesOverlay(
    cv::Mat &img, 
    std::vector<cv::KeyPoint> &kptsPrev, 
    std::vector<cv::KeyPoint> &kptsCurr, 
    std::vector<cv::DMatch> &kptMatches,
    BoundingBox &trackedBoundingBox,
    int frameIndex = -1,
    const std::string &dataPath = "",
    bool bVis = true);

/**
 * @brief Visualizes all bounding boxes on the camera image
 * 
 * Draws all detected bounding boxes on the camera image for documentation purposes.
 * Each bounding box is drawn with a unique color and labeled with its boxID.
 * Useful for showing the results of object detection in FP.1.
 * 
 * @param img Camera image
 * @param boundingBoxes Vector of all bounding boxes to visualize
 * @param frameIndex Current frame index for file naming
 * @param dataPath Path to save output images
 * @param bVis Enable visualization display
 */
void showAllBoundingBoxes(
    cv::Mat &img,
    std::vector<BoundingBox> &boundingBoxes,
    int frameIndex,
    const std::string &dataPath,
    bool bVis = true);

#endif /* camFusion_hpp */
