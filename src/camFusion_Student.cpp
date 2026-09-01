/**
 * @file camFusion_Student.cpp
 * @brief Implementation of camera-LIDAR fusion and TTC computation functions
 */

#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


/**
 * @brief Clusters LIDAR points whose projection into the camera falls into bounding box ROIs
 * 
 * For each LIDAR point, projects it into camera coordinates and checks which bounding box
 * (shrunk by shrinkFactor) contains the projected point. Each LIDAR point is associated
 * with at most one bounding box.
 */
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/**
 * @brief Visualizes 3D objects from LIDAR points in a top-down view
 * 
 * Creates a top-view image showing LIDAR points clustered by bounding boxes.
 * The tracked preceding vehicle is highlighted in red, others in blue.
 * Distance markers are drawn as horizontal lines.
 * 
 * Note: Text output is tuned for 2000x2000 image size. For other sizes,
 * text positions should be adjusted proportionally.
 */
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, int trackedPrecedingVehicleTrackID)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // Determine color based on trackID: red for tracked preceding vehicle, blue for others
        cv::Scalar currColor;
        if (trackedPrecedingVehicleTrackID != -1 && it1->trackID == trackedPrecedingVehicleTrackID)
        {
            // Tracked preceding vehicle: RED
            currColor = cv::Scalar(0, 0, 255); // BGR format: red
        }
        else
        {
            // Other vehicles: BLUE
            currColor = cv::Scalar(255, 0, 0); // BGR format: blue
        }

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0,0,0), 2);

        // augment object with some key data including track info
        std::string str1 = cv::format("box_id=%d, track_id=%d, age=%d, #pts=%d", 
                                      it1->boxID, it1->trackID, it1->trackAge, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        std::string str2 = cv::format("xmin=%.2f m, yw=%.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


/**
 * @brief Filters keypoint matches by Euclidean distance percentile
 * 
 * Computes the Euclidean distance between matched keypoints in both frames,
 * then removes the first and last 10% of matches based on distance (outlier removal).
 * 
 * @param matches Vector of keypoint matches to filter
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @return Filtered vector of matches
 */
std::vector<cv::DMatch> filterMatchesByDistance(
    const std::vector<cv::DMatch> &matches,
    const std::vector<cv::KeyPoint> &kptsPrev,
    const std::vector<cv::KeyPoint> &kptsCurr)
{
    if (matches.size() <= 2) return matches;
    
    // Calculate displacement distances
    std::vector<double> distances;
    for (const auto &match : matches)
    {
        const cv::KeyPoint &prevKp = kptsPrev[match.queryIdx];
        const cv::KeyPoint &currKp = kptsCurr[match.trainIdx];
        double dist = cv::norm(prevKp.pt - currKp.pt);
        distances.push_back(dist);
    }
    
    // Sort and filter by percentile (remove first and last 10%)
    std::sort(distances.begin(), distances.end());
    size_t lowerIdx = distances.size() / 10;
    size_t upperIdx = distances.size() * 9 / 10;
    
    if (lowerIdx < distances.size() && upperIdx < distances.size())
    {
        double lowerThreshold = distances[lowerIdx];
        double upperThreshold = distances[upperIdx];
        
        std::vector<cv::DMatch> filtered;
        for (size_t i = 0; i < matches.size(); ++i)
        {
            const cv::KeyPoint &prevKp = kptsPrev[matches[i].queryIdx];
            const cv::KeyPoint &currKp = kptsCurr[matches[i].trainIdx];
            double dist = cv::norm(prevKp.pt - currKp.pt);
            
            if (dist >= lowerThreshold && dist <= upperThreshold)
            {
                filtered.push_back(matches[i]);
            }
        }
        return filtered;
    }
    
    return matches;
}

/**
 * @brief Associates keypoint matches with a single bounding box
 * 
 * Collects all keypoint matches where the current keypoint falls within the bounding box ROI,
 * then applies percentile-based outlier removal to filter noisy matches.
 * Each keypoint match is assigned to at most one bounding box.
 * 
 * @param boundingBox Bounding box to assign matches to (output)
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @param kptMatches All keypoint matches to filter
 */
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Clear the keypoint matches for this bounding box
    boundingBox.kptMatches.clear();

    // Collect matches within bounding box
    std::vector<cv::DMatch> matchesInBox;
    for (const auto &match : kptMatches)
    {
        const cv::KeyPoint &currKp = kptsCurr[match.trainIdx];
        
        // Check if current keypoint is within the bounding box ROI
        if (boundingBox.roi.contains(currKp.pt))
        {
            matchesInBox.push_back(match);
        }
    }
    
    // Apply percentile-based outlier removal
    boundingBox.kptMatches = filterMatchesByDistance(matchesInBox, kptsPrev, kptsCurr);
}


/**
 * @brief Clusters keypoint matches to all bounding boxes, handling overlaps
 * 
 * For each keypoint match, finds all bounding boxes containing the current keypoint
 * and assigns it to the box with the smallest area (most specific box wins).
 * This ensures each match is assigned to at most one bounding box.
 * Also applies percentile-based outlier removal per box.
 * 
 * @param boundingBoxes Vector of all bounding boxes (in/out)
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @param kptMatches All keypoint matches
 * @return Vector of tuples (boxID, matchesBefore, matchesAfter) for statistics
 */
std::vector<std::tuple<int, int, int>> clusterAllKptMatchesWithROI(
    std::vector<BoundingBox> &boundingBoxes,
    std::vector<cv::KeyPoint> &kptsPrev,
    std::vector<cv::KeyPoint> &kptsCurr,
    std::vector<cv::DMatch> &kptMatches)
{
    // Clear all keypoint matches for all bounding boxes
    for (auto &bb : boundingBoxes)
    {
        bb.kptMatches.clear();
    }

    // For each match, find all bounding boxes that contain the current keypoint
    // and assign it to the box with the smallest area (most specific)
    // Track how many matches are assigned to each box before filtering
    std::map<int, int> boxMatchCountsBefore;
    
    for (const auto &match : kptMatches)
    {
        const cv::KeyPoint &currKp = kptsCurr[match.trainIdx];
        
        BoundingBox* bestBB = nullptr;
        int bestArea = std::numeric_limits<int>::max();
        
        // Find all bounding boxes containing this keypoint
        for (auto &bb : boundingBoxes)
        {
            if (bb.roi.contains(currKp.pt))
            {
                int area = bb.roi.width * bb.roi.height;
                if (area < bestArea)
                {
                    bestArea = area;
                    bestBB = &bb;
                }
            }
        }
        
        // If we found at least one bounding box, assign this match to the best one
        if (bestBB != nullptr)
        {
            bestBB->kptMatches.push_back(match);
            boxMatchCountsBefore[bestBB->boxID]++;
        }
    }
    
    // Apply percentile-based outlier removal for each bounding box
    std::vector<std::tuple<int, int, int>> stats; // (boxID, before, after)
    for (auto &bb : boundingBoxes)
    {
        int before = boxMatchCountsBefore[bb.boxID];
        int after = static_cast<int>(bb.kptMatches.size());
        
        if (!bb.kptMatches.empty())
        {
            bb.kptMatches = filterMatchesByDistance(bb.kptMatches, kptsPrev, kptsCurr);
        }
        
        after = static_cast<int>(bb.kptMatches.size());
        stats.emplace_back(bb.boxID, before, after);
    }
    
    return stats;
}


/**
 * @brief Gets keypoint matches for a specific bounding box pair
 * 
 * Returns only matches where the previous keypoint is in prevBB and the current keypoint
 * is in currBB. This ensures we only use matches that belong to the tracked object.
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
    const std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> filteredMatches;
    
    for (const auto &match : kptMatches)
    {
        const cv::KeyPoint &prevKp = kptsPrev[match.queryIdx];
        const cv::KeyPoint &currKp = kptsCurr[match.trainIdx];
        
        // Check if previous keypoint is in previous bounding box
        // and current keypoint is in current bounding box
        if (prevBB.roi.contains(prevKp.pt) && currBB.roi.contains(currKp.pt))
        {
            filteredMatches.push_back(match);
        }
    }
    
    return filteredMatches;
}


/**
 * @brief Computes statistics for keypoint matches before and after filtering
 * 
 * Calculates various statistics including count before/after filtering,
 * percentage of outliers removed, and distance statistics (mean, median, std dev).
 * 
 * @param matchesBefore Matches before filtering
 * @param matchesAfter Matches after filtering
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @return Tuple of (matchesBefore, matchesAfter, outliersRemovedPct, meanDistance, medianDistance, stddevDistance)
 */
std::tuple<int, int, double, double, double, double> 
computeKptMatchStats(const std::vector<cv::DMatch> &matchesBefore,
                     const std::vector<cv::DMatch> &matchesAfter,
                     const std::vector<cv::KeyPoint> &kptsPrev,
                     const std::vector<cv::KeyPoint> &kptsCurr)
{
    int before = static_cast<int>(matchesBefore.size());
    int after = static_cast<int>(matchesAfter.size());
    double outliersRemovedPct = (before > 0) ? (100.0 * (before - after) / before) : 0.0;
    
    double meanDist = 0.0, medianDist = 0.0, stddevDist = 0.0;
    
    if (after > 0)
    {
        std::vector<double> distances;
        for (const auto &match : matchesAfter)
        {
            const cv::KeyPoint &prevKp = kptsPrev[match.queryIdx];
            const cv::KeyPoint &currKp = kptsCurr[match.trainIdx];
            double dist = cv::norm(prevKp.pt - currKp.pt);
            distances.push_back(dist);
        }
        
        if (!distances.empty())
        {
            meanDist = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
            std::sort(distances.begin(), distances.end());
            medianDist = distances[distances.size() / 2];
            double sq_sum = std::inner_product(distances.begin(), distances.end(), distances.begin(), 0.0);
            stddevDist = std::sqrt(sq_sum / distances.size() - meanDist * meanDist);
        }
    }
    
    return std::make_tuple(before, after, outliersRemovedPct, meanDist, medianDist, stddevDist);
}


/**
 * @brief Computes the median of a vector of doubles
 * 
 * Uses std::nth_element for efficient median computation without full sorting.
 * This is used for robust TTC estimation from distance ratios.
 * 
 * @param values Vector of double values
 * @return Median value, or 0.0 if the vector is empty
 */
double computeMedian(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    
    size_t size = values.size();
    size_t mid = size / 2;
    
    if (size % 2 == 1)
    {
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        return values[mid];
    }
    else
    {
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        double valueA = values[mid - 1];
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        double valueB = values[mid];
        return (valueA + valueB) / 2.0;
    }
}


/**
 * @brief Filters out background keypoint matches based on distance ratio clustering
 * 
 * Background matches have distance ratios near 1.0 (no scale change).
 * This function identifies and removes the background cluster to retain only
 * foreground matches (significantly different from 1.0).
 * 
 * Uses median absolute deviation from 1.0 to compute a threshold for background detection.
 * Only filters if a clear bimodal distribution is detected (background + foreground).
 * 
 * @param distRatios Vector of distance ratios to filter
 * @param thresholdMultiplier Multiplier for median deviation (default: 0.2, higher = more aggressive)
 * @param bgMinRatio Minimum background ratio for cluster detection (default: 0.2)
 * @param bgMaxRatio Maximum background ratio for cluster detection (default: 0.6)
 * @param maxDevMultiplier Multiplier for bgThreshold in max deviation check (default: 3.0)
 * @return Filtered vector of distance ratios (foreground only)
 */
std::vector<double> filterBackgroundCluster(const std::vector<double>& distRatios,
                                             double thresholdMultiplier,
                                             double bgMinRatio,
                                             double bgMaxRatio,
                                             double maxDevMultiplier)
{
    if (distRatios.size() < 5) 
        return distRatios; // Not enough data to filter
    
    // Compute statistics to identify clusters
    double minRatio = *std::min_element(distRatios.begin(), distRatios.end());
    double maxRatio = *std::max_element(distRatios.begin(), distRatios.end());
    double medianRatio = computeMedian(distRatios);
    
    // Compute deviation from 1.0 (scale change)
    std::vector<double> deviations;
    for (double ratio : distRatios)
    {
        deviations.push_back(std::abs(ratio - 1.0));
    }
    
    double minDeviation = *std::min_element(deviations.begin(), deviations.end());
    double maxDeviation = *std::max_element(deviations.begin(), deviations.end());
    double medianDeviation = computeMedian(deviations);
    
    // Use threshold multiplier to control filtering aggressiveness
    // Higher values = more aggressive filtering (filters more ratios as background)
    // Lower values = more conservative filtering (keeps more ratios)
    // Note: medianDeviation is typically ~0.007-0.01, so 0.2 multiplier gives ~0.0014-0.002
    double bgThreshold = medianDeviation * thresholdMultiplier;
    
    // Count how many would be filtered
    int backgroundCount = 0;
    for (double dev : deviations)
    {
        if (dev < bgThreshold)
            backgroundCount++;
    }
    
    // Check if we have a clear bimodal distribution (background + foreground)
    // Narrower window (0.2-0.6) better isolates background cluster
    // maxDeviation check with lower multiplier (3.0) to be more sensitive
    double backgroundRatio = static_cast<double>(backgroundCount) / distRatios.size();
    bool hasBackgroundCluster = (maxDeviation > maxDevMultiplier * bgThreshold) && 
                                 (backgroundRatio > bgMinRatio) && 
                                 (backgroundRatio < bgMaxRatio); // Don't filter if >60% would be removed
    
    if (hasBackgroundCluster)
    {
        // Filter out background points (deviation < threshold)
        std::vector<double> filtered;
        for (size_t i = 0; i < distRatios.size(); ++i)
        {
            if (deviations[i] >= bgThreshold)
            {
                filtered.push_back(distRatios[i]);
            }
        }
        
        std::cout << "All ratios counts: " << distRatios.size() << "  / background removed ratios count: " << filtered.size() << std::endl;

        // More defensive: only use filtered if we keep at least 50% of the original ratios
        if (filtered.size() >= 3 && filtered.size() >= distRatios.size() * 0.5)
        {
            return filtered;
        }
    }
    
    // No clear background cluster or not enough foreground points
    // Return original ratios
    return distRatios;
}


/**
 * @brief Computes Time-to-Collision (TTC) based on camera keypoint correspondences
 * 
 * Uses the scale expansion principle from optical flow. For all pairs of matched keypoints,
 * computes distance ratios and uses the median for robust TTC estimation.
 * 
 * Formula: TTC = -dT / (1 - medianDistRatio), where dT = 1/frameRate.
 * 
 * Handles three modes:
 * - ratio > 1.0: Approaching object (positive TTC)
 * - ratio < 1.0: Object moving away (absolute value taken)
 * - ratio == 1.0: No scale change (NaN returned)
 * 
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame
 * @param kptMatches Keypoint matches between frames
 * @param frameRate Camera frame rate in Hz
 * @param minDist Minimum distance threshold to filter noise from very close keypoints
 * @param TTC Output: computed TTC in seconds
 * @param visImg Optional visualization image (currently unused)
 */
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double minDist, double &TTC, cv::Mat *visImg)
{
    // Handle empty input - need at least 2 matches to compute pairwise distances
    if (kptMatches.size() < 2)
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    
    // Compute distance ratios between all pairs of matched keypoints.
    std::vector<double> distRatios;
    
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop
        // get current keypoint and its matched partner in the prev. frame
        const cv::KeyPoint &kpOuterCurr = kptsCurr.at(it1->trainIdx);
        const cv::KeyPoint &kpOuterPrev = kptsPrev.at(it1->queryIdx);
        
        for (auto it2 = it1 + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop
            // get next keypoint and its matched partner in the prev. frame
            const cv::KeyPoint &kpInnerCurr = kptsCurr.at(it2->trainIdx);
            const cv::KeyPoint &kpInnerPrev = kptsPrev.at(it2->queryIdx);
            
            // compute distances between keypoint pairs in both frames
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
            
            // avoid division by zero and very small distances (noise)
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist && distPrev >= minDist)
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    } // eof outer loop over all matched kpts
    
    // Only continue if we have at least 5 valid distance ratios
    if (distRatios.size() < 5)
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    
    // Compute median distance ratio directly from all ratios (no background filtering)
    double medianDistRatio = computeMedian(distRatios);
    
    // Compute time-to-collision using the formula:
    // TTC = -dT / (1 - medianDistRatio)
    // where dT = 1/frameRate
    // We need to handle three modes:
    // 1. ratio > 1.0: Approaching object - scale is expanding, TTC is positive
    // 2. ratio < 1.0: Object moving away - scale is shrinking, TTC is negative (or we take absolute value)
    // 3. ratio == 1.0: No scale change - TTC is infinite or undefined
    double dT = 1.0 / frameRate;
    
    // Check for valid ratio (not equal to 1 to avoid division by zero)
    if (std::abs(medianDistRatio - 1.0) > 0.001)
    {
        // For approaching objects (ratio > 1): TTC is positive
        // For objects moving away (ratio < 1): TTC is negative, take absolute value
        // The formula TTC = -dT / (1 - ratio) gives:
        //   - ratio > 1: positive TTC (approaching)
        //   - ratio < 1: negative TTC (moving away) - we take absolute value
        TTC = -dT / (1.0 - medianDistRatio);
        
        // Handle objects moving away (ratio < 1.0)
        // If TTC is negative, the object is moving away
        // We take the absolute value to represent the time to reach current distance
        if (TTC < 0)
        {
            TTC = std::abs(TTC);
        }
    }
    else
    {
        // Ratio is essentially 1, no scale change detected
        TTC = std::numeric_limits<double>::quiet_NaN();
    }
}


/**
 * @brief Filters a vector of values by percentile range
 * 
 * Removes the first and last N% of sorted values.
 * For example, with lowerPercentile=0.10 and upperPercentile=0.90,
 * removes the bottom 10% and top 10% of values.
 * 
 * @param values Vector of double values to filter
 * @param lowerPercentile Lower percentile bound (0.0-1.0)
 * @param upperPercentile Upper percentile bound (0.0-1.0)
 * @return Filtered vector of values within the percentile range
 */
std::vector<double> filterPercentiles(
    const std::vector<double>& values, 
    double lowerPercentile,
    double upperPercentile)
{
    if (values.empty()) return {};
    
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    
    size_t lowerIdx = static_cast<size_t>(lowerPercentile * sorted.size());
    size_t upperIdx = static_cast<size_t>(upperPercentile * sorted.size());
    
    // Ensure we have at least one element
    if (lowerIdx >= sorted.size()) lowerIdx = sorted.size() - 1;
    if (upperIdx >= sorted.size()) upperIdx = sorted.size() - 1;
    if (lowerIdx > upperIdx) lowerIdx = upperIdx;
    
    return std::vector<double>(sorted.begin() + lowerIdx, sorted.begin() + upperIdx + 1);
}


/**
 * @brief Computes Time-to-Collision (TTC) based on LIDAR measurements
 * 
 * Calculates TTC using the formula: TTC = d1 / v_rel, where d1 is the current distance
 * and v_rel is the relative speed: (d0 - d1) * frameRate.
 * 
 * Supports three methods for distance estimation:
 * - UNFILTERED: Raw mean of all X coordinates
 * - PERCENTILE_MEAN: Mean after removing first/last 10% of sorted X values
 * - PERCENTILE_MEDIAN: Median after removing first/last 10% of sorted X values
 * 
 * @param lidarPointsPrev LIDAR points from previous frame
 * @param lidarPointsCurr LIDAR points from current frame
 * @param frameRate LIDAR frame rate in Hz
 * @param TTC Output: computed TTC in seconds
 * @param method TTC computation method to use
 */
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, 
                     double frameRate,
                     double &TTC,
                     TTCMethod method)
{
    // Handle empty input
    if (lidarPointsPrev.empty() || lidarPointsCurr.empty())
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    
    // Extract X coordinates (forward distance in meters)
    std::vector<double> prevX, currX;
    for (const auto& pt : lidarPointsPrev) prevX.push_back(pt.x);
    for (const auto& pt : lidarPointsCurr) currX.push_back(pt.x);
    
    // Filter valid points (x > 0, within reasonable bounds)
    auto filterValid = [](const std::vector<double>& vals)
    {
        std::vector<double> result;
        for (double x : vals) {
            if (x > 0.0 && x < 100.0) { // Very wide bounds, let percentile do the work
                result.push_back(x);
            }
        }
        return result;
    };
    
    prevX = filterValid(prevX);
    currX = filterValid(currX);
    
    if (prevX.empty() || currX.empty())
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    
    // Get distance estimate based on method
    auto getDistance = [&](const std::vector<double>& vals) -> double
    {
        switch (method)
        {
            case TTCMethod::UNFILTERED:
            {
                double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
                return sum / vals.size();
            }
            
            case TTCMethod::PERCENTILE_MEAN:
            {
                std::vector<double> filtered = filterPercentiles(vals, 0.10, 0.90);
                if (filtered.empty()) return 0.0;
                double sum = std::accumulate(filtered.begin(), filtered.end(), 0.0);
                return sum / filtered.size();
            }
            
            case TTCMethod::PERCENTILE_MEDIAN:
            {
                std::vector<double> filtered = filterPercentiles(vals, 0.10, 0.90);
                if (filtered.empty()) return 0.0;
                size_t mid = filtered.size() / 2;
                if (filtered.size() % 2 == 0) {
                    return (filtered[mid - 1] + filtered[mid]) / 2.0;
                }
                return filtered[mid];
            }
            
            default:
            {
                // Default to PERCENTILE_MEDIAN
                std::vector<double> filtered = filterPercentiles(vals, 0.10, 0.90);
                if (filtered.empty()) return 0.0;
                size_t mid = filtered.size() / 2;
                if (filtered.size() % 2 == 0)
                {
                    return (filtered[mid - 1] + filtered[mid]) / 2.0;
                }
                return filtered[mid];
            }
        }
    };
    
    double d0 = getDistance(prevX);
    double d1 = getDistance(currX);
    
    // Calculate relative speed and TTC
    double v_rel = (d0 - d1) * frameRate;
    
    if ((v_rel > 0.0) && (d1 > 0.0))
    {
        TTC = d1 / v_rel;
    }
    else
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
    }
}


/**
 * @brief Overload for backward compatibility with default PERCENTILE_MEDIAN method
 * 
 * @param lidarPointsPrev LIDAR points from previous frame
 * @param lidarPointsCurr LIDAR points from current frame
 * @param frameRate LIDAR frame rate in Hz
 * @param TTC Output: computed TTC in seconds
 */
void computeTTCLidar(
    std::vector<LidarPoint> &lidarPointsPrev,
    std::vector<LidarPoint> &lidarPointsCurr, 
    double frameRate,
    double &TTC)
{
    computeTTCLidar(lidarPointsPrev, lidarPointsCurr, frameRate, TTC, TTCMethod::PERCENTILE_MEDIAN);
}


/**
 * @brief Finds bounding box ID that contains a given point
 * 
 * @param bounding_boxes Vector of bounding boxes to search
 * @param pt Point to find containing bounding box for
 * @return Box ID if found, or -1 if no bounding box contains the point
 */
inline int findBBIdByPoint(const std::vector<BoundingBox> &bounding_boxes, const cv::Point2f &pt)
{
    auto it = std::find_if(
        bounding_boxes.begin(),
        bounding_boxes.end(),
        [&pt](const BoundingBox &bb) { return bb.roi.contains(pt); });

    return (it != bounding_boxes.end()) ? it->boxID : -1;
}


/**
 * @brief Finds the bounding box that represents the preceding vehicle
 * 
 * Criteria: must have LIDAR points and be most central on ego lane.
 * Uses a scoring system based on:
 * - Number of LIDAR points (more = better)
 * - Horizontal position (closer to image center = better)
 * - Mean X distance of LIDAR points (5-20m range is ideal)
 * 
 * @param boundingBoxes Vector of all bounding boxes
 * @param cameraImg Camera image for determining center
 * @return BoxID of the preceding vehicle, or -1 if not found
 */
int findPrecedingVehicleBox(const std::vector<BoundingBox> &boundingBoxes, const cv::Mat &cameraImg)
{
    if (boundingBoxes.empty() || cameraImg.empty())
    {
        return -1;
    }
    
    int imageCenterX = cameraImg.cols / 2;
    int bestBoxID = -1;
    double bestScore = -1.0;
    
    for (const auto &bb : boundingBoxes)
    {
        // Must have Lidar points to be considered
        if (bb.lidarPoints.empty()) continue;
        
        // Calculate score based on:
        // 1. Number of Lidar points (more = better)
        // 2. Horizontal position (closer to center = better)
        // 3. Mean X distance from Lidar points (closer to camera = better, but not too close)
        int boxCenterX = bb.roi.x + bb.roi.width / 2;
        int distanceFromCenter = std::abs(boxCenterX - imageCenterX);
        
        // Calculate mean X coordinate of Lidar points (forward distance)
        // Prefer boxes with Lidar points at reasonable distance (not too close, not too far)
        double meanX = 0.0;
        double minX = 1e9;
        for (const auto &lp : bb.lidarPoints)
        {
            meanX += lp.x;
            minX = std::min(minX, lp.x);
        }
        meanX /= bb.lidarPoints.size();
        
        // Penalize boxes that are too close (minX < 5m) or too far (meanX > 30m)
        // Ideal range: 5-20 meters in front of camera
        double distancePenalty = 0.0;
        if (minX < 5.0)
        {
            // Too close - heavily penalize
            distancePenalty = 1000.0;
        }
        else if (meanX > 30.0)
        {
            // Too far - penalize
            distancePenalty = (meanX - 30.0) * 10.0;
        }
        
        // Score: high Lidar count + low distance from center - distance penalty
        // Normalize: Lidar count (0-100) + center distance (0-1000) + distance penalty
        double score =
            (static_cast<double>(bb.lidarPoints.size()) *  2.0) -
            (static_cast<double>(distanceFromCenter)    / 20.0) -
            distancePenalty;
        
        if (score > bestScore)
        {
            bestScore = score;
            bestBoxID = bb.boxID;
        }
    }
    
    return bestBoxID;
}


/**
 * @brief Finds bounding box by boxID (const version)
 * 
 * @param boundingBoxes Vector of bounding boxes to search
 * @param boxID Box ID to find
 * @return Iterator to the bounding box if found, or end() if not found
 */
inline std::vector<BoundingBox>::const_iterator findBBById(
    const std::vector<BoundingBox> &boundingBoxes,
    int boxID)
{
    return std::find_if(
        boundingBoxes.begin(),
        boundingBoxes.end(),
        [boxID](const BoundingBox &bb) { return bb.boxID == boxID; });
}

/**
 * @brief Finds bounding box by boxID (non-const version)
 * 
 * @param boundingBoxes Vector of bounding boxes to search
 * @param boxID Box ID to find
 * @return Iterator to the bounding box if found, or end() if not found
 */
inline std::vector<BoundingBox>::iterator findBBById(
    std::vector<BoundingBox> &boundingBoxes,
    int boxID)
{
    return std::find_if(
        boundingBoxes.begin(),
        boundingBoxes.end(),
        [boxID](const BoundingBox &bb) { return bb.boxID == boxID; });
}

/**
 * @brief Finds trackID by boxID from a vector of bounding boxes
 * 
 * @param boundingBoxes Vector of bounding boxes to search
 * @param boxID Box ID to find track for
 * @return TrackID if found, or -1 if not found
 */
inline int findTrackIDForGivenBoxID(const std::vector<BoundingBox> &boundingBoxes, int boxID)
{
    auto it = findBBById(boundingBoxes, boxID);
    return (it != boundingBoxes.end()) ? it->trackID : -1;
}

/**
 * @brief Assigns track ID and age to a single bounding box
 * 
 * Handles three cases:
 * - prevTrackID >= 0: Continue existing track (increment age)
 * - prevTrackID == -1: New detection, assign new track (age = 0)
 * 
 * Updates trackIDMap, trackAgeMap, and currBB in all cases.
 * 
 * @param currBB Current bounding box to assign track to (output)
 * @param prevTrackID Previous track ID, or -1 for new track
 * @param trackIDMap Map from boxID to trackID (updated)
 * @param trackAgeMap Map from trackID to age (updated)
 * @param nextTrackID Next available track ID (incremented for new tracks)
 */
void assignTrackIDToBox(
    BoundingBox &currBB,
    int prevTrackID,
    std::map<int, int> &trackIDMap,
    std::map<int, int> &trackAgeMap,
    int &nextTrackID)
{
    if (prevTrackID >= 0)
    {
        // Continue the track
        currBB.trackID = prevTrackID;
        currBB.trackAge = trackAgeMap[prevTrackID] + 1;
        trackAgeMap[prevTrackID] = currBB.trackAge;
        trackIDMap[currBB.boxID] = prevTrackID;
    }
    else
    {
        // New track (no previous trackID found)
        currBB.trackID = nextTrackID++;
        currBB.trackAge = 0;
        trackIDMap[currBB.boxID] = currBB.trackID;
        trackAgeMap[currBB.trackID] = 0;
    }
}


/**
 * @brief Assigns track IDs and ages to bounding boxes, and finds the tracked preceding vehicle
 * 
 * Uses keypoint matches to maintain object identity across frames.
 * The preceding vehicle is considered tracked if trackedPrecedingVehicleTrackID >= 0.
 * trackedPrecedingVehicleBoxID is cached for faster lookup.
 * 
 * For each current bounding box, determines its trackID based on matches to previous frame.
 * If no match found, a new track is assigned.
 * 
 * @param currBoundingBoxes Current frame bounding boxes
 * @param bbBestMatches Map of best bounding box matches between frames
 * @param prevBoundingBoxes Previous frame bounding boxes
 * @param cameraImg Camera image for finding preceding vehicle
 * @param trackedPrecedingVehicleBoxID Cached boxID for preceding vehicle (in/out)
 * @param trackedPrecedingVehicleTrackID TrackID for preceding vehicle (in/out, >= 0 = tracked)
 * @param trackIDMap Map from current boxID to trackID (output)
 * @param trackAgeMap Map from trackID to age (output)
 * @param nextTrackID Next available track ID (in/out)
 */
void assignTrackIDsAndFindPreceding(
    std::vector<BoundingBox> &currBoundingBoxes,
    const std::map<int, int> &bbBestMatches,
    const std::vector<BoundingBox> &prevBoundingBoxes,
    const cv::Mat &cameraImg,
    int &trackedPrecedingVehicleBoxID,  // Cached boxID for the preceding vehicle (for fast lookup)
    int &trackedPrecedingVehicleTrackID,  // Main identifier: >= 0 means preceding vehicle is tracked
    std::map<int, int> &trackIDMap,  // Maps current boxID to trackID
    std::map<int, int> &trackAgeMap, // Maps trackID to age
    int &nextTrackID)               // Counter for new track IDs (passed by reference)
{
    int currPrecedingVehicleBoxID = -1;
    
    // For each current bounding box, determine its trackID and trackAge
    for (auto &currBB : currBoundingBoxes)
    {
        // Try to find a match from previous frame using std::find_if
        auto matchIt = std::find_if(
            bbBestMatches.begin(),
            bbBestMatches.end(),
            [currBoxID = currBB.boxID](const std::pair<const int, int> &match) 
            { return match.second == currBoxID; });
        
        const int prevBoxIDFromMatches = (matchIt != bbBestMatches.end()) ? matchIt->first : -1;
        
        if (prevBoxIDFromMatches != -1)
        {
            // Find the previous box's trackID using helper function
            const int prevTrackID = findTrackIDForGivenBoxID(prevBoundingBoxes, prevBoxIDFromMatches);
            
            // Found preceding box with track ID, assign track ID to current box
            assignTrackIDToBox(currBB, prevTrackID, trackIDMap, trackAgeMap, nextTrackID);
        }
        else
        {
            // No match found, new detection - assign new track to current box
            assignTrackIDToBox(currBB, -1, trackIDMap, trackAgeMap, nextTrackID);
        }
    }
    
    // Now find and track the preceding vehicle
    if (trackedPrecedingVehicleBoxID == -1)
    {
        // First time: find the preceding vehicle
        trackedPrecedingVehicleBoxID = findPrecedingVehicleBox(currBoundingBoxes, cameraImg);

        // Get its trackID using helper function
        trackedPrecedingVehicleTrackID =
            (trackedPrecedingVehicleBoxID != -1) ?
            findTrackIDForGivenBoxID(currBoundingBoxes, trackedPrecedingVehicleBoxID) :
            -1;  // No preceeding vehicle found!
    }
    else
    {
        // Follow the match from previous tracked box
        auto it = bbBestMatches.find(trackedPrecedingVehicleBoxID);
        if (it != bbBestMatches.end())
        {
            currPrecedingVehicleBoxID = it->second;
            // Update the tracked boxID
            trackedPrecedingVehicleBoxID = currPrecedingVehicleBoxID;
            // Get the trackID for the current box using helper function
            trackedPrecedingVehicleTrackID = findTrackIDForGivenBoxID(currBoundingBoxes, trackedPrecedingVehicleBoxID);
        }
        else
        {
            // Lost track, find again
            trackedPrecedingVehicleBoxID = findPrecedingVehicleBox(currBoundingBoxes, cameraImg);
            // Get its trackID using helper function
            trackedPrecedingVehicleTrackID =
                (trackedPrecedingVehicleBoxID != -1) ?
                    findTrackIDForGivenBoxID(currBoundingBoxes, trackedPrecedingVehicleBoxID) :
                    -1;  // No preceeding vehicle found!
        }
    }
}


/**
 * @brief Matches bounding boxes between previous and current frame based on keypoint correspondences
 * 
 * For each bounding box in the previous frame, finds the bounding box in the current frame
 * with which it has the highest number of keypoint matches. Matches are unique.
 * 
 * @param matches Vector of keypoint matches between frames
 * @param bbBestMatches Output map of (prevBoxID -> currBoxID) representing best matches
 * @param prevFrame Previous frame data
 * @param currFrame Current frame data
 */
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Clear output map
    bbBestMatches.clear();

    // Step 1: Count matches between bounding boxes
    // Structure: prevBoxID -> (currBoxID -> matchCount)
    std::map<int, std::map<int, int>> bbMatchCounts;

    for (const auto &match : matches)
    {
        const cv::KeyPoint &prevKp = prevFrame.keypoints[match.queryIdx];
        const cv::KeyPoint &currKp = currFrame.keypoints[match.trainIdx];

        int prevBoxID = findBBIdByPoint(prevFrame.boundingBoxes, prevKp.pt);
        int currBoxID = findBBIdByPoint(currFrame.boundingBoxes, currKp.pt);

        if (prevBoxID != -1 && currBoxID != -1)
        {
            bbMatchCounts[prevBoxID][currBoxID]++;
        }
    }

    // Step 2: For each previous bounding box, find the current bounding box with most matches
    for (const auto &prevBB : prevFrame.boundingBoxes)
    {
        int prevBoxID = prevBB.boxID;

        if (bbMatchCounts.count(prevBoxID))
        {
            // Find the currBoxID with the maximum count
            auto best = std::max_element(
                bbMatchCounts[prevBoxID].begin(),
                bbMatchCounts[prevBoxID].end(),
                [](const auto &a, const auto &b)
                { return a.second < b.second; });
            
            bbBestMatches[prevBoxID] = best->first;
        }
    }
}

/**
 * @brief Prints bounding box match information for debugging
 * 
 * Shows which previous boxes matched to which current boxes and the match counts.
 * Useful for debugging the matching algorithm.
 * 
 * @param bbBestMatches Map of best bounding box matches
 * @param prevFrame Previous frame data
 * @param currFrame Current frame data
 * @param matches Keypoint matches for detailed count information
 */
void printBBMatchInfo(const std::map<int, int> &bbBestMatches, 
                      const DataFrame &prevFrame, 
                      const DataFrame &currFrame,
                      const std::vector<cv::DMatch> &matches)
{
    std::cout << "Bounding Box Match Information:" << std::endl;
    
    if (bbBestMatches.empty())
    {
        std::cout << "  No matches found" << std::endl;
        return;
    }
    
    // Build a map of match counts: (prevBoxID, currBoxID) -> count
    std::map<std::pair<int, int>, int> matchCounts;
    for (const auto &match : matches)
    {
        const cv::KeyPoint &prevKp = prevFrame.keypoints[match.queryIdx];
        const cv::KeyPoint &currKp = currFrame.keypoints[match.trainIdx];
        
        // Use helper function to find bounding box IDs by point
        int prevBoxID = findBBIdByPoint(prevFrame.boundingBoxes, prevKp.pt);
        int currBoxID = findBBIdByPoint(currFrame.boundingBoxes, currKp.pt);
        
        if (prevBoxID != -1 && currBoxID != -1)
        {
            matchCounts[{prevBoxID, currBoxID}]++;
        }
    }
    
    // Print matches
    for (const auto &matchPair : bbBestMatches)
    {
        int prevBoxID = matchPair.first;
        int currBoxID = matchPair.second;
        int count = matchCounts[{prevBoxID, currBoxID}];
        
        std::cout << "  Prev BB " << prevBoxID << " -> Curr BB " << currBoxID 
                  << " (" << count << " keypoint matches)" << std::endl;
    }
}
