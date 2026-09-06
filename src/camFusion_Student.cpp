/**
 * @file camFusion_Student.cpp
 * @brief Implementation of camera-LIDAR fusion and TTC computation functions
 */

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
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
void show3DObjects(
    std::vector<BoundingBox> &boundingBoxes,
    cv::Size worldSize, cv::Size imageSize,
    bool bWait,
    int trackedPrecedingVehicleTrackID,
    int frameIndex,
    const std::string &dataPath)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    // First pass: find global min/max z values across all bounding boxes for consistent coloring
    float global_zwmin=1e8, global_zwmax=-1e8;
    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            float zw = (*it2).z; // world position in m with z facing up from sensor
            global_zwmin = global_zwmin < zw ? global_zwmin : zw;
            global_zwmax = global_zwmax > zw ? global_zwmax : zw;
        }
    }
    
    // Default to reasonable values if no points found
    if (global_zwmin > global_zwmax) {
        global_zwmin = -2.0;
        global_zwmax = 2.0;
    }
    
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
        
        // Second pass: draw points with height-based coloring
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            float zw = (*it2).z; // world position in m with z facing up from sensor
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

            // Height-based coloring: blue (low) to red (high)
            cv::Scalar pointColor;
            if (global_zwmax > global_zwmin) // Avoid division by zero
            {
                // Normalize z to [0, 1] range using global min/max for consistent coloring
                float normalized_z = (zw - global_zwmin) / (global_zwmax - global_zwmin);
                // Blue to red gradient: B(255,0,0) to R(0,0,255) in OpenCV BGR
                pointColor = cv::Scalar(255 * (1 - normalized_z), 0, 255 * normalized_z);
            }
            else
            {
                // All points at same height, use default color
                pointColor = currColor;
            }
            
            // Draw all points with height-based coloring
            cv::circle(topviewImg, cv::Point(x, y), 4, pointColor, -1);
        }

        // draw enclosing rectangle - use red for tracked preceding vehicle, black for others
        cv::Scalar boxColor = (trackedPrecedingVehicleTrackID != -1 && it1->trackID == trackedPrecedingVehicleTrackID) 
                             ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 0, 0);
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), boxColor, 2);

        // augment object with some key data including track info
        std::string str1 = cv::format("box_id=%d, track_id=%d, age=%d, #pts=%d", 
                                      it1->boxID, it1->trackID, it1->trackAge, (int)it1->lidarPoints.size());
        cv::Scalar textColor = (trackedPrecedingVehicleTrackID != -1 && it1->trackID == trackedPrecedingVehicleTrackID) 
                              ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
        cv::putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 1.5, textColor, 2);
        std::string str2 = cv::format("xmin=%.2f m, yw=%.2f m", xwmin, ywmax-ywmin);
        cv::putText(topviewImg, str2, cv::Point2f(left-250, bottom+100), cv::FONT_ITALIC, 1.5, textColor, 2);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }
    
    // Add vertical color bar legend for height visualization with tick marks every 20cm
    int legendBarWidth = 70;  // Width of the color bar
    int legendBarHeight = 600; // Height of the color bar (triple the original size)
    int legendX = imageSize.width - legendBarWidth - 120;  // Position from right
    int legendY = 100;  // Position from top
    
    // Create vertical color bar from blue (bottom/low) to red (top/high)
    cv::Mat colorBar(legendBarHeight, legendBarWidth, CV_8UC3);
    for (int y = 0; y < legendBarHeight; ++y)
    {
        float ratio = static_cast<float>(y) / legendBarHeight;
        cv::Scalar color = cv::Scalar(255 * (1 - ratio), 0, 255 * ratio);
        cv::line(colorBar, cv::Point(0, y), cv::Point(legendBarWidth-1, y), color, 1);
    }
    
    // Overlay color bar on main image
    cv::Mat roi = topviewImg(cv::Rect(legendX, legendY, legendBarWidth, legendBarHeight));
    colorBar.copyTo(roi);
    
    // Add black outline around color bar for better visibility
    cv::rectangle(topviewImg, cv::Rect(legendX, legendY, legendBarWidth, legendBarHeight), 
                 cv::Scalar(0, 0, 0), 2);
    
    // Calculate height range and add tick marks every 20cm
    float heightRange = global_zwmax - global_zwmin;
    if (heightRange > 0.1) // Only add ticks if we have meaningful range
    {
        // Round the range to nearest multiple of 10cm for nice tick placement
        int numTicks = static_cast<int>(heightRange / 0.1) + 1;
        
        for (int i = 0; i <= numTicks; ++i)
        {
            float tickHeight = global_zwmin + (i * 0.1f); // Every 20cm
            if (tickHeight > global_zwmax) break;
            
            // Calculate y position for this tick
            float ratio = (tickHeight - global_zwmin) / heightRange;
            int tickY = legendY + legendBarHeight - static_cast<int>(ratio * legendBarHeight);
            
            // Draw tick mark (horizontal line extending right from color bar)
            int tickLength = 25;
            cv::line(topviewImg, cv::Point(legendX + legendBarWidth, tickY), 
                    cv::Point(legendX + legendBarWidth + tickLength, tickY), 
                    cv::Scalar(0, 0, 0), 1);
            
            // // Add height value label for significant ticks
            // if (i % 2 == 0 || i == numTicks) // Show every other tick to avoid clutter
            // {
            std::string tickText = cv::format("%.1f", tickHeight);
            cv::putText(topviewImg, tickText, cv::Point(legendX + legendBarWidth + tickLength + 5, tickY + 4), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1);
            // }
        }
    }
    
    // Add main annotations
    cv::putText(topviewImg, "Height (z)", cv::Point(legendX - 50, legendY - 15), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    
    // Add unit label
    cv::putText(topviewImg, "[m]", cv::Point(legendX + legendBarWidth + 20, legendY - 15), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    
    // Add low/high indicators
    std::string lowText = cv::format("Low: %.2f m", global_zwmin);
    std::string highText = cv::format("High: %.2f m", global_zwmax);
    cv::putText(topviewImg, lowText, cv::Point(legendX - 100, legendY + legendBarHeight + 25), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1); // Blue for low
    cv::putText(topviewImg, highText, cv::Point(legendX - 100, legendY + legendBarHeight + 50), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1); // Red for high

    // Save the bird's-eye view image to file for FP.5 analysis
    if (frameIndex >= 0 && !dataPath.empty())
    {
        std::string filename = dataPath + "preceding_vehicle_lidar_bev_" + std::to_string(frameIndex) + ".png";
        cv::imwrite(filename, topviewImg);
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
 * @brief Filters keypoint matches by maximum displacement threshold
 * 
 * For FP.3: Removes matches where the Euclidean displacement between matched keypoints
 * exceeds a threshold based on the bounding box size. Large displacements can indicate
 * mismatches or objects moving unrealistically fast. This is different from the percentile
 * filtering in filterMatchesByDistance.
 * 
 * The threshold is set to 1/8 of the bounding box's longest dimension (width or height).
 * This ensures we only keep matches with reasonable displacement for the tracked object.
 * 
 * @param matches Vector of keypoint matches to filter
 * @param kptsPrev Keypoints from previous frame
 * @param kptsCurr Keypoints from current frame  
 * @param boundingBox Bounding box for threshold calculation (optional, can be null)
 * @param maxDisplacementThreshold Optional explicit threshold in pixels (overrides box-based threshold)
 * @return Filtered vector of matches with displacement <= threshold
 */
std::vector<cv::DMatch> filterMatchesByDisplacement(
    const std::vector<cv::DMatch> &matches,
    const std::vector<cv::KeyPoint> &kptsPrev,
    const std::vector<cv::KeyPoint> &kptsCurr,
    const BoundingBox *boundingBox,
    double maxDisplacementThreshold)
{
    if (matches.empty()) return matches;
    
    // Determine threshold
    double threshold = maxDisplacementThreshold;
    if (threshold <= 0 && boundingBox != nullptr)
    {
        // Use 1/8 of the bounding box's longest dimension as threshold
        int longestDim = std::max(boundingBox->roi.width, boundingBox->roi.height);
        threshold = longestDim / 8.0;
    }
    else if (threshold <= 0)
    {
        // Default threshold if no bounding box provided
        threshold = 100.0; // pixels - reasonable default for KITTI
    }
    
    std::vector<cv::DMatch> filtered;
    for (const auto &match : matches)
    {
        const cv::KeyPoint &prevKp = kptsPrev[match.queryIdx];
        const cv::KeyPoint &currKp = kptsCurr[match.trainIdx];
        double displacement = cv::norm(prevKp.pt - currKp.pt);
        
        // Keep matches with displacement <= threshold (remove large displacements)
        if (displacement <= threshold)
        {
            filtered.push_back(match);
        }
    }
    
    return filtered;
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
    
    // Apply displacement-based outlier removal for FP.3
    boundingBox.kptMatches = filterMatchesByDisplacement(matchesInBox, kptsPrev, kptsCurr, &boundingBox, -1.0);
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
    
    // Apply displacement-based outlier removal for each bounding box (FP.3)
    // Uses upper threshold on keypoint displacement (removes large displacements)
    // Threshold: 1/8 of bounding box's longest dimension
    std::vector<std::tuple<int, int, int>> stats; // (boxID, before, after)
    for (auto &bb : boundingBoxes)
    {
        int before = boxMatchCountsBefore[bb.boxID];
        int after = static_cast<int>(bb.kptMatches.size());
        
        if (!bb.kptMatches.empty())
        {
            // For FP.3: Use displacement filtering with upper threshold based on box size
            bb.kptMatches = filterMatchesByDisplacement(bb.kptMatches, kptsPrev, kptsCurr, &bb, -1.0);
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
 * @return Tuple of (matchesBefore, matchesAfter, outliersRemovedPct, meanDistance, medianDistance, stddevDistance, minDistance, maxDistance)
 */
std::tuple<int, int, double, double, double, double, double, double> 
computeKptMatchStats(const std::vector<cv::DMatch> &matchesBefore,
                     const std::vector<cv::DMatch> &matchesAfter,
                     const std::vector<cv::KeyPoint> &kptsPrev,
                     const std::vector<cv::KeyPoint> &kptsCurr)
{
    int before = static_cast<int>(matchesBefore.size());
    int after = static_cast<int>(matchesAfter.size());
    double outliersRemovedPct = (before > 0) ? (100.0 * (before - after) / before) : 0.0;
    
    double meanDist = 0.0, medianDist = 0.0, stddevDist = 0.0;
    double minDist = 0.0, maxDist = 0.0;
    
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
            minDist = distances.front();
            maxDist = distances.back();
        }
    }
    
    return std::make_tuple(before, after, outliersRemovedPct, meanDist, medianDist, stddevDist, minDist, maxDist);
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
    
    // Compute median distance ratio directly from all pairwise keypoint distance ratios
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

/**
 * @brief Visualizes keypoint matches on bounding boxes and optionally saves to file
 * 
 * Draws matched keypoints between previous and current frames on the camera image,
 * highlighting keypoints within the tracked bounding box. Useful for FP.5 analysis.
 */
void showKeypointMatchesOverlay(
    cv::Mat &img, 
    std::vector<cv::KeyPoint> &kptsPrev, 
    std::vector<cv::KeyPoint> &kptsCurr, 
    std::vector<cv::DMatch> &kptMatches,
    BoundingBox &trackedBoundingBox,
    int frameIndex,
    const std::string &dataPath,
    bool bVis)
{
    // Create visualization image
    cv::Mat visImg = img.clone();
    
    // Draw bounding box of tracked vehicle
    cv::rectangle(visImg, trackedBoundingBox.roi, cv::Scalar(0, 255, 0), 2);
    
    // Define colors
    cv::Scalar colorPrev(0, 255, 255);   // Yellow for previous frame keypoints
    cv::Scalar colorCurr(0, 255, 0);     // Green for current frame keypoints
    
    // Draw all keypoints from current frame within the bounding box
    int kpRadius = 3;  // Radius of keypoint circles
    int matchedCount = 0;
        
    // Draw matched keypoints with lines connecting previous to current
    // Note: The input kptMatches should already be filtered to contain only matches
    // where previous keypoint is in the previous bounding box AND current keypoint is in the current bounding box
    for (const auto &match : kptMatches)
    {
        const cv::KeyPoint &prevKp = kptsPrev[match.queryIdx];
        const cv::KeyPoint &currKp = kptsCurr[match.trainIdx];
        
        // Draw circles at both keypoint positions
        cv::circle(visImg, prevKp.pt, kpRadius, colorPrev, 0, cv::LINE_AA);
        cv::circle(visImg, currKp.pt, kpRadius, colorCurr, 0, cv::LINE_AA);
        
        ++matchedCount;
    }
    
    // Add legend for the visualization
    int legendX = 50;
    int legendY = 90;
    int legendSpacing = 30;
    
    // Magenta color for better visibility on bright images (BGR format)
    cv::Scalar legendColor = cv::Scalar(0, 255, 0); // Green
    cv::Scalar textColor = cv::Scalar(255, 0, 255); // Magenta
    
    // Title
    cv::putText(visImg, "Keypoint Matches on Tracked Vehicle", cv::Point(legendX, legendY - 40), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, legendColor, 2);
    
    // Legend items
    cv::putText(visImg, "Previous Frame", cv::Point(legendX, legendY + legendSpacing * 0), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1);
    cv::putText(visImg, "Current Frame", cv::Point(legendX, legendY + legendSpacing * 1), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1);
    
    // Legend color indicators
    cv::circle(visImg, cv::Point(legendX - 15, legendY + legendSpacing * 0 - 2), kpRadius, colorPrev, -1);
    cv::circle(visImg, cv::Point(legendX - 15, legendY + legendSpacing * 1 - 2), kpRadius, colorCurr, -1);
    
    // Add match count information
    cv::putText(visImg, cv::format("Matches within BB: %d", matchedCount), cv::Point(legendX, legendY + legendSpacing * 3 + 10), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1);
    
    // Add bounding box info
    cv::putText(visImg, cv::format("BB ID: %d, Track ID: %d", trackedBoundingBox.boxID, trackedBoundingBox.trackID), 
               cv::Point(legendX, legendY + legendSpacing * 4 + 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1);
    
    // Save image to file if dataPath is provided and frameIndex >= 0
    if (frameIndex >= 0 && !dataPath.empty() && bVis)
    {
        std::string filename = dataPath + "keypoint_matches_" + std::to_string(frameIndex) + ".png";
        cv::imwrite(filename, visImg);
    }
    
    // Display image if visualization is enabled
    if (bVis)
    {
        std::string windowName = "Keypoint Matches on Tracked Vehicle - Frame " + std::to_string(frameIndex);
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::imshow(windowName, visImg);
        cv::waitKey(10); // Short wait for visualization
    }
}


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
    bool bVis)
{
    // Create visualization image
    cv::Mat visImg = img.clone();
    
    // Define colors for different bounding boxes
    std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0),     // Blue
        cv::Scalar(0, 255, 0),     // Green
        cv::Scalar(0, 0, 255),     // Red
        cv::Scalar(255, 255, 0),   // Yellow
        cv::Scalar(0, 255, 255),   // Cyan
        cv::Scalar(255, 0, 255),   // Magenta
        cv::Scalar(128, 0, 0),     // Dark Blue
        cv::Scalar(0, 128, 0),     // Dark Green
        cv::Scalar(0, 0, 128),     // Dark Red
        cv::Scalar(128, 128, 0),   // Dark Yellow
        cv::Scalar(0, 128, 128),   // Dark Cyan
        cv::Scalar(128, 0, 128),   // Dark Magenta
        cv::Scalar(255, 128, 0),   // Orange
        cv::Scalar(128, 255, 0),   // Light Green
        cv::Scalar(0, 128, 255),   // Light Blue
        cv::Scalar(255, 0, 128),   // Pink
    };
    
    // Draw all bounding boxes
    for (size_t i = 0; i < boundingBoxes.size(); ++i)
    {
        BoundingBox &bb = boundingBoxes[i];
        cv::Scalar color = colors[i % colors.size()];
        
        // Draw bounding box rectangle
        cv::rectangle(visImg, 
                     cv::Point(bb.roi.x, bb.roi.y),
                     cv::Point(bb.roi.x + bb.roi.width, bb.roi.y + bb.roi.height),
                     color, 3);
        
        // Add label with boxID
        std::string label = cv::format("ID: %d", bb.boxID);
        cv::putText(visImg, label, 
                   cv::Point(bb.roi.x + 10, bb.roi.y + 30),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 255), 2);
        
        // Add trackID if available
        if (bb.trackID >= 0)
        {
            std::string trackLabel = cv::format("Track: %d", bb.trackID);
            cv::putText(visImg, trackLabel,
                       cv::Point(bb.roi.x + 10, bb.roi.y + 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 255), 1);
        }
    }
    
    // Add title
    cv::putText(visImg, cv::format("Frame %d: All Detected Bounding Boxes", frameIndex),
               cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 0, 255), 2);
        
    // Save image to file
    if (frameIndex >= 0 && !dataPath.empty())
    {
        std::string filename = dataPath + "fp1_all_bounding_boxes_frame_" + std::to_string(frameIndex) + ".png";
        cv::imwrite(filename, visImg);
        std::cout << "Saved all bounding boxes visualization to: " << filename << std::endl;
    }
    
    // Display image if visualization is enabled
    if (bVis)
    {
        std::string windowName = "All Bounding Boxes - Frame " + std::to_string(frameIndex);
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::imshow(windowName, visImg);
        cv::waitKey(10);
    }
}
