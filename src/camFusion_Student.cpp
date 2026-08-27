
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
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

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
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


// Helper: Filter matches by Euclidean distance percentile
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

// Associate keypoint matches with a single bounding box, handling overlaps
// Each keypoint match is assigned to at most one bounding box (the first one that contains it)
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


// Cluster keypoint matches to all bounding boxes, ensuring each match is assigned to at most one box
// This handles overlapping bounding boxes by assigning each match to the box with the smallest area
// that contains the current keypoint (most specific box wins)
// Returns a vector of statistics for each bounding box: (boxID, matchesBefore, matchesAfter)
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


// Helper function to compute statistics for keypoint matches in a bounding box
// Returns a tuple of (matchesBefore, matchesAfter, outliersRemovedPct, meanDistance, medianDistance, stddevDistance)
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


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


// Helper: Filter values by percentile range (remove first/last N%)
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


// Main TTC computation with method selection
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


// Overload for backward compatibility (default method)
void computeTTCLidar(
    std::vector<LidarPoint> &lidarPointsPrev,
    std::vector<LidarPoint> &lidarPointsCurr, 
    double frameRate,
    double &TTC)
{
    computeTTCLidar(lidarPointsPrev, lidarPointsCurr, frameRate, TTC, TTCMethod::PERCENTILE_MEDIAN);
}


// Helper function: Find bounding box ID that contains a given point
// Returns -1 if no bounding box contains the point
inline int findBBIdByPoint(const std::vector<BoundingBox> &bounding_boxes, const cv::Point2f &pt)
{
    auto it = std::find_if(
        bounding_boxes.begin(),
        bounding_boxes.end(),
        [&pt](const BoundingBox &bb) { return bb.roi.contains(pt); });

    return (it != bounding_boxes.end()) ? it->boxID : -1;
}


// Helper function: Find the bounding box that represents the preceding vehicle
// Criteria: has Lidar points and is most central on ego lane (closest to image center with closest Lidar points)
// Returns the boxID of the preceding vehicle, or -1 if not found
int findPrecedingVehicleBox(const std::vector<BoundingBox> &boundingBoxes, const cv::Mat &cameraImg)
{
    if (boundingBoxes.empty() || cameraImg.empty()) return -1;
    
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
        if (minX < 5.0) {
            // Too close - heavily penalize
            distancePenalty = 1000.0;
        } else if (meanX > 30.0) {
            // Too far - penalize
            distancePenalty = (meanX - 30.0) * 10.0;
        }
        
        // Score: high Lidar count + low distance from center - distance penalty
        // Normalize: Lidar count (0-100) + center distance (0-1000) + distance penalty
        double score = static_cast<double>(bb.lidarPoints.size()) * 2.0 
                     - static_cast<double>(distanceFromCenter) / 20.0
                     - distancePenalty;
        
        if (score > bestScore)
        {
            bestScore = score;
            bestBoxID = bb.boxID;
        }
    }
    
    return bestBoxID;
}


// Helper function: Assign track IDs and ages to bounding boxes
// Uses keypoint matches to maintain object identity across frames
// Also identifies and tracks the preceding vehicle
// Returns the boxID of the tracked preceding vehicle
int assignTrackIDsAndFindPreceding(
    std::vector<BoundingBox> &currBoundingBoxes,
    const std::map<int, int> &bbBestMatches,
    const std::vector<BoundingBox> &prevBoundingBoxes,
    const cv::Mat &cameraImg,
    int &trackedPrecedingVehicleBoxID,
    int &trackedPrecedingVehicleTrackID,
    std::map<int, int> &trackIDMap,  // Maps current boxID to trackID
    std::map<int, int> &trackAgeMap) // Maps trackID to age
{
    int currPrecedingVehicleBoxID = -1;
    static int nextTrackID = 0; // Single static counter for all new tracks
    
    // For each current bounding box, determine its trackID and trackAge
    for (auto &currBB : currBoundingBoxes)
    {
        // Try to find a match from previous frame
        int prevBoxID = -1;
        for (const auto &match : bbBestMatches)
        {
            if (match.second == currBB.boxID)
            {
                prevBoxID = match.first;
                break;
            }
        }
        
        if (prevBoxID != -1)
        {
            // Find the previous box's trackID
            int prevTrackID = -1;
            for (const auto &prevBB : prevBoundingBoxes)
            {
                if (prevBB.boxID == prevBoxID)
                {
                    prevTrackID = prevBB.trackID;
                    break;
                }
            }
            
            if (prevTrackID != -1)
            {
                // Continue the track
                currBB.trackID = prevTrackID;
                currBB.trackAge = trackAgeMap[prevTrackID] + 1;
                trackAgeMap[prevTrackID] = currBB.trackAge;
                trackIDMap[currBB.boxID] = prevTrackID;
            }
            else
            {
                // New track (previous box didn't have a trackID)
                currBB.trackID = nextTrackID++;
                currBB.trackAge = 0;
                trackIDMap[currBB.boxID] = currBB.trackID;
                trackAgeMap[currBB.trackID] = 0;
            }
        }
        else
        {
            // New detection, assign new track
            currBB.trackID = nextTrackID++;
            currBB.trackAge = 0;
            trackIDMap[currBB.boxID] = currBB.trackID;
            trackAgeMap[currBB.trackID] = 0;
        }
    }
    
    // Now find and track the preceding vehicle
    if (trackedPrecedingVehicleBoxID == -1)
    {
        // First time: find the preceding vehicle
        trackedPrecedingVehicleBoxID = findPrecedingVehicleBox(currBoundingBoxes, cameraImg);
        // Get its trackID
        if (trackedPrecedingVehicleBoxID != -1)
        {
            for (const auto &bb : currBoundingBoxes)
            {
                if (bb.boxID == trackedPrecedingVehicleBoxID)
                {
                    trackedPrecedingVehicleTrackID = bb.trackID;
                    break;
                }
            }
        }
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
            // Get the trackID for the current box
            for (const auto &bb : currBoundingBoxes)
            {
                if (bb.boxID == trackedPrecedingVehicleBoxID)
                {
                    trackedPrecedingVehicleTrackID = bb.trackID;
                    break;
                }
            }
        }
        else
        {
            // Lost track, find again
            trackedPrecedingVehicleBoxID = findPrecedingVehicleBox(currBoundingBoxes, cameraImg);
            // Get its trackID
            if (trackedPrecedingVehicleBoxID != -1)
            {
                for (const auto &bb : currBoundingBoxes)
                {
                    if (bb.boxID == trackedPrecedingVehicleBoxID)
                    {
                        trackedPrecedingVehicleTrackID = bb.trackID;
                        break;
                    }
                }
            }
        }
    }
    
    return trackedPrecedingVehicleBoxID;
}


// FP.1 - Match bounding boxes between previous and current frame based on keypoint correspondences
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

// Helper function to print bounding box match information
// Shows which prev boxes matched to which curr boxes and the match counts
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
        
        int prevBoxID = -1;
        for (const auto &prevBB : prevFrame.boundingBoxes)
        {
            if (prevBB.roi.contains(prevKp.pt))
            {
                prevBoxID = prevBB.boxID;
                break;
            }
        }
        
        int currBoxID = -1;
        for (const auto &currBB : currFrame.boundingBoxes)
        {
            if (currBB.roi.contains(currKp.pt))
            {
                currBoxID = currBB.boxID;
                break;
            }
        }
        
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
