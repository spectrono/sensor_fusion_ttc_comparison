/**
 * @file FinalProject_Camera.cpp
 * @brief Main application for sensor fusion and TTC computation
 *
 * This is the main entry point for the 3D object tracking application.
 * It loads KITTI dataset images and LIDAR data, performs object detection,
 * feature matching, and computes TTC using both LIDAR and camera sensors.
 */

/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

// Keypoint library from previous project - modern detector/descriptor implementations
#include "keypoint_lib/detection.hpp"
#include "keypoint_lib/descriptors.hpp"
#include "keypoint_lib/matching.hpp"
#include "keypoint_lib/analysis.hpp"

using namespace std;
using namespace kp;


/**
 * @brief Main entry point for the 3D object tracking application
 *
 * Processes a sequence of KITTI dataset frames, performing:
 * 1. Object detection using YOLOv7-tiny ONNX model
 * 2. LIDAR point clustering within bounding boxes
 * 3. Keypoint detection and matching between frames
 * 4. Bounding box matching across frames
 * 5. Object tracking with track IDs
 * 6. TTC computation using both LIDAR and camera sensors
 * 7. Visualization of 3D objects and camera images
 *
 * @param argc Number of command-line arguments
 * @param argv Command-line arguments
 * @return 0 on success
 */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load (10 fps dataset)
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)
    
    // Camera TTC: minimum distance threshold for keypoint pairs (single source of truth)
    double cameraMinDist = 110.0;

    // Comprehensive testing mode: test all detector-descriptor combinations
    bool bTestAllCombinations = false;
    
    // For comprehensive testing: store results for each combination
    struct CombinationResult {
        string detectorType;
        string descriptorType;
        int imageIndex;
        int keypointCount;
        cv::Size descriptorSize;
        double detectionTimeMs;
        double descriptorTimeMs;
    };
    std::vector<CombinationResult> combinationResults;
    
    // For FP.1 analysis: store bounding box match data
    struct BBMatchData {
        int frameIndex;
        int trackId;  // Track ID of the matched object (for filtering to preceding vehicle)
        int prevBoxId;
        int currBoxId;
        int matchCount;
    };
    std::vector<BBMatchData> bbMatchResults;
    
    // Current detector and descriptor types for CSV export
    std::string currentDetectorType = "SHITOMASI";
    std::string currentDescriptorType = "ORB";

    // FP.2: TTC method selection
    TTCMethod ttcLidarMethod = TTCMethod::PERCENTILE_MEDIAN; // Default method
    bool bTestAllTTCMethods = true; // Enable comparison mode for FP.2 analysis
    
    // For storing TTC results from different methods
    struct TTCResult
    {
        int frameIndex;
        int trackId;  // Track ID of the preceding vehicle (stable across frames)
        int prevBoxId;
        int currBoxId;
        std::map<std::string, double> ttcValues; // method name -> TTC value
    };
    std::vector<TTCResult> ttcResults;

    // For FP.3: Store keypoint match filtering statistics
    struct KptMatchStats
    {
        int frameIndex;
        int trackId;  // Track ID of the bounding box
        int boxId;
        int matchesBefore;
        int matchesAfter;
        double outliersRemovedPct;
        double meanDistance;
        double medianDistance;
        double stddevDistance;
    };
    std::vector<KptMatchStats> kptMatchStats;
    bool bRecordKptStats = true; // Enable to record FP.3 statistics

    // For FP.4: Store camera-based TTC results
    struct CameraTTCResult
    {
        int frameIndex;
        int trackId;  // Track ID of the preceding vehicle (stable across frames)
        int prevBoxId;
        int currBoxId;
        double ttcCamera;
    };
    std::vector<CameraTTCResult> cameraTTCResults;
    bool bRecordCameraTTC = true; // Enable to record FP.4 camera TTC results
    
    // For FP.4 debugging: Store distance ratio statistics to analyze background filtering
    struct CameraTTCScaleStats
    {
        int frameIndex;
        int trackId;  // Track ID of the preceding vehicle (stable across frames)
        int numRatios;
        double minRatio;
        double maxRatio;
        double medianRatio;
        double meanRatio;
        double stddevRatio;
        int numFiltered; // Number of ratios after background filtering
        double filteredMinRatio;
        double filteredMaxRatio;
        double filteredMedianRatio;
    };
    std::vector<CameraTTCScaleStats> cameraTTCScaleStats;
    bool bRecordCameraScaleStats = true; // Enable to record distance ratio statistics

    // Load class names from coco.yaml
    std::string yolo_base_path = dataPath + "dat/yolo/";
    std::string coco_yaml_path = yolo_base_path + "coco.yaml";
    std::vector<std::string> class_names = loadClassNames(coco_yaml_path);
    if (class_names.empty())
    {
        std::cerr << "Warning: Could not load class names from " << coco_yaml_path << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Loaded " << class_names.size() << " class names from coco.yaml" << std::endl;
    }

    // Load yolov7.onnx model
    std::string onnx_file = "yolov7-tiny.onnx";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(yolo_base_path + onnx_file);
    if (net.empty())
    {
        std::cerr << "Error when loading the onnx model!" << std::endl;
        return -1;
    }
    else
    {
        std::cerr << "Successfully loaded " <<  (yolo_base_path + onnx_file) << std::endl;
    }

    // Setup calibration data
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera (10 fps dataset)
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = true;             // visualize results
    
    // For tracking objects across frames
    // trackedPrecedingVehicleTrackID is the main identifier: >= 0 means preceding vehicle is tracked
    // trackedPrecedingVehicleBoxID is cached for fast lookup (avoids searching through all boxes)
    int trackedPrecedingVehicleBoxID = -1; // Cached boxID for fast lookup
    int trackedPrecedingVehicleTrackID = -1; // Main identifier: >= 0 means tracked, -1 means not yet initialized
    std::map<int, int> trackIDMap; // Maps current boxID to trackID
    std::map<int, int> trackAgeMap; // Maps trackID to age
    int nextTrackID = 0; // Counter for assigning new unique track IDs (0, 1, 2, ...)

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */
        
        bVis = true; // Ensure visualization is enabled for each frame

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);
        
        // Limit data buffer size
        if (dataBuffer.size() > dataBufferSize)
        {
            dataBuffer.erase(dataBuffer.begin());
        }

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */
        
        // UPDATED version of detectObjects() to be compatible with modern opencv 5.x approach!!!
        const float confThreshold = 0.3;
        const float nmsThreshold = 0.4;
        detectObjects(
            (dataBuffer.end() - 1)->cameraImg,
            (dataBuffer.end() - 1)->boundingBoxes,
            net,
            640,  // Onnx model's fixed input width
            640,  // Onnx model's fixed input height
            confThreshold,
            nmsThreshold,
            class_names,
            false); // Disable internal visualization

        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
    
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        // For frame 0, assign initial track IDs starting from nextTrackID (0)
        // and find the preceding vehicle
        if (dataBuffer.size() == 1)
        {
            // Assign track IDs starting from nextTrackID for frame 0
            for (size_t i = 0; i < (dataBuffer.end() - 1)->boundingBoxes.size(); ++i)
            {
                auto &bb = (dataBuffer.end() - 1)->boundingBoxes[i];
                bb.trackID = nextTrackID++; // Assign and increment: 0, 1, 2, ...
                bb.trackAge = 0;
                // Initialize the track maps
                trackIDMap[bb.boxID] = bb.trackID;
                trackAgeMap[bb.trackID] = bb.trackAge;
            }
            
            // Find and track the preceding vehicle for frame 0
            // trackedPrecedingVehicleTrackID is the main identifier (>= 0 means tracked)
            // trackedPrecedingVehicleBoxID is cached for fast lookup
            trackedPrecedingVehicleBoxID = findPrecedingVehicleBox((dataBuffer.end() - 1)->boundingBoxes, (dataBuffer.end() - 1)->cameraImg);
            trackedPrecedingVehicleTrackID = (trackedPrecedingVehicleBoxID != -1)
                ? findTrackIDForGivenBoxID((dataBuffer.end() - 1)->boundingBoxes, trackedPrecedingVehicleBoxID)
                : -1;
            
            // Visualize bounding boxes for frame 0
            if (bVis)
            {
                visualizeBoundingBoxes(
                    (dataBuffer.end() - 1)->cameraImg,
                    (dataBuffer.end() - 1)->boundingBoxes,
                    class_names,
                    trackedPrecedingVehicleTrackID);
            }
        }

        cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by 10% to balance overlap reduction and data retention
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects for first frame
        if (dataBuffer.size() == 1 && bVis)
        {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true, trackedPrecedingVehicleTrackID);
        }

        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        
        
        // REMOVED: continue statement to enable keypoint processing
        // continue; // skips directly to the next image without processing what comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        /* DETECT IMAGE KEYPOINTS */

        // Get all available detector and descriptor types from keypoint_lib
        const auto& allDetectorTypes = getAllDetectorTypes();
        const auto& allDescriptorTypes = desc::getAllDescriptorTypes();

        if (bTestAllCombinations)
        {
            // Test all detector-descriptor combinations
            cout << "=== Testing all detector-descriptor combinations ===" << endl;
            cout << "Detectors: ";
            for (const auto& det : allDetectorTypes) cout << det << " ";
            cout << endl << "Descriptors: ";
            for (const auto& desc : allDescriptorTypes) cout << desc << " ";
            cout << endl << endl;

            for (const auto& detectorType : allDetectorTypes)
            {
                for (const auto& descriptorType : allDescriptorTypes)
                {
                    vector<cv::KeyPoint> keypoints;
                    cv::Mat descriptors;

                    // Detect keypoints with timing
                    double t = static_cast<double>(cv::getTickCount());
                    detectKeypoints(keypoints, imgGray, detectorType, false);
                    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
                    double detectionTimeMs = 1000.0 * t;

                    // Extract descriptors with timing
                    double t2 = static_cast<double>(cv::getTickCount());
                    desc::descKeypoints(keypoints, imgGray, descriptors, descriptorType, false);
                    t2 = (static_cast<double>(cv::getTickCount()) - t2) / cv::getTickFrequency();
                    double descriptorTimeMs = 1000.0 * t2;

                    // Record results
                    CombinationResult result;
                    result.detectorType = detectorType;
                    result.descriptorType = descriptorType;
                    result.imageIndex = imgStartIndex + imgIndex;
                    result.keypointCount = static_cast<int>(keypoints.size());
                    result.descriptorSize = descriptors.size();
                    result.detectionTimeMs = detectionTimeMs;
                    result.descriptorTimeMs = descriptorTimeMs;
                    combinationResults.push_back(result);

                    cout << "  [" << detectorType << "/" << descriptorType << "] "
                         << keypoints.size() << " keypoints, "
                         << descriptors.size() << " descriptors (det=" << detectionTimeMs << "ms, desc=" << descriptorTimeMs << "ms)" << endl;
                }
            }
            cout << "=== End of combination testing ===" << endl << endl;

            // Use first combination for the rest of the pipeline
            string detectorType = allDetectorTypes[0];
            string descriptorType = allDescriptorTypes[0];
            
            // Re-detect with first combination for pipeline
            detectKeypoints((dataBuffer.end() - 1)->keypoints, imgGray, detectorType, false);
            desc::descKeypoints((dataBuffer.end() - 1)->keypoints, imgGray, (dataBuffer.end() - 1)->descriptors, descriptorType, false);
        }
        else
        {
            // Single detector-descriptor as before
            vector<cv::KeyPoint> keypoints;
            string detectorType = "SHITOMASI";
            detectKeypoints(keypoints, imgGray, detectorType, false);

            // optional : limit number of keypoints (helpful for debugging and learning)
            bool bLimitKpts = false;
            if (bLimitKpts)
            {
                int maxKeypoints = 50;
                if (detectorType == "SHITOMASI")
                { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                    keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                }
                cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                cout << " NOTE: Keypoints have been limited!" << endl;
            }

            (dataBuffer.end() - 1)->keypoints = keypoints;
            
            string descriptorType = "ORB";
            desc::descKeypoints((dataBuffer.end() - 1)->keypoints, imgGray, (dataBuffer.end() - 1)->descriptors, descriptorType, false);
        }

        cout << "#5 : DETECT KEYPOINTS done" << endl;
        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorCategory = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN
            float ratioThreshold = 0.8f;           // Ratio threshold for Lowe's test

            match::matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                      (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                      matches, descriptorCategory, matcherType, selectorType, ratioThreshold);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            
            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;
            
            // Collect match counts for FP.1 analysis
            // Count matches between bounding boxes for CSV export
            std::map<std::pair<int, int>, int> matchCounts;
            for (const auto &match : matches)
            {
                const cv::KeyPoint &prevKp = (dataBuffer.end()-2)->keypoints[match.queryIdx];
                const cv::KeyPoint &currKp = (dataBuffer.end()-1)->keypoints[match.trainIdx];

                int prevBoxID = -1;
                for (const auto &prevBB : (dataBuffer.end()-2)->boundingBoxes)
                {
                    if (prevBB.roi.contains(prevKp.pt))
                    {
                        prevBoxID = prevBB.boxID;
                        break;
                    }
                }

                int currBoxID = -1;
                for (const auto &currBB : (dataBuffer.end()-1)->boundingBoxes)
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
            
            // Store match data for CSV export
            for (const auto &pair : bbBestMatches)
            {
                int prevBoxID = pair.first;
                int currBoxID = pair.second;
                int count = matchCounts[{prevBoxID, currBoxID}];
                
                // Get track ID from previous frame's trackIDMap
                int trackId = -1;
                if (trackIDMap.find(prevBoxID) != trackIDMap.end())
                {
                    trackId = trackIDMap.at(prevBoxID);
                }
                
                bbMatchResults.push_back({
                    static_cast<int>(imgStartIndex + imgIndex),
                    trackId,
                    prevBoxID,
                    currBoxID,
                    count
                });
            }
            
            // Optional: Print bounding box match information
            bool bPrintBBMatchInfo = true;
            if (bPrintBBMatchInfo)
            {
                printBBMatchInfo(bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1), matches);
            }

            cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // Cluster all keypoint matches to bounding boxes (handles overlapping boxes)
            // This ensures each match is assigned to at most one bounding box
            auto clusterStats = clusterAllKptMatchesWithROI((dataBuffer.end() - 1)->boundingBoxes,
                                                             (dataBuffer.end() - 2)->keypoints,
                                                             (dataBuffer.end() - 1)->keypoints,
                                                             (dataBuffer.end() - 1)->kptMatches);
            
            // Build a map for quick lookup of stats by boxID
            std::map<int, std::tuple<int, int, int>> boxStatsMap;
            for (const auto &stat : clusterStats)
            {
                int boxID = std::get<0>(stat);
                boxStatsMap[boxID] = stat;
            }

            // Assign track IDs and find the tracked preceding vehicle
            // trackedPrecedingVehicleTrackID >= 0 indicates the preceding vehicle is tracked
            // trackedPrecedingVehicleBoxID is cached for fast lookup
            assignTrackIDsAndFindPreceding(
                (dataBuffer.end() - 1)->boundingBoxes,
                (dataBuffer.end() - 1)->bbMatches,
                (dataBuffer.end() - 2)->boundingBoxes,
                (dataBuffer.end() - 1)->cameraImg,
                trackedPrecedingVehicleBoxID,
                trackedPrecedingVehicleTrackID,
                trackIDMap,
                trackAgeMap,
                nextTrackID);
            
            // Update camera image visualization with current tracking
            if (bVis)
            {
                visualizeBoundingBoxes(
                    (dataBuffer.end() - 1)->cameraImg,
                    (dataBuffer.end() - 1)->boundingBoxes,
                    class_names,
                    trackedPrecedingVehicleTrackID);
            }
            
            // Visualize 3D objects with tracking for subsequent frames
            if (bVis)
            {
                show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true, trackedPrecedingVehicleTrackID);
            }

            // Find the bounding boxes for the preceding vehicle using the cached boxID
            BoundingBox *prevBB = nullptr, *currBB = nullptr;
            int prevPrecedingVehicleBoxID = -1;  // Previous frame's boxID for the tracked vehicle
            
            // Only proceed if we have a tracked preceding vehicle (trackID >= 0)
            if (trackedPrecedingVehicleTrackID >= 0 && trackedPrecedingVehicleBoxID != -1)
            {
                // Find current BB using the cached boxID
                for (auto &bb : (dataBuffer.end() - 1)->boundingBoxes)
                {
                    if (bb.boxID == trackedPrecedingVehicleBoxID)
                    {
                        currBB = &bb;
                        break;
                    }
                }
                
                // Find matching previous BB
                for (const auto &match : (dataBuffer.end() - 1)->bbMatches)
                {
                    if (match.second == trackedPrecedingVehicleBoxID)
                    {
                        prevPrecedingVehicleBoxID = match.first;
                        for (auto &bb : (dataBuffer.end() - 2)->boundingBoxes)
                        {
                            if (bb.boxID == prevPrecedingVehicleBoxID)
                            {
                                prevBB = &bb;
                                break;
                            }
                        }
                        break;
                    }
                }
            }

            // compute TTC for the tracked preceding vehicle
            if (currBB != nullptr && prevBB != nullptr && 
                currBB->lidarPoints.size() > 0 && prevBB->lidarPoints.size() > 0)
                {
                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar; 
                    
                    if (bTestAllTTCMethods)
                    {
                        // Test all methods and record results
                        TTCResult result;
                        result.frameIndex = imgStartIndex + imgIndex;
                        result.trackId = trackedPrecedingVehicleTrackID;
                        result.prevBoxId = prevPrecedingVehicleBoxID;
                        result.currBoxId = trackedPrecedingVehicleBoxID;
                        
                        // Define methods to test
                        std::map<std::string, TTCMethod> methodNames = {
                            {"unfiltered", TTCMethod::UNFILTERED},
                            {"percentile_mean", TTCMethod::PERCENTILE_MEAN},
                            {"percentile_median", TTCMethod::PERCENTILE_MEDIAN}
                        };
                        
                        for (const auto& kv : methodNames)
                        {
                            double ttc;
                            computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttc, kv.second);
                            result.ttcValues[kv.first] = ttc;
                        }
                        
                        ttcResults.push_back(result);
                        ttcLidar = result.ttcValues["percentile_median"]; // Use median for display
                        
                    }
                    else
                    {
                        // Use selected method
                        computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar, ttcLidarMethod);
                    }
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    
                    // For FP.3: Get cluster statistics for this box
                    int matchesBefore = 0, matchesAfter = 0;
                    if (boxStatsMap.find(currBB->boxID) != boxStatsMap.end())
                    {
                        matchesBefore = std::get<1>(boxStatsMap[currBB->boxID]);
                        matchesAfter = std::get<2>(boxStatsMap[currBB->boxID]);
                    }
                    else
                    {
                        matchesAfter = static_cast<int>(currBB->kptMatches.size());
                    }
                    
                    // Record FP.3 statistics if enabled
                    if (bRecordKptStats && matchesAfter > 0)
                    {
                        KptMatchStats stats;
                        stats.frameIndex = imgStartIndex + imgIndex;
                        stats.trackId = currBB->trackID;  // Track ID for filtering
                        stats.boxId = currBB->boxID;
                        stats.matchesBefore = matchesBefore;
                        stats.matchesAfter = matchesAfter;
                        stats.outliersRemovedPct = (matchesBefore > 0) ? (100.0 * (matchesBefore - matchesAfter) / matchesBefore) : 0.0;
                        
                        // Calculate distance statistics
                        std::vector<double> distances;
                        for (const auto &match : currBB->kptMatches)
                        {
                            const cv::KeyPoint &prevKp = (dataBuffer.end() - 2)->keypoints[match.queryIdx];
                            const cv::KeyPoint &currKp = (dataBuffer.end() - 1)->keypoints[match.trainIdx];
                            double dist = cv::norm(prevKp.pt - currKp.pt);
                            distances.push_back(dist);
                        }
                        
                        if (!distances.empty())
                        {
                            stats.meanDistance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
                            std::sort(distances.begin(), distances.end());
                            stats.medianDistance = distances[distances.size() / 2];
                            double sq_sum = std::inner_product(distances.begin(), distances.end(), distances.begin(), 0.0);
                            stats.stddevDistance = std::sqrt(sq_sum / distances.size() - stats.meanDistance * stats.meanDistance);
                        }
                        else
                        {
                            stats.meanDistance = 0.0;
                            stats.medianDistance = 0.0;
                            stats.stddevDistance = 0.0;
                        }
                        
                        kptMatchStats.push_back(stats);
                    }
                    
                    // For FP.4: Get keypoint matches for the matched bounding box pair
                    // We need matches where prev keypoint is in prevBB AND curr keypoint is in currBB
                    std::vector<cv::DMatch> pairedMatches = getKptMatchesForBBPair(
                        *prevBB, *currBB,
                        (dataBuffer.end() - 2)->keypoints,
                        (dataBuffer.end() - 1)->keypoints,
                        (dataBuffer.end() - 1)->kptMatches);
                    
                    // Apply the same filtering as FP.3 for consistency
                    pairedMatches = filterMatchesByDistance(pairedMatches,
                        (dataBuffer.end() - 2)->keypoints,
                        (dataBuffer.end() - 1)->keypoints);
                    
                    // Compute distance ratios for debugging/analysis
                    std::vector<double> distRatios;
                    for (auto it1 = pairedMatches.begin(); it1 != pairedMatches.end() - 1; ++it1)
                    {
                        const cv::KeyPoint &kpOuterCurr = (dataBuffer.end() - 1)->keypoints.at(it1->trainIdx);
                        const cv::KeyPoint &kpOuterPrev = (dataBuffer.end() - 2)->keypoints.at(it1->queryIdx);
                        
                        for (auto it2 = it1 + 1; it2 != pairedMatches.end(); ++it2)
                        {
                            const cv::KeyPoint &kpInnerCurr = (dataBuffer.end() - 1)->keypoints.at(it2->trainIdx);
                            const cv::KeyPoint &kpInnerPrev = (dataBuffer.end() - 2)->keypoints.at(it2->queryIdx);
                            
                            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
                            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
                            
                            if (distPrev > std::numeric_limits<double>::epsilon() && 
                                distCurr >= cameraMinDist && distPrev >= cameraMinDist)
                            {
                                double distRatio = distCurr / distPrev;
                                distRatios.push_back(distRatio);
                            }
                        }
                    }
                    
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, pairedMatches, sensorFrameRate, cameraMinDist, ttcCamera);
                    
                    // Record FP.4 camera TTC results
                    if (bRecordCameraTTC)
                    {
                        CameraTTCResult cameraResult;
                        cameraResult.frameIndex = imgStartIndex + imgIndex;
                        cameraResult.trackId = trackedPrecedingVehicleTrackID;
                        cameraResult.prevBoxId = prevPrecedingVehicleBoxID;
                        cameraResult.currBoxId = trackedPrecedingVehicleBoxID;
                        cameraResult.ttcCamera = ttcCamera;
                        cameraTTCResults.push_back(cameraResult);
                    }
                    
                    // Record distance ratio statistics for analysis
                    if (bRecordCameraScaleStats && !distRatios.empty())
                    {
                        CameraTTCScaleStats stats;
                        stats.frameIndex = imgStartIndex + imgIndex;
                        stats.trackId = trackedPrecedingVehicleTrackID;
                        stats.numRatios = static_cast<int>(distRatios.size());
                        
                        if (!distRatios.empty())
                        {
                            stats.minRatio = *std::min_element(distRatios.begin(), distRatios.end());
                            stats.maxRatio = *std::max_element(distRatios.begin(), distRatios.end());
                            stats.medianRatio = computeMedian(distRatios);
                            stats.meanRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
                            double sq_sum = std::inner_product(distRatios.begin(), distRatios.end(), distRatios.begin(), 0.0);
                            stats.stddevRatio = std::sqrt(sq_sum / distRatios.size() - stats.meanRatio * stats.meanRatio);
                        }
                        else
                        {
                            stats.minRatio = 0.0;
                            stats.maxRatio = 0.0;
                            stats.medianRatio = 0.0;
                            stats.meanRatio = 0.0;
                            stats.stddevRatio = 0.0;
                        }
                        
                        // Without background filtering, all ratios are retained
                        stats.numFiltered = stats.numRatios;
                        stats.filteredMinRatio = stats.minRatio;
                        stats.filteredMaxRatio = stats.maxRatio;
                        stats.filteredMedianRatio = stats.medianRatio;
                        
                        cameraTTCScaleStats.push_back(stats);
                    }
                    //// EOF STUDENT ASSIGNMENT

                    bVis = false;  // Disable visualization for automated runs
                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        std::string str = cv::format("TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(10);  // Short wait for visualization
                    }

                } // eof TTC computation for tracked preceding vehicle            

        }

    } // eof loop over all images

    // Save combination test results to CSV for Python evaluation
    if (bTestAllCombinations && !combinationResults.empty())
    {
        std::string csvFilename = "./detector_descriptor_results.csv";
        std::ofstream csvFile(csvFilename);
        
        if (csvFile.is_open())
        {
            // Write CSV header
            csvFile << "ImageIndex,Detector,Descriptor,KeypointCount,DescriptorRows,DescriptorCols,DescriptorType,DetectionTimeMs,DescriptorTimeMs\n";
            
            // Write data rows
            for (const auto& result : combinationResults)
            {
                csvFile << result.imageIndex << ","
                        << result.detectorType << ","
                        << result.descriptorType << ","
                        << result.keypointCount << ","
                        << result.descriptorSize.height << ","
                        << result.descriptorSize.width << ","
                        << (result.descriptorSize.width * 8) << "," // descriptor type size in bits
                        << std::fixed << std::setprecision(2)
                        << result.detectionTimeMs << ","
                        << result.descriptorTimeMs << "\n";
            }
            
            csvFile.close();
            std::cout << "\nCombination test results saved to " << csvFilename << std::endl;
            std::cout << "Total combinations tested: " << combinationResults.size() << std::endl;
            std::cout << "(7 detectors x 5 descriptors x " << (imgEndIndex - imgStartIndex + 1) << " images = " 
                      << combinationResults.size() << " results)" << std::endl;
        }
        else
        {
            std::cerr << "Error: Could not open " << csvFilename << " for writing" << std::endl;
        }
    }

    // Save FP.1 bounding box match results to CSV for Python analysis
    if (!bbMatchResults.empty())
    {
        std::string bbCsvFilename = dataPath + "analysis/output/bb_matches.csv";
        std::ofstream bbCsvFile(bbCsvFilename);
        
        if (bbCsvFile.is_open())
        {
            // Write CSV header
            bbCsvFile << "frame_index,track_id,prev_box_id,curr_box_id,match_count\n";
            
            // Write data rows
            for (const auto& result : bbMatchResults)
            {
                bbCsvFile << result.frameIndex << ","
                          << result.trackId << ","
                          << result.prevBoxId << ","
                          << result.currBoxId << ","
                          << result.matchCount << "\n";
            }
            
            bbCsvFile.close();
            std::cout << "\nFP.1 Bounding box match results saved to " << bbCsvFilename << std::endl;
            std::cout << "Total matches recorded: " << bbMatchResults.size() << std::endl;
            std::cout << "Use: python analysis/fp1_analysis.py" << std::endl;
        }
        else
        {
            std::cerr << "Error: Could not open " << bbCsvFilename << " for writing" << std::endl;
        }
    }

    // Save FP.2 TTC comparison results to CSV for Python analysis
    if (bTestAllTTCMethods && !ttcResults.empty())
    {
        std::string csvFilename = dataPath + "analysis/output/ttc_lidar_comparison.csv";
        std::ofstream csvFile(csvFilename);
        
        if (csvFile.is_open())
        {
            // Write header
            csvFile << "frame_index,track_id,prev_box_id,curr_box_id,";
            std::vector<std::string> methodNames =
            {
                "unfiltered", "percentile_mean", "percentile_median"
            };
            for (const auto& name : methodNames)
            {
                csvFile << name << ",";
            }
            csvFile << "\n";
            
            // Write data
            for (const auto& result : ttcResults)
            {
                csvFile << result.frameIndex << ","
                        << result.trackId << ","
                        << result.prevBoxId << ","
                        << result.currBoxId << ",";
                
                for (const auto& name : methodNames)
                {
                    if (result.ttcValues.count(name))
                    {
                        double val = result.ttcValues.at(name);
                        if (std::isnan(val))
                        {
                            csvFile << "nan";
                        }
                        else
                        {
                            csvFile << val;
                        }
                    }
                    csvFile << ",";
                }
                csvFile << "\n";
            }
            
            csvFile.close();
            std::cout << "\nFP.2 TTC comparison results saved to " << csvFilename << std::endl;
            std::cout << "Total TTC records: " << ttcResults.size() << std::endl;
            std::cout << "Use: python analysis/fp2_analysis.py" << std::endl;
        }
        else
        {
            std::cerr << "Error: Could not open " << csvFilename << " for writing" << std::endl;
        }
    }

    // Save FP.3 keypoint match filtering results to CSV for Python analysis
    if (bRecordKptStats && !kptMatchStats.empty())
    {
        std::string csvFilename = dataPath + "analysis/output/kpt_matches_filtering.csv";
        std::ofstream csvFile(csvFilename);
        
        if (csvFile.is_open())
        {
            // Write CSV header
            csvFile << "frame_index,track_id,box_id,matches_before,matches_after,outliers_removed_pct,mean_distance,median_distance,stddev_distance\n";
            
            // Write data rows
            for (const auto& stats : kptMatchStats)
            {
                csvFile << stats.frameIndex << ","
                        << stats.trackId << ","
                        << stats.boxId << ","
                        << stats.matchesBefore << ","
                        << stats.matchesAfter << ","
                        << std::fixed << std::setprecision(2)
                        << stats.outliersRemovedPct << ","
                        << stats.meanDistance << ","
                        << stats.medianDistance << ","
                        << stats.stddevDistance << "\n";
            }
            
            csvFile.close();
            std::cout << "\nFP.3 Keypoint match filtering results saved to " << csvFilename << std::endl;
            std::cout << "Total records: " << kptMatchStats.size() << std::endl;
            std::cout << "Use: python analysis/fp3_analysis.py" << std::endl;
        }
        else
        {
            std::cerr << "Error: Could not open " << csvFilename << " for writing" << std::endl;
        }
    }

    // Save FP.4 camera-based TTC results to CSV for Python analysis
    if (bRecordCameraTTC && !cameraTTCResults.empty())
    {
        std::string csvFilename = dataPath + "analysis/output/ttc_camera.csv";
        std::ofstream csvFile(csvFilename);
        
        if (csvFile.is_open())
        {
            // Write CSV header
            csvFile << "frame_index,track_id,prev_box_id,curr_box_id,ttc_camera\n";
            
            // Write data rows
            for (const auto& result : cameraTTCResults)
            {
                csvFile << result.frameIndex << ","
                        << result.trackId << ","
                        << result.prevBoxId << ","
                        << result.currBoxId << ",";
                
                if (std::isnan(result.ttcCamera))
                {
                    csvFile << "nan";
                }
                else
                {
                    csvFile << result.ttcCamera;
                }
                csvFile << "\n";
            }
            
            csvFile.close();
            std::cout << "\nCamera TTC results saved to " << csvFilename << std::endl;
            std::cout << "Total TTC records: " << cameraTTCResults.size() << std::endl;
        }
        else
        {
            std::cerr << "Error: Could not open " << csvFilename << " for writing" << std::endl;
        }
    }

    // Save FP.4 distance ratio scale statistics to CSV for analysis
    if (bRecordCameraScaleStats && !cameraTTCScaleStats.empty())
    {
        std::string csvFilename = dataPath + "analysis/output/ttc_camera_scale_stats.csv";
        std::ofstream csvFile(csvFilename);
        
        if (csvFile.is_open())
        {
            // Write CSV header
            csvFile << "frame_index,track_id,num_ratios,min_ratio,max_ratio,median_ratio,mean_ratio,stddev_ratio,";
            csvFile << "num_filtered,filtered_min,filtered_max,filtered_median\n";
            
            // Write data rows
            for (const auto& stats : cameraTTCScaleStats)
            {
                csvFile << stats.frameIndex << ","
                        << stats.trackId << ","
                        << stats.numRatios << ","
                        << stats.minRatio << ","
                        << stats.maxRatio << ","
                        << stats.medianRatio << ","
                        << stats.meanRatio << ","
                        << stats.stddevRatio << ","
                        << stats.numFiltered << ","
                        << stats.filteredMinRatio << ","
                        << stats.filteredMaxRatio << ","
                        << stats.filteredMedianRatio << "\n";
            }
            
            csvFile.close();
            std::cout << "\nFP.4 Camera TTC scale statistics saved to " << csvFilename << std::endl;
            std::cout << "Total scale stat records: " << cameraTTCScaleStats.size() << std::endl;
        }
        else
        {
            std::cerr << "Error: Could not open " << csvFilename << " for writing" << std::endl;
        }
    }

    // Write the tracked preceding vehicle track ID to a file for Python analysis scripts
    // This ensures analysis scripts always use the correct track_id, regardless of YOLO detection order
    std::string trackIdFilename = dataPath + "analysis/output/tracked_preceding_vehicle_track_id.txt";
    std::ofstream trackIdFile(trackIdFilename);
    
    if (trackIdFile.is_open())
    {
        trackIdFile << trackedPrecedingVehicleTrackID << "\n";
        trackIdFile.close();
        std::cout << "\nTracked preceding vehicle track_id saved to: " << trackIdFilename << std::endl;
        std::cout << "Value: " << trackedPrecedingVehicleTrackID << std::endl;
    }
    else
    {
        std::cerr << "Error: Could not open " << trackIdFilename << " for writing" << std::endl;
    }

    return 0;
}
