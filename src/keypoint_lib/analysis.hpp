#ifndef analysis_hpp
#define analysis_hpp

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "detection.hpp"
#include "descriptors.hpp"
#include "matching.hpp"

namespace kp
{

// ============================================================================
// Detection Result with Metrics (for MP.7, MP.8, MP.9)
// ============================================================================

/// @brief Result structure containing keypoints and performance metrics
/// Used for performance analysis tasks (MP.7, MP.8, MP.9)
struct DetectionResult
{
    std::vector<cv::KeyPoint> keypoints;
    double detectionTimeMs = 0.0;       ///< Time taken for detection in milliseconds (MP.9)
    std::string detectorName;           ///< Name of the detector used
    size_t keypointCount = 0;           ///< Total number of keypoints detected (MP.7)
    std::vector<float> responses;        ///< Response values for all keypoints (MP.7)

    /// @brief Get min and max response values
    /// @return Pair of (min_response, max_response)
    std::pair<float, float> responseStats() const
    {
        if (responses.empty())
        {
            return {0.0f, 0.0f};
        }
        
        auto [min_it, max_it] = std::minmax_element(responses.begin(), responses.end());
        return {*min_it, *max_it};
    }

    /// @brief Filter keypoints by response threshold
    /// @param threshold Minimum response value to keep
    void filterByResponse(float threshold)
    {
        auto it = std::remove_if(
            keypoints.begin(),
            keypoints.end(),
            [threshold](const cv::KeyPoint& kp) { return kp.response < threshold; });

        keypoints.erase(it, keypoints.end());
        keypointCount = keypoints.size();
        
        // Also filter responses
        responses.erase(
            std::remove_if(
                responses.begin(),
                responses.end(),
                [threshold](float r) { return r < threshold; }),
            responses.end());
    }
};

/// @brief Vehicle ROI analysis result for MP.7
struct VehicleAnalysisResult
{
    std::string detectorName;
    int totalKeypoints = 0;
    int vehicleKeypoints = 0;
    std::map<int, int> neighborhoodSizeDist;  ///< Map of keypoint size -> count
    std::vector<float> vehicleResponses;
};

/// @brief Performance metrics for a detector (for MP.9)
struct DetectorPerformance
{
    std::string detectorName;
    double avgDetectionTimeMs = 0.0;
    double avgKeypointCount = 0.0;
    double avgMatchingScore = 0.0;
};

// ============================================================================
// Analysis Functions
// ============================================================================

/// @brief Detect keypoints with metrics collection for performance analysis
/// Used for MP.7, MP.8, MP.9
/// @param img Input grayscale image
/// @param detectorType String specifying the detector type
/// @param visualize If true, display detection results
/// @return DetectionResult containing keypoints and metrics
inline DetectionResult detectKeypointsWithMetrics(
    const cv::Mat& img,
    const std::string& detectorType,
    bool visualize = false)
{
    DetectionResult result;
    result.detectorName = detectorType;

    double t = static_cast<double>(cv::getTickCount());

    // Call the appropriate detector from detection.hpp
    detectKeypoints(result.keypoints, img, detectorType, visualize);

    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    result.detectionTimeMs = 1000.0 * t;
    result.keypointCount = result.keypoints.size();

    // Collect responses for distribution analysis (MP.7)
    for (const auto& kp : result.keypoints)
    {
        result.responses.push_back(kp.response);
    }

    return result;
}

/// @brief Batch detection across multiple images and detector types
/// @param images Vector of input images
/// @param detectorTypes Vector of detector type strings
/// @param visualize If true, display detection results
/// @return Vector of DetectionResult for each image-detector combination
inline std::vector<DetectionResult> detectAllWithMetrics(
    const std::vector<cv::Mat>& images,
    const std::vector<std::string>& detectorTypes,
    bool visualize = false)
{
    std::vector<DetectionResult> allResults;
    allResults.reserve(images.size() * detectorTypes.size());

    for (const auto& img : images)
    {
        for (const auto& detectorType : detectorTypes)
        {
            DetectionResult result = detectKeypointsWithMetrics(img, detectorType, visualize);
            allResults.push_back(result);
        }
    }

    return allResults;
}

/// @brief Get all available detector type strings
/// @return Const reference to vector of all detector type strings
inline const std::vector<std::string>& getAllDetectorTypes()
{
    static const std::vector<std::string> types =
    {
        "HARRIS",
        "FAST",
        "BRISK",
        "ORB",
        "AKAZE",
        "SIFT",
        "SHITOMASI"
    };
 
    return types;
}

/// @brief Analyze keypoints on a specific vehicle ROI for MP.7
/// @param keypoints Vector of detected keypoints
/// @param vehicleRect ROI rectangle defining the vehicle area
/// @param detectorName Name of the detector used
/// @return VehicleAnalysisResult with counts and distribution data
inline VehicleAnalysisResult analyzeKeypointsOnVehicle(
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Rect& vehicleRect,
    const std::string& detectorName)
{

    VehicleAnalysisResult result;
    result.detectorName = detectorName;
    result.totalKeypoints = static_cast<int>(keypoints.size());

    for (const auto& kp : keypoints)
    {
        if (vehicleRect.contains(kp.pt))
        {
            result.vehicleKeypoints++;
            result.vehicleResponses.push_back(kp.response);
            // Bin by keypoint size (neighborhood size distribution)
            int sizeBin = static_cast<int>(kp.size) / 2 * 2; // Round to even
            result.neighborhoodSizeDist[sizeBin]++;
        }
    }

    return result;
}

/// @brief Compute average performance metrics from multiple detection results
/// @param results Vector of DetectionResult from multiple runs
/// @return Vector of DetectorPerformance sorted by detection time
inline std::vector<DetectorPerformance> computePerformanceMetrics(
    const std::vector<DetectionResult>& results)
{
    std::unordered_map<std::string, DetectorPerformance> perfMap;
    std::unordered_map<std::string, int> countMap;

    for (const auto& result : results)
    {
        auto& perf = perfMap[result.detectorName];
        perf.detectorName = result.detectorName;
        perf.avgDetectionTimeMs += result.detectionTimeMs;
        perf.avgKeypointCount += static_cast<double>(result.keypointCount);
        countMap[result.detectorName]++;
    }

    // Normalize by count
    for (auto& [name, perf] : perfMap)
    {
        int count = countMap[name];
        perf.avgDetectionTimeMs /= count;
        perf.avgKeypointCount /= count;
    }

    // Convert to vector
    std::vector<DetectorPerformance> sortedPerf;
    sortedPerf.reserve(perfMap.size());
    for (auto& [name, perf] : perfMap)
    {
        sortedPerf.push_back(perf);
    }

    // Sort by detection time (ascending = faster is better)
    std::sort(sortedPerf.begin(), sortedPerf.end(),
        [](const DetectorPerformance& a, const DetectorPerformance& b)
        {
            return a.avgDetectionTimeMs < b.avgDetectionTimeMs;
        });

    return sortedPerf;
}

// ============================================================================
// MP.7 Specific Analysis Functions
// ============================================================================

/// @brief Result for MP.7 - keypoint count on vehicle for a specific detector and image
struct MP7Result
{
    std::string detectorName;
    int imageIndex = 0;
    int totalKeypoints = 0;
    int vehicleKeypoints = 0;
    std::map<int, int> neighborhoodSizeDist;  ///< Distribution of keypoint sizes on vehicle
    std::vector<float> vehicleResponses;
    double detectionTimeMs = 0.0;

    /// @brief Print the result in a formatted way
    void print() const
    {
        std::cout << "MP.7 Result - Image " << imageIndex << ", Detector: " << detectorName << std::endl;
        std::cout << "  Total Keypoints: " << totalKeypoints << std::endl;
        std::cout << "  Vehicle Keypoints: " << vehicleKeypoints << std::endl;
        std::cout << "  Detection Time: " << detectionTimeMs << " ms" << std::endl;
        
        if (!neighborhoodSizeDist.empty())
        {
            std::cout << "  Neighborhood Size Distribution:" << std::endl;
            for (const auto& [size, count] : neighborhoodSizeDist)
            {
                std::cout << "    Size " << size << ": " << count << " keypoints" << std::endl;
            }
        }
        std::cout << std::endl;
    }
};

/// @brief Perform MP.7 analysis: Count keypoints on vehicle for all images and detectors
/// @param imgBasePath Base path to images
/// @param imgPrefix Image filename prefix
/// @param imgFileType Image file extension
/// @param imgStartIndex First image index
/// @param imgEndIndex Last image index
/// @param imgFillWidth Number of digits in filename
/// @param vehicleRect ROI for the preceding vehicle
/// @param bVis Whether to visualize results
/// @return Vector of MP7Result for all combinations
inline std::vector<MP7Result> runMP7Analysis(
    const std::string& imgBasePath,
    const std::string& imgPrefix,
    const std::string& imgFileType,
    int imgStartIndex,
    int imgEndIndex,
    int imgFillWidth,
    const cv::Rect& vehicleRect = cv::Rect(535, 180, 180, 150),
    bool bVis = false)
{
    std::vector<MP7Result> results;
    const auto& detectorTypes = getAllDetectorTypes();
    
    std::cout << "=== MP.7 Performance Evaluation 1 ===" << std::endl;
    std::cout << "Counting keypoints on preceding vehicle for all detectors and images" << std::endl;
    std::cout << "Vehicle ROI: x=" << vehicleRect.x << ", y=" << vehicleRect.y << 
              ", width=" << vehicleRect.width << ", height=" << vehicleRect.height << std::endl;
    std::cout << "Processing " << detectorTypes.size() << " detectors across " 
              << (imgEndIndex - imgStartIndex + 1) << " images..." << std::endl << std::endl;

    for (int imgIndex = imgStartIndex; imgIndex <= imgEndIndex; imgIndex++)
    {
        // Load image
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgIndex;
        std::string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;
        
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        if (img.empty())
        {
            std::cerr << "Error: Could not load image " << imgFullFilename << std::endl;
            continue;
        }
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        for (const auto& detectorType : detectorTypes)
        {
            MP7Result result;
            result.detectorName = detectorType;
            result.imageIndex = imgIndex;

            // Detect keypoints with metrics
            double t = static_cast<double>(cv::getTickCount());
            std::vector<cv::KeyPoint> allKeypoints;
            detectKeypoints(allKeypoints, imgGray, detectorType, bVis);
            t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
            result.detectionTimeMs = 1000.0 * t;
            result.totalKeypoints = static_cast<int>(allKeypoints.size());

            // Analyze keypoints on vehicle
            for (const auto& kp : allKeypoints)
            {
                if (vehicleRect.contains(kp.pt))
                {
                    result.vehicleKeypoints++;
                    result.vehicleResponses.push_back(kp.response);
                    // Bin by keypoint size for neighborhood size distribution
                    int sizeBin = static_cast<int>(std::round(kp.size));
                    result.neighborhoodSizeDist[sizeBin]++;
                }
            }

            results.push_back(result);
            
            // Print progress
            std::cout << "Image " << imgIndex << ", " << detectorType << ": " 
                      << result.totalKeypoints << " total, " << result.vehicleKeypoints 
                      << " on vehicle (" << result.detectionTimeMs << " ms)" << std::endl;
        }
        std::cout << std::endl;
    }

    return results;
}

/// @brief Generate a Markdown table string from MP.7 results
/// @param results Vector of MP7Result from runMP7Analysis
/// @return String containing Markdown table
inline std::string generateMP7MarkdownTable(const std::vector<MP7Result>& results)
{
    if (results.empty()) return "";
    
    // Find unique image indices and detector names
    std::set<int> uniqueImages;
    std::set<std::string> uniqueDetectors;
    
    for (const auto& result : results)
    {
        uniqueImages.insert(result.imageIndex);
        uniqueDetectors.insert(result.detectorName);
    }
    
    // Sort for consistent output
    std::vector<int> sortedImages(uniqueImages.begin(), uniqueImages.end());
    std::vector<std::string> sortedDetectors(uniqueDetectors.begin(), uniqueDetectors.end());
    
    std::sort(sortedImages.begin(), sortedImages.end());
    std::sort(sortedDetectors.begin(), sortedDetectors.end());
    
    // Build table header
    std::ostringstream table;
    table << "| Image | ";
    
    for (const auto& detector : sortedDetectors)
    {
        table << detector << " | ";
    }
    table << "\n";
    
    // Build separator row
    table << "|------|";
    for (size_t i = 0; i < sortedDetectors.size(); i++)
    {
        table << "------|";
    }
    table << "\n";
    
    // Build data rows - one row per image
    for (int imgIndex : sortedImages)
    {
        table << "| " << imgIndex << " | ";
        
        for (const auto& detector : sortedDetectors)
        {
            // Find the result for this image and detector
            int count = 0;
            for (const auto& result : results)
            {
                if (result.imageIndex == imgIndex && result.detectorName == detector)
                {
                    count = result.vehicleKeypoints;
                    break;
                }
            }
            table << count << " | ";
        }
        table << "\n";
    }
    
    return table.str();
}

/// @brief Generate neighborhood size distribution table for all detectors
/// @param results Vector of MP7Result from runMP7Analysis
/// @return String containing Markdown table for neighborhood size distributions
inline std::string generateNeighborhoodSizeDistributionTable(const std::vector<MP7Result>& results)
{
    if (results.empty()) return "";
    
    // Aggregate neighborhood size distributions by detector
    std::unordered_map<std::string, std::map<int, int>> detectorDistributions;
    
    for (const auto& result : results)
    {
        auto& dist = detectorDistributions[result.detectorName];
        for (const auto& [size, count] : result.neighborhoodSizeDist)
        {
            dist[size] += count;
        }
    }
    
    // Build table
    std::ostringstream table;
    table << "| Detector | Neighborhood Size Distribution |\n";
    table << "|----------|-------------------------------|\n";
    
    for (const auto& [detector, dist] : detectorDistributions)
    {
        table << "| " << detector << " | ";
        
        // Sort sizes and format as size:count pairs
        std::vector<std::pair<int, int>> sortedSizes(dist.begin(), dist.end());
        std::sort(sortedSizes.begin(), sortedSizes.end());
        
        for (size_t i = 0; i < sortedSizes.size(); i++)
        {
            if (i > 0) table << ", ";
            table << sortedSizes[i].first << ":" << sortedSizes[i].second;
        }
        table << " |\n";
    }
    
    return table.str();
}

/// @brief Generate summary statistics table
/// @param results Vector of MP7Result from runMP7Analysis
/// @return String containing summary statistics table
inline std::string generateMP7SummaryTable(const std::vector<MP7Result>& results)
{
    if (results.empty()) return "";
    
    // Aggregate statistics by detector
    std::unordered_map<std::string, std::vector<int>> detectorStats;
    
    for (const auto& result : results)
    {
        detectorStats[result.detectorName].push_back(result.vehicleKeypoints);
    }
    
    // Build table
    std::ostringstream table;
    table << "| Detector | Min | Max | Mean | Total |\n";
    table << "|----------|-----|-----|------|-------|\n";
    
    // Sort detectors alphabetically
    std::vector<std::string> sortedDetectors;
    for (const auto& [detector, counts] : detectorStats)
    {
        sortedDetectors.push_back(detector);
    }
    std::sort(sortedDetectors.begin(), sortedDetectors.end());
    
    for (const auto& detector : sortedDetectors)
    {
        const auto& counts = detectorStats[detector];
        
        if (!counts.empty())
        {
            int minCount = *std::min_element(counts.begin(), counts.end());
            int maxCount = *std::max_element(counts.begin(), counts.end());
            double meanCount = std::accumulate(counts.begin(), counts.end(), 0.0) / counts.size();
            int totalCount = std::accumulate(counts.begin(), counts.end(), 0);
            
            table << "| " << detector << " | " 
                  << minCount << " | " 
                  << maxCount << " | " 
                  << std::fixed << std::setprecision(1) << meanCount << " | " 
                  << totalCount << " |\n";
        }
    }
    
    return table.str();
}

// ============================================================================
// MP.8 Specific Analysis Functions
// ============================================================================

/// @brief Result for MP.8 - match count for a specific detector-descriptor-image pair combination
struct MP8Result
{
    int imagePairIndex = 0;  ///< Index of the image pair (0 = images 0-1, 1 = images 1-2, etc.)
    std::string detectorName;
    std::string descriptorName;
    int keypointsPrevCount = 0;   ///< Number of keypoints in previous image
    int keypointsCurrCount = 0;   ///< Number of keypoints in current image
    int matchCount = 0;            ///< Number of matched keypoints after ratio test
    double detectionTimeMs = 0.0;
    double descriptorTimeMs = 0.0;
    double matchingTimeMs = 0.0;

    /// @brief Print the result in a formatted way
    void print() const
    {
        std::cout << "MP.8 Result - Pair " << imagePairIndex << ", " << detectorName << "/" << descriptorName
                  << ": " << keypointsPrevCount << "+" << keypointsCurrCount << " kpts, " << matchCount << " matches"
                  << " (det=" << detectionTimeMs << "ms, desc=" << descriptorTimeMs 
                  << "ms, match=" << matchingTimeMs << "ms)" << std::endl;
    }
};

/// @brief Perform MP.8 analysis: Count matching keypoints for all detector-descriptor combinations
/// @param imgBasePath Base path to images
/// @param imgPrefix Image filename prefix
/// @param imgFileType Image file extension
/// @param imgStartIndex First image index
/// @param imgEndIndex Last image index
/// @param imgFillWidth Number of digits in filename
/// @param bVis Whether to visualize results
/// @return Vector of MP8Result for all combinations
inline std::vector<MP8Result> runMP8Analysis(
    const std::string& imgBasePath,
    const std::string& imgPrefix,
    const std::string& imgFileType,
    int imgStartIndex,
    int imgEndIndex,
    int imgFillWidth,
    bool bVis = false)
{
    std::vector<MP8Result> results;
    const auto& detectorTypes = getAllDetectorTypes();
    const auto& descriptorTypes = desc::getAllDescriptorTypes();

    std::cout << "=== MP.8 Performance Evaluation 2 ===" << std::endl;
    std::cout << "Counting matching keypoints for all detector-descriptor combinations" << std::endl;
    std::cout << "Using Brute-Force matching with descriptor distance ratio of 0.8" << std::endl;
    std::cout << "Processing " << detectorTypes.size() << " detectors x " << descriptorTypes.size() << " descriptors" << std::endl;
    std::cout << "Across " << (imgEndIndex - imgStartIndex) << " image pairs..." << std::endl << std::endl;

    // Process each consecutive image pair
    for (int prevIndex = imgStartIndex; prevIndex < imgEndIndex; prevIndex++)
    {
        int currIndex = prevIndex + 1;
        int pairIndex = prevIndex;

        // Load previous and current images
        std::ostringstream prevNumber, currNumber;
        prevNumber << std::setfill('0') << std::setw(imgFillWidth) << prevIndex;
        currNumber << std::setfill('0') << std::setw(imgFillWidth) << currIndex;

        std::string prevFilename = imgBasePath + imgPrefix + prevNumber.str() + imgFileType;
        std::string currFilename = imgBasePath + imgPrefix + currNumber.str() + imgFileType;

        cv::Mat imgPrev, imgPrevGray, imgCurr, imgCurrGray;
        imgPrev = cv::imread(prevFilename);
        imgCurr = cv::imread(currFilename);

        if (imgPrev.empty() || imgCurr.empty())
        {
            std::cerr << "Error: Could not load images " << prevFilename << " or " << currFilename << std::endl;
            continue;
        }

        cv::cvtColor(imgPrev, imgPrevGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgCurr, imgCurrGray, cv::COLOR_BGR2GRAY);

        // Process each detector-descriptor combination
        for (const auto& detectorType : detectorTypes)
        {
            for (const auto& descriptorType : descriptorTypes)
            {
                MP8Result result;
                result.imagePairIndex = pairIndex;
                result.detectorName = detectorType;
                result.descriptorName = descriptorType;

                // Detect keypoints on previous and current images
                std::vector<cv::KeyPoint> kptsPrev, kptsCurr;
                
                double t = static_cast<double>(cv::getTickCount());
                detectKeypoints(kptsPrev, imgPrevGray, detectorType, bVis);
                t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
                result.detectionTimeMs = 1000.0 * t;
                result.keypointsPrevCount = static_cast<int>(kptsPrev.size());

                t = static_cast<double>(cv::getTickCount());
                detectKeypoints(kptsCurr, imgCurrGray, detectorType, bVis);
                t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
                result.detectionTimeMs += 1000.0 * t; // Accumulate both images
                result.keypointsCurrCount = static_cast<int>(kptsCurr.size());

                // Skip if no keypoints in either image
                if (kptsPrev.empty() || kptsCurr.empty())
                {
                    result.matchCount = 0;
                    result.descriptorTimeMs = 0.0;
                    result.matchingTimeMs = 0.0;
                    results.push_back(result);
                    result.print();
                    continue;
                }

                // Extract descriptors on previous image
                cv::Mat descPrev;
                t = static_cast<double>(cv::getTickCount());
                try {
                    desc::descKeypoints(kptsPrev, imgPrevGray, descPrev, descriptorType, bVis);
                } catch (const cv::Exception& e) {
                    std::cerr << "Descriptor extraction failed for " << detectorType << "/" << descriptorType
                              << " on prev image: " << e.what() << std::endl;
                    result.matchCount = 0;
                    result.descriptorTimeMs = 0.0;
                    result.matchingTimeMs = 0.0;
                    results.push_back(result);
                    result.print();
                    continue;
                }
                t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
                result.descriptorTimeMs = 1000.0 * t;

                // Skip if descriptor extraction failed (empty descriptors)
                if (descPrev.empty())
                {
                    result.matchCount = 0;
                    result.descriptorTimeMs = 0.0;
                    result.matchingTimeMs = 0.0;
                    results.push_back(result);
                    result.print();
                    continue;
                }

                // Extract descriptors on current image
                cv::Mat descCurr;
                t = static_cast<double>(cv::getTickCount());
                try {
                    desc::descKeypoints(kptsCurr, imgCurrGray, descCurr, descriptorType, bVis);
                } catch (const cv::Exception& e) {
                    std::cerr << "Descriptor extraction failed for " << detectorType << "/" << descriptorType
                              << " on curr image: " << e.what() << std::endl;
                    result.matchCount = 0;
                    result.descriptorTimeMs = 0.0;
                    result.matchingTimeMs = 0.0;
                    results.push_back(result);
                    result.print();
                    continue;
                }
                t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
                result.descriptorTimeMs += 1000.0 * t; // Accumulate both images

                // Skip if descriptor extraction failed (empty descriptors)
                if (descCurr.empty())
                {
                    result.matchCount = 0;
                    result.descriptorTimeMs = 0.0;
                    result.matchingTimeMs = 0.0;
                    results.push_back(result);
                    result.print();
                    continue;
                }

                // Match descriptors using Brute-Force with ratio test at 0.8
                std::vector<cv::DMatch> matches;
                t = static_cast<double>(cv::getTickCount());

                // Determine descriptor category for matching
                std::string descCategory = (descriptorType == "SIFT") ? "DES_HOG" : "DES_BINARY";

                match::matchDescriptors(kptsPrev, kptsCurr, descPrev, descCurr, matches,
                                       descCategory, "MAT_BF", "SEL_KNN", 0.8f);

                t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
                result.matchingTimeMs = 1000.0 * t;
                result.matchCount = static_cast<int>(matches.size());

                results.push_back(result);
                result.print();
            }
        }
        std::cout << std::endl;
    }

    return results;
}

/// @brief Save MP.8 results to CSV file
/// @param results Vector of MP8Result from runMP8Analysis
/// @param outputDir Directory to save results
/// @param filename Base filename for output (default: MP8_Results)
inline void saveMP8Results(const std::vector<MP8Result>& results, 
                           const std::string& outputDir = "./",
                           const std::string& filename = "MP8_Results")
{
    std::ostringstream csvPath;
    csvPath << outputDir << filename << ".csv";

    std::ofstream csvFile(csvPath.str());
    if (!csvFile.is_open())
    {
        std::cerr << "Error: Could not open " << csvPath.str() << std::endl;
        return;
    }

    // Write CSV header
    csvFile << "ImagePair,Detector,Descriptor,KeypointsPrev,KeypointsCurr,MatchCount,DetectionTimeMs,DescriptorTimeMs,MatchingTimeMs\n";

    // Write data rows
    for (const auto& result : results)
    {
        csvFile << result.imagePairIndex << ","
                << result.detectorName << ","
                << result.descriptorName << ","
                << result.keypointsPrevCount << ","
                << result.keypointsCurrCount << ","
                << result.matchCount << ","
                << std::fixed << std::setprecision(2)
                << result.detectionTimeMs/2 << ","  // Average per image
                << result.descriptorTimeMs/2 << ","   // Average per image
                << result.matchingTimeMs << "\n";
    }

    csvFile.close();
    std::cout << "MP.8 results saved to " << csvPath.str() << std::endl;

    // Save heatmap data (aggregated by detector-descriptor)
    // Aggregate match counts
    std::unordered_map<std::string, std::unordered_map<std::string, int>> detectorDescMatches;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> detectorDescDetectionTime;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> detectorDescDescriptorTime;
    std::unordered_map<std::string, std::unordered_map<std::string, double>> detectorDescMatchingTime;

    int numPairs = 0;
    for (const auto& result : results)
    {
        detectorDescMatches[result.detectorName][result.descriptorName] += result.matchCount;
        detectorDescDetectionTime[result.detectorName][result.descriptorName] += result.detectionTimeMs / 2;
        detectorDescDescriptorTime[result.detectorName][result.descriptorName] += result.descriptorTimeMs / 2;
        detectorDescMatchingTime[result.detectorName][result.descriptorName] += result.matchingTimeMs;
        numPairs = std::max(numPairs, result.imagePairIndex + 1);
    }

    // Save heatmap CSV
    std::ostringstream heatmapPath;
    heatmapPath << outputDir << "MP8_Heatmap.csv";

    std::ofstream heatmapFile(heatmapPath.str());
    if (heatmapFile.is_open())
    {
        heatmapFile << "Detector,Descriptor,TotalMatchCount\n";

        // Get sorted detector and descriptor lists
        const auto& detectorTypes = getAllDetectorTypes();
        const auto& descriptorTypes = desc::getAllDescriptorTypes();

        for (const auto& detector : detectorTypes)
        {
            for (const auto& descriptor : descriptorTypes)
            {
                int totalMatches = detectorDescMatches[detector][descriptor];
                heatmapFile << detector << "," << descriptor << "," << totalMatches << "\n";
            }
        }

        heatmapFile.close();
        std::cout << "MP.8 heatmap data saved to " << heatmapPath.str() << std::endl;
    }
}

} // namespace kp

#endif // analysis_hpp
