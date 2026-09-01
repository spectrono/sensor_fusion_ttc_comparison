/**
 * @file lidarData.hpp
 * @brief LIDAR data loading and visualization functions
 */

#ifndef lidarData_hpp
#define lidarData_hpp

#include <stdio.h>
#include <fstream>
#include <string>

#include "dataStructures.h"

/**
 * @brief Crops LIDAR points to a specified 3D region
 * @param lidarPoints Vector of LIDAR points to crop
 * @param minX Minimum X coordinate
 * @param maxX Maximum X coordinate
 * @param maxY Maximum Y coordinate
 * @param minZ Minimum Z coordinate
 * @param maxZ Maximum Z coordinate
 * @param minR Minimum reflectivity
 */
void cropLidarPoints(std::vector<LidarPoint> &lidarPoints, float minX, float maxX, float maxY, float minZ, float maxZ, float minR);

/**
 * @brief Loads LIDAR points from a file
 * @param lidarPoints Output vector to store loaded points
 * @param filename Path to the LIDAR data file
 */
void loadLidarFromFile(std::vector<LidarPoint> &lidarPoints, std::string filename);

/**
 * @brief Visualizes LIDAR points in a top-down view
 * @param lidarPoints Vector of LIDAR points to visualize
 * @param worldSize World dimensions for scaling
 * @param imageSize Output image dimensions
 * @param bWait Whether to wait for key press
 */
void showLidarTopview(std::vector<LidarPoint> &lidarPoints, cv::Size worldSize, cv::Size imageSize, bool bWait=true);

/**
 * @brief Overlays LIDAR points on a camera image
 * @param img Camera image
 * @param lidarPoints LIDAR points to overlay
 * @param P_rect_xx Camera projection matrix (rectified)
 * @param R_rect_xx Rectification matrix
 * @param RT Rotation-translation matrix
 * @param extVisImg Optional external visualization image
 */
void showLidarImgOverlay(cv::Mat &img, std::vector<LidarPoint> &lidarPoints, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT, cv::Mat *extVisImg=nullptr);
#endif /* lidarData_hpp */
