#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "edge.h"

#define areaThd 100
#define THDBOUND 20

class Patch
{
public:
    // 2d
    int cls;
    int area;
    double top, left, width, height;
    cv::Vec3b color;
    cv::Mat mask;
    cv::Rect roi;
    // 3d
    Eigen::Vector3f centroid;
    Eigen::Matrix3f cov;
    Eigen::Matrix3f ev; // eigen vectors for local patch axes

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points;
};
