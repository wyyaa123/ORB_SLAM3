#ifndef EDGE_EXTRACTER_H
#define EDGE_EXTRACTER_H

#include "Edge.h"
#include "EdgeCluster.h"
#include "BezierCurve.h"

class EdgeExtracter
{
public:
    EdgeExtracter(double _angleThreshold, double _threshold1, double _threshold2) : angleThreshold(_angleThreshold), threshold1(_threshold1), threshold2(_threshold2) {}

    std::vector<Edge> operator()(cv::Mat imGray, cv::Mat imSem = cv::Mat());

    std::vector<Edge> getEdges() const { return mvEdges; }

private:
    inline float calcAngleBias(float angle_1, float angle_2);

    void preprocessSem(const cv::Mat &imSem);

    void preprocessEdge();

    void cvt2OrderedEdges();

    void regionGrowthClusteringOCanny(float angle_Thres = 20.0f, const cv::Point &offset = cv::Point(0, 0));

    void getFineSampledPoints(int bias);

    void getBezierSampledPoints(int sampleSpacing);

    cv::Mat mSem;
    cv::Mat mMatGradMagnitude;
    cv::Mat mMatGradAngle;
    cv::Mat mCanny;
    std::map<int, int> mmIndexMap;
    cv::Mat mMatSearch;

    double angleThreshold;
    double threshold1;
    double threshold2;

    std::vector<Edge> mvEdges;
    std::vector<EdgeCluster> mvEdgeClusters;
    std::vector<BezierCurve> mvBezierCurves; // List of Bezier curves fitted to edges
};

#endif