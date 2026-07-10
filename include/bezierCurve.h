#ifndef BEZIERCURVE_H
#define BEZIERCURVE_H

#include <cfloat>
#include <vector>
#include "edge.h"
#include <Eigen/Core>

Eigen::Vector2f evaluateBezier(const std::vector<Eigen::Vector2f> &points, double t);
Eigen::Vector2f evaluate(const std::vector<Eigen::Vector2f> &controlPoints, double t);
double squaredDistance(const Eigen::Vector2f &a, const Eigen::Vector2f &b);
double squaredDistanceToBezierPoint(const std::vector<Eigen::Vector2f> &controlPoints, const Eigen::Vector2f &queryPoint, double t);
bool sameRoundedPixel(const Eigen::Vector2f &a, const Eigen::Vector2f &b);
double bernsteinBasis(int n, int i, double t);

struct ArcLengthTable
{
    std::vector<double> parameters;
    std::vector<double> lengths;

    double totalLength() const { return lengths.empty() ? 0.0 : lengths.back(); }
};

class BezierCurve
{
public:
    int edge_ID;
    int cls;
    std::vector<Eigen::Vector2f> controlPoints; // 每条边缘可能有多段Bezier曲线，每段曲线由一组控制点定义
    std::vector<Eigen::Vector2f> sampledPoints; // 从Bezier曲线上均匀采样得到的点列表
    std::vector<float> depths;
    std::vector<orderedEdgePoint> edge;
    BezierCurve() : edge_ID(-1), cls(-1) {}

    double approximateArcLength(std::size_t lookupSegmentCount = 200) const;
    void sampleByArcLengthSpacing(int spacing, std::size_t lookupSegmentCount = 200, bool removeDuplicatePixels = true);

    Eigen::Vector2f derivative(double t) const;
    Eigen::Vector2f secondDerivative(double t) const;
    Eigen::Vector2f closestPointOnCurve(const Eigen::Vector2f& p, int coarseSamples, int newtonIterations) const;
    bool estimateNormalFromSamples(const Eigen::Vector2f &point, Eigen::Vector2d &normal) const;

    void reserve(size_t controlPointCount) { controlPoints.reserve(controlPointCount); }
    void push_back(const Eigen::Vector2f &point) { controlPoints.push_back(point); }
};

class BezierCurveFitter
{
public:
    explicit BezierCurveFitter(double rho_p = 3.0, std::size_t minSplitPoints = 10);

    std::vector<BezierCurve> fitAdaptive(const std::vector<Eigen::Vector2f> &points) const;

    std::vector<double> computeResiduals(const std::vector<Eigen::Vector2f> &controlPoints, const std::vector<Eigen::Vector2f> &edge) const;
    BezierCurve fitWithEndPoints(const std::vector<Eigen::Vector2f> &edge, int order) const;

private:
    std::vector<double> chordLengthParameters(const std::vector<Eigen::Vector2f> &points) const;

    double rho_p_;
    std::size_t minSplitPoints_;
};

#endif // BEZIERCURVE_H
