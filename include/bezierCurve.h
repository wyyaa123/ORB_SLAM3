#ifndef BEZIERCURVE_H
#define BEZIERCURVE_H

#include <cfloat>
#include <vector>
#include "edge.h"
#include <spdlog/spdlog.h>

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
    std::vector<orderedEdgePoint> controlPoints; // 每条边缘可能有多段Bezier曲线，每段曲线由一组控制点定义
    std::vector<orderedEdgePoint> sampledPoints; // 从Bezier曲线上均匀采样得到的点列表
    BezierCurve() : edge_ID(-1), cls(-1) {}

    double approximateArcLength(std::size_t lookupSegmentCount = 200) const;
    void sampleByArcLengthSpacing(int spacing, std::size_t lookupSegmentCount = 200, bool removeDuplicatePixels = true);
    void reserve(size_t controlPointCount) { controlPoints.reserve(controlPointCount); }
    void push_back(const orderedEdgePoint &point) { controlPoints.push_back(point); }
};

class BezierCurveFitter
{
public:
    explicit BezierCurveFitter(double rho_p = 3.0, std::size_t minSplitPoints = 10);

    std::vector<BezierCurve> fitAdaptive(const std::vector<orderedEdgePoint> &points) const;

    std::vector<double> computeResiduals(const std::vector<orderedEdgePoint> &controlPoints, const std::vector<orderedEdgePoint> &edge) const;
    BezierCurve fitWithEndPoints(const std::vector<orderedEdgePoint> &edge, int order) const;

private:
    std::vector<double> chordLengthParameters(const std::vector<orderedEdgePoint> &points) const;

    double rho_p_;
    std::size_t minSplitPoints_;
};

#endif // BEZIERCURVE_H