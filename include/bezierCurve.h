#ifndef BEZIERCURVE_H
#define BEZIERCURVE_H

#include <vector>
#include "edge.h"
#include <spdlog/spdlog.h>

class BezierCurveFitter
{
public:
    explicit BezierCurveFitter(double rho_p = 3.0, std::size_t minSplitPoints = 10);

    std::vector<std::vector<orderedEdgePoint>> fitAdaptive(const std::vector<orderedEdgePoint> &points) const;

    std::vector<double> computeResiduals(const std::vector<orderedEdgePoint> &controlPoints, const Edge &edge) const;
    orderedEdgePoint evaluate(const std::vector<orderedEdgePoint> &controlPoints, double t) const;
    std::vector<orderedEdgePoint> fitWithEndPoints(const Edge &edge, int order) const;
    double bernsteinBasis(int n, int i, double t) const;

private:
    std::vector<double> chordLengthParameters(const std::vector<orderedEdgePoint> &points) const;
    double minDistanceToBezier(const std::vector<orderedEdgePoint> &controlPoints, const orderedEdgePoint &point) const;

    double rho_p_;
    std::size_t minSplitPoints_;
};

#endif // BEZIERCURVE_H