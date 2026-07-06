#include "bezierCurve.h"

#include <algorithm>
#include <cmath>
#include <limits>

Eigen::Vector2f evaluateBezier(const std::vector<Eigen::Vector2f> &points, double t)
{
    if (points.empty())
        return Eigen::Vector2f::Zero();

    if (points.size() == 1)
        return points[0];

    t = std::max(0.0, std::min(1.0, t));

    std::vector<Eigen::Vector2f> tmp = points;
    const int n = static_cast<int>(tmp.size());

    for (int r = 1; r < n; ++r)
    {
        for (int i = 0; i < n - r; ++i)
        {
            tmp[i] = (1.0 - t) * tmp[i] + (t) * tmp[i + 1];
        }
    }

    return tmp[0];
}

Eigen::Vector2f evaluate(const std::vector<Eigen::Vector2f> &controlPoints, double t)
{
    t = std::max(0.0, std::min(1.0, t));
    if (controlPoints.empty())
        return Eigen::Vector2f(0.0, 0.0);

    if (controlPoints.size() == 1)
        return controlPoints.front();

    std::vector<Eigen::Vector2f> points = controlPoints;
    for (std::size_t count = points.size(); count > 1; --count)
    {
        for (std::size_t i = 0; i + 1 < count; ++i)
        {
            points[i].x() = (1.0 - t) * points[i].x() + t * points[i + 1].x();
            points[i].y() = (1.0 - t) * points[i].y() + t * points[i + 1].y();
        }
    }
    return points.front();
}

double squaredDistance(const Eigen::Vector2f &a, const Eigen::Vector2f &b)
{
    const double dx = a.x() - b.x();
    const double dy = a.y() - b.y();
    return dx * dx + dy * dy;
}

double squaredDistanceToBezierPoint(const std::vector<Eigen::Vector2f> &controlPoints, const Eigen::Vector2f &queryPoint, double t)
{
    const Eigen::Vector2f curvePoint = evaluate(controlPoints, t);
    return squaredDistance(curvePoint, queryPoint);
}

bool sameRoundedPixel(const Eigen::Vector2f &a, const Eigen::Vector2f &b)
{
    return cvRound(a.x()) == cvRound(b.x()) && cvRound(a.y()) == cvRound(b.y());
}

double bernsteinBasis(int n, int i, double t)
{
    if (i < 0 || i > n)
        return 0.0;

    t = std::max(0.0, std::min(1.0, t));
    const double u = 1.0 - t;
    double binomialCoeff = 1.0;
    for (int j = 0; j < i; ++j)
        binomialCoeff *= (n - j) / static_cast<double>(j + 1);

    return binomialCoeff * std::pow(u, n - i) * std::pow(t, i);
}

static inline ArcLengthTable buildArcLengthTable(const BezierCurve &curve, std::size_t segmentCount)
{
    segmentCount = std::max<std::size_t>(segmentCount, 1);

    ArcLengthTable table;
    table.parameters.resize(segmentCount + 1, 0.0);
    table.lengths.resize(segmentCount + 1, 0.0);

    Eigen::Vector2f prevPoint = evaluate(curve.controlPoints, 0.0);
    for (std::size_t i = 1; i <= segmentCount; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(segmentCount);
        const Eigen::Vector2f point = evaluate(curve.controlPoints, t);
        table.parameters[i] = t;
        table.lengths[i] = table.lengths[i - 1] + std::sqrt(squaredDistance(prevPoint, point));
        prevPoint = point;
    }

    return table;
}

static inline double parameterAtLength(const ArcLengthTable &table, double targetLength)
{
    const auto upper = std::lower_bound(table.lengths.begin(), table.lengths.end(), targetLength);
    const std::size_t upperIndex = static_cast<std::size_t>(std::distance(table.lengths.begin(), upper));

    if (upperIndex == 0)
        return 0.0;
    if (upperIndex >= table.lengths.size())
        return 1.0;

    const double length0 = table.lengths[upperIndex - 1];
    const double length1 = table.lengths[upperIndex];
    const double denom = length1 - length0;
    if (denom <= DBL_EPSILON)
        return table.parameters[upperIndex - 1];

    const double ratio = (targetLength - length0) / denom;
    const double t0 = table.parameters[upperIndex - 1];
    const double t1 = table.parameters[upperIndex];
    return t0 + ratio * (t1 - t0);
}

static inline void appendSample(BezierCurve &curve, Eigen::Vector2f point, bool removeDuplicatePixels)
{
    if (removeDuplicatePixels && !curve.sampledPoints.empty() && sameRoundedPixel(curve.sampledPoints.back(), point))
        return;

    curve.sampledPoints.push_back(point);
}

double BezierCurve::approximateArcLength(std::size_t lookupSegmentCount) const
{
    if (controlPoints.size() < 2)
        return 0.0;
    return buildArcLengthTable(*this, lookupSegmentCount).totalLength();
}

void BezierCurve::sampleByArcLengthSpacing(int spacing, std::size_t lookupSegmentCount, bool removeDuplicatePixels)
{
    sampledPoints.clear();
    if (controlPoints.empty())
        return;

    if (spacing <= DBL_EPSILON)
        spacing = 1.0;

    const double length = approximateArcLength(lookupSegmentCount);
    spdlog::debug("Approximated arc length: {:.5f}", length);
    const std::size_t segmentCount = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(length / spacing)));
    // sampleByArcLength(segmentCount, lookupSegmentCount, removeDuplicatePixels);

    if (controlPoints.size() == 1 || segmentCount == 0)
    {
        appendSample(*this, evaluate(controlPoints, 0.0), false);
        return;
    }

    lookupSegmentCount = std::max(std::max(lookupSegmentCount, segmentCount), static_cast<std::size_t>(1));
    const ArcLengthTable table = buildArcLengthTable(*this, lookupSegmentCount);
    const double totalLength = table.totalLength();
    sampledPoints.reserve(segmentCount + 1);

    for (std::size_t i = 0; i <= segmentCount; ++i)
    {
        const double ratio = static_cast<double>(i) / static_cast<double>(segmentCount);
        const double t = totalLength > DBL_EPSILON
                             ? parameterAtLength(table, totalLength * ratio)
                             : ratio;
        appendSample(*this, evaluate(controlPoints, t), removeDuplicatePixels);
    }
    spdlog::debug("Sampled {} points with spacing {:.5f}", sampledPoints.size(), spacing);
}

Eigen::Vector2f BezierCurve::derivative(double t) const
{
    const int n = controlPoints.size() - 1;

    if (n <= 0)
        return Eigen::Vector2f::Zero();

    std::vector<Eigen::Vector2f> dPoints;
    dPoints.reserve(n);

    for (int i = 0; i < n; ++i)
    {
        dPoints.push_back(
            static_cast<float>(n) * (controlPoints[i + 1] - controlPoints[i])
        );
    }

    return evaluateBezier(dPoints, t);
}

Eigen::Vector2f BezierCurve::secondDerivative(double t) const
{
    const int n = static_cast<int>(controlPoints.size()) - 1;

    if (n <= 1)
        return Eigen::Vector2f::Zero();

    std::vector<Eigen::Vector2f> ddPoints;
    ddPoints.reserve(n - 1);

    for (int i = 0; i < n - 1; ++i)
    {
        ddPoints.push_back(
            static_cast<float>(n * (n - 1)) *
            (controlPoints[i + 2]
             - 2.0f * controlPoints[i + 1]
             + controlPoints[i])
        );
    }

    return evaluateBezier(ddPoints, t);
}

Eigen::Vector2f BezierCurve::closestPointOnCurve(const Eigen::Vector2f& p, int coarseSamples, int newtonIterations) const
{
    if (controlPoints.empty())
        return Eigen::Vector2f::Zero();

    if (controlPoints.size() == 1)
        return controlPoints.front();

    coarseSamples = std::max(coarseSamples, 2);

    // 1. 粗采样找初值
    double bestT = 0.0;
    double bestD2 = DBL_MAX;

    for (int i = 0; i <= coarseSamples; ++i)
    {
        double t = static_cast<double>(i) / static_cast<double>(coarseSamples);
        float d2 = evaluateBezier(controlPoints, t).squaredNorm();

        if (d2 < bestD2)
        {
            bestD2 = d2;
            bestT = t;
        }
    }

    // 2. Newton 迭代
    double t = bestT;

    for (int iter = 0; iter < newtonIterations; ++iter)
    {
        Eigen::Vector2f Bt = evaluateBezier(controlPoints, t);
        Eigen::Vector2f dBt = derivative(t);
        Eigen::Vector2f ddBt = secondDerivative(t);

        Eigen::Vector2f r = Bt - p;

        double g = static_cast<double>(r.dot(dBt));

        double h = static_cast<double>(dBt.dot(dBt) + r.dot(ddBt));

        if (std::abs(h) < 1e-12)
            break;

        double step = g / h;

        if (!std::isfinite(step))
            break;

        double oldT = t;
        double oldD2 = evaluateBezier(controlPoints, oldT).squaredNorm();

        double newT = std::max(0.0, std::min(1.0, t - step));
        double newD2 = evaluateBezier(controlPoints, newT).squaredNorm();

        // 3. 简单 line search，防止 Newton 跑到更差的位置
        int lineSearchCount = 0;
        while (newD2 > oldD2 && lineSearchCount < 10)
        {
            step *= 0.5;
            newT = std::max(0.0, std::min(1.0, t - step));
            newD2 = evaluateBezier(controlPoints, newT).squaredNorm();
            ++lineSearchCount;
        }

        t = newT;

        if (std::abs(t - oldT) < 1e-6)
            break;
    }

    // 4. 与端点再比较一次，防止最近点在端点
    double dStart = evaluateBezier(controlPoints, 0.0).squaredNorm();
    double dEnd = evaluateBezier(controlPoints, 1.0).squaredNorm();
    double dMid = evaluateBezier(controlPoints, t).squaredNorm();

    if (dStart <= dMid && dStart <= dEnd)
        return evaluateBezier(controlPoints, 0.0);

    if (dEnd <= dMid && dEnd <= dStart)
        return evaluateBezier(controlPoints, 1.0);

    return evaluateBezier(controlPoints, t);
}

BezierCurveFitter::BezierCurveFitter(double rho_p, std::size_t minSplitPoints)
    : rho_p_(rho_p), minSplitPoints_(std::max<std::size_t>(2, minSplitPoints))
{
}

std::vector<double> BezierCurveFitter::chordLengthParameters(const std::vector<Eigen::Vector2f> &edge) const
{
    const size_t n = edge.size();
    std::vector<double> parameters(n, 0.0);
    if (n < 2)
        return parameters;

    for (size_t i = 1; i < n; ++i)
        parameters[i] = parameters[i - 1] + std::sqrt(squaredDistance(edge[i - 1], edge[i]));

    const double totalLength = parameters.back();
    if (totalLength <= DBL_EPSILON)
    {
        for (size_t i = 0; i < n; ++i)
            parameters[i] = static_cast<double>(i) / static_cast<double>(n - 1);
        return parameters;
    }

    for (double &parameter : parameters)
        parameter /= totalLength;

    return parameters;
}

std::vector<double> BezierCurveFitter::computeResiduals(const std::vector<Eigen::Vector2f> &controlPoints, const std::vector<Eigen::Vector2f> &edge) const
{
    const size_t n = edge.size();
    std::vector<double> residuals;
    residuals.reserve(n);

    const std::vector<double> parameters = chordLengthParameters(edge);
    for (size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f &edgePoint = edge[i];
        const Eigen::Vector2f curvePoint = evaluate(controlPoints, parameters[i]);
        residuals.push_back(std::sqrt(squaredDistance(curvePoint, edgePoint)));
    }

    return residuals;
}

std::vector<BezierCurve> BezierCurveFitter::fitAdaptive(const std::vector<Eigen::Vector2f> &edge) const
{
    if (edge.size() < 2)
        return std::vector<BezierCurve>();

    std::vector<BezierCurve> fittedCurves;
    std::vector<std::vector<Eigen::Vector2f>> pendingSegments;
    pendingSegments.push_back(edge);

    while (!pendingSegments.empty())
    {
        std::vector<Eigen::Vector2f> segment = std::move(pendingSegments.back());
        pendingSegments.pop_back();

        if (segment.size() < 2)
            continue;

        std::vector<double> residuals;
        BezierCurve curve;
        bool fitted = false;

        for (int order = 1; order <= 3; ++order)
        {
            curve = fitWithEndPoints(segment, order);
            if (curve.controlPoints.empty())
            {
                fitted = true;
                break;
            }

            residuals = computeResiduals(curve.controlPoints, segment);
            double maxResidual = *std::max_element(residuals.begin(), residuals.end());

            if (maxResidual <= rho_p_)
            {
                // printf("Order %d: Max Residual = %.4f\n", order, maxResidual);
                fittedCurves.push_back(curve);
                fitted = true;
                break;
            }
        }

        if (fitted)
            continue;

        if (residuals.empty() || curve.controlPoints.empty())
            continue;

        if (segment.size() < 2 * minSplitPoints_ - 1)
        {
            fittedCurves.push_back(curve);
            continue;
        }

        const size_t firstValidSplit = minSplitPoints_ - 1;
        const size_t lastValidSplit = segment.size() - minSplitPoints_;
        auto maxIt = std::max_element(residuals.begin() + firstValidSplit,
                                      residuals.begin() + lastValidSplit + 1);
        size_t splitIndex = static_cast<size_t>(std::distance(residuals.begin(), maxIt));

        if (splitIndex < firstValidSplit || splitIndex > lastValidSplit)
        {
            fittedCurves.push_back(curve);
            continue;
        }

        std::vector<Eigen::Vector2f> points1(segment.begin(), segment.begin() + splitIndex + 1);
        std::vector<Eigen::Vector2f> points2(segment.begin() + splitIndex, segment.end());

        pendingSegments.push_back(std::move(points2));
        pendingSegments.push_back(std::move(points1));
    }

    return fittedCurves;
}

BezierCurve BezierCurveFitter::fitWithEndPoints(const std::vector<Eigen::Vector2f> &edge, int order) const
{
    int n = edge.size();

    BezierCurve curve;
    Eigen::Vector2f startPoint = edge[0];
    Eigen::Vector2f endPoint = edge[n - 1];

    if (order == 1)
    {
        curve.push_back(startPoint);
        curve.push_back(endPoint);
        return curve;
    }

    int unknowns = order - 1;
    Eigen::MatrixXd A(n, unknowns);
    Eigen::VectorXd b_x(n);
    Eigen::VectorXd b_y(n);
    const std::vector<double> parameters = chordLengthParameters(edge);

    for (int i = 0; i < n; ++i)
    {
        double t = parameters[i];

        double B_0 = bernsteinBasis(order, 0, t);
        double B_end = bernsteinBasis(order, order, t);

        double known_x = B_0 * startPoint.x() + B_end * endPoint.x();
        double known_y = B_0 * startPoint.y() + B_end * endPoint.y();

        for (int k = 1; k <= unknowns; ++k)
            A(i, k - 1) = bernsteinBasis(order, k, t);

        const Eigen::Vector2f &q_i = edge[i];
        b_x(i) = q_i.x() - known_x;
        b_y(i) = q_i.y() - known_y;
    }

    Eigen::VectorXd middle_x = A.colPivHouseholderQr().solve(b_x);
    Eigen::VectorXd middle_y = A.colPivHouseholderQr().solve(b_y);

    curve.reserve(order + 1);
    curve.push_back(startPoint);

    for (int k = 0; k < unknowns; ++k)
    {
        curve.push_back(Eigen::Vector2f(middle_x(k), middle_y(k)));
    }

    curve.push_back(endPoint);
    return curve;
}
