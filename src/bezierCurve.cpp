#include "bezierCurve.h"

static double squaredDistance(const orderedEdgePoint &a, const orderedEdgePoint &b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

BezierCurveFitter::BezierCurveFitter(double rho_p, std::size_t minSplitPoints)
    : rho_p_(rho_p), minSplitPoints_(std::max<std::size_t>(2, minSplitPoints))
{
}

std::vector<double> BezierCurveFitter::chordLengthParameters(const std::vector<orderedEdgePoint> &points) const
{
    const size_t n = points.size();
    std::vector<double> parameters(n, 0.0);
    if (n < 2)
        return parameters;

    for (size_t i = 1; i < n; ++i)
        parameters[i] = parameters[i - 1] + std::sqrt(squaredDistance(points[i - 1], points[i]));

    const double totalLength = parameters.back();
    if (totalLength <= std::numeric_limits<double>::epsilon())
    {
        for (size_t i = 0; i < n; ++i)
            parameters[i] = static_cast<double>(i) / static_cast<double>(n - 1);
        return parameters;
    }

    for (double &parameter : parameters)
        parameter /= totalLength;

    return parameters;
}

double BezierCurveFitter::minDistanceToBezier(const std::vector<orderedEdgePoint> &controlPoints,
                                              const orderedEdgePoint &point) const
{
    if (controlPoints.empty())
        return 0.0;

    const int samples = 32;
    double bestT = 0.0;
    double bestD2 = squaredDistance(evaluate(controlPoints, 0.0), point);

    for (int i = 1; i <= samples; ++i)
    {
        double t = static_cast<double>(i) / static_cast<double>(samples);
        double d2 = squaredDistance(evaluate(controlPoints, t), point);
        if (d2 < bestD2)
        {
            bestD2 = d2;
            bestT = t;
        }
    }

    const double step = 1.0 / static_cast<double>(samples);
    double a = std::max(0.0, bestT - step);
    double b = std::min(1.0, bestT + step);

    // Local refinement with golden-section search over the best sample interval.
    const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
    double c = b - gr * (b - a);
    double d = a + gr * (b - a);
    double fc = squaredDistance(evaluate(controlPoints, c), point);
    double fd = squaredDistance(evaluate(controlPoints, d), point);

    for (int iter = 0; iter < 16; ++iter)
    {
        if (fc < fd)
        {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = squaredDistance(evaluate(controlPoints, c), point);
        }
        else
        {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = squaredDistance(evaluate(controlPoints, d), point);
        }
    }

    const double minD2 = std::min({bestD2, fc, fd});
    return std::sqrt(minD2);
}

std::vector<double> BezierCurveFitter::computeResiduals(const std::vector<orderedEdgePoint> &controlPoints, const Edge &edge) const
{
    const size_t n = edge.mvPoints.size();
    std::vector<double> residuals;
    residuals.reserve(n);

    const std::vector<double> parameters = chordLengthParameters(edge.mvPoints);
    for (size_t i = 0; i < n; ++i)
    {
        const orderedEdgePoint &edgePoint = edge.mvPoints[i];
        const orderedEdgePoint curvePoint = evaluate(controlPoints, parameters[i]);
        residuals.push_back(std::sqrt(squaredDistance(curvePoint, edgePoint)));
    }

    return residuals;
}

orderedEdgePoint BezierCurveFitter::evaluate(const std::vector<orderedEdgePoint> &controlPoints, double t) const
{
    const int order = controlPoints.size() - 1;
    const double u = 1.0 - t;

    if (order == 1)
    {
        orderedEdgePoint point(0.0, 0.0);
        point.x = u * controlPoints[0].x + t * controlPoints[1].x;
        point.y = u * controlPoints[0].y + t * controlPoints[1].y;
        return point;
    }

    if (order == 2)
    {
        const double uu = u * u;
        const double ut2 = 2.0 * u * t;
        const double tt = t * t;
        orderedEdgePoint point(0.0, 0.0);
        point.x = uu * controlPoints[0].x + ut2 * controlPoints[1].x + tt * controlPoints[2].x;
        point.y = uu * controlPoints[0].y + ut2 * controlPoints[1].y + tt * controlPoints[2].y;
        return point;
    }

    if (order == 3)
    {
        const double uu = u * u;
        const double tt = t * t;
        const double uuu = uu * u;
        const double uut3 = 3.0 * uu * t;
        const double utt3 = 3.0 * u * tt;
        const double ttt = tt * t;
        orderedEdgePoint point(0.0, 0.0);
        point.x = uuu * controlPoints[0].x + uut3 * controlPoints[1].x + utt3 * controlPoints[2].x + ttt * controlPoints[3].x;
        point.y = uuu * controlPoints[0].y + uut3 * controlPoints[1].y + utt3 * controlPoints[2].y + ttt * controlPoints[3].y;
        return point;
    }

    orderedEdgePoint point(0.0, 0.0);
    for (int i = 0; i <= order; ++i)
    {
        double weight = bernsteinBasis(order, i, t);
        point.x += weight * controlPoints[i].x;
        point.y += weight * controlPoints[i].y;
    }
    return point;
}

std::vector<std::vector<orderedEdgePoint>> BezierCurveFitter::fitAdaptive(const std::vector<orderedEdgePoint> &points) const
{
    if (points.size() < 2)
        return std::vector<std::vector<orderedEdgePoint>>();

    std::vector<std::vector<orderedEdgePoint>> fittedCurves;
    std::vector<std::vector<orderedEdgePoint>> pendingSegments;
    pendingSegments.push_back(points);

    while (!pendingSegments.empty())
    {
        std::vector<orderedEdgePoint> segment = std::move(pendingSegments.back());
        pendingSegments.pop_back();

        if (segment.size() < 2)
            continue;

        Edge localEdge;
        localEdge.mvPoints = segment;

        std::vector<double> residuals;
        std::vector<orderedEdgePoint> lastControlPoints;
        bool fitted = false;

        const int maxOrder = std::min<int>(3, static_cast<int>(segment.size()) - 1);
        for (int order = 1; order <= maxOrder; ++order)
        {
            lastControlPoints = fitWithEndPoints(localEdge, order);
            if (lastControlPoints.empty())
            {
                fitted = true;
                break;
            }

            residuals = computeResiduals(lastControlPoints, localEdge);
            double maxResidual = *std::max_element(residuals.begin(), residuals.end());

            if (maxResidual <= rho_p_)
            {
                // printf("Order %d: Max Residual = %.4f\n", order, maxResidual);
                fittedCurves.push_back(lastControlPoints);
                fitted = true;
                break;
            }
        }

        if (fitted)
            continue;

        if (residuals.empty() || lastControlPoints.empty())
            continue;

        if (segment.size() < 2 * minSplitPoints_ - 1)
        {
            fittedCurves.push_back(lastControlPoints);
            continue;
        }

        const size_t firstValidSplit = minSplitPoints_ - 1;
        const size_t lastValidSplit = segment.size() - minSplitPoints_;
        auto maxIt = std::max_element(residuals.begin() + firstValidSplit,
                                      residuals.begin() + lastValidSplit + 1);
        size_t splitIndex = static_cast<size_t>(std::distance(residuals.begin(), maxIt));

        if (splitIndex < firstValidSplit || splitIndex > lastValidSplit)
        {
            fittedCurves.push_back(lastControlPoints);
            continue;
        }

        std::vector<orderedEdgePoint> points1(segment.begin(), segment.begin() + splitIndex + 1);
        std::vector<orderedEdgePoint> points2(segment.begin() + splitIndex, segment.end());

        pendingSegments.push_back(std::move(points2));
        pendingSegments.push_back(std::move(points1));
    }

    return fittedCurves;
}

std::vector<orderedEdgePoint> BezierCurveFitter::fitWithEndPoints(const Edge &edge, int order) const
{
    int n = edge.mvPoints.size();
    if (n < 2 || order < 1)
        return std::vector<orderedEdgePoint>();
    
    orderedEdgePoint startPoint = edge.mvPoints[0];
    orderedEdgePoint endPoint = edge.mvPoints[n - 1];

    if (order == 1)
        return {startPoint, endPoint};

    int unknowns = order - 1;
    Eigen::MatrixXd A(n, unknowns);
    Eigen::VectorXd b_x(n);
    Eigen::VectorXd b_y(n);
    const std::vector<double> parameters = chordLengthParameters(edge.mvPoints);

    for (int i = 0; i < n; ++i)
    {
        double t = parameters[i];

        double B_0 = bernsteinBasis(order, 0, t);
        double B_end = bernsteinBasis(order, order, t);

        double known_x = B_0 * startPoint.x + B_end * endPoint.x;
        double known_y = B_0 * startPoint.y + B_end * endPoint.y;

        for (int k = 1; k <= unknowns; ++k)
        {
            A(i, k - 1) = bernsteinBasis(order, k, t);
        }

        const orderedEdgePoint &q_i = edge.mvPoints[i];
        b_x(i) = q_i.x - known_x;
        b_y(i) = q_i.y - known_y;
    }

    Eigen::VectorXd middle_x = A.colPivHouseholderQr().solve(b_x);
    Eigen::VectorXd middle_y = A.colPivHouseholderQr().solve(b_y);

    std::vector<orderedEdgePoint> controlPoints;
    controlPoints.reserve(order + 1);
    controlPoints.push_back(startPoint);

    for (int k = 0; k < unknowns; ++k)
    {
        controlPoints.emplace_back(middle_x(k), middle_y(k));
    }

    controlPoints.push_back(endPoint);
    return controlPoints;
}

double BezierCurveFitter::bernsteinBasis(int n, int i, double t) const
{
    const double u = 1.0 - t;
    if (n == 1)
        return i == 0 ? u : t;

    if (n == 2)
    {
        if (i == 0)
            return u * u;
        if (i == 1)
            return 2.0 * u * t;
        return t * t;
    }

    if (n == 3)
    {
        if (i == 0)
            return u * u * u;
        if (i == 1)
            return 3.0 * u * u * t;
        if (i == 2)
            return 3.0 * u * t * t;
        return t * t * t;
    }

    double binomialCoeff = 1.0;
    for (int j = 0; j < i; ++j)
        binomialCoeff *= (n - j) / static_cast<double>(j + 1);

    return binomialCoeff * std::pow(u, n - i) * std::pow(t, i);
}
