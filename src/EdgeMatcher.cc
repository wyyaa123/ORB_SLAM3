#include "EdgeMatcher.h"
#include "Hungarian.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace ORB_SLAM3
{
    namespace
    {
        const char *kBezierMatchWindow = "Bezier edge matches";
        const char *kBezierMatchOutputDirectory = "output";

        cv::Mat PrepareFrameImage(const cv::Mat &frameImage, const cv::Size &imageSize)
        {
            cv::Mat edgeImage;
            if (frameImage.empty())
            {
                edgeImage = cv::Mat::zeros(imageSize, CV_8UC3);
            }
            else
            {
                cv::Mat image8U;
                if (frameImage.depth() == CV_8U)
                    image8U = frameImage;
                else
                    cv::normalize(frameImage, image8U, 0, 255, cv::NORM_MINMAX, CV_8U);

                if (image8U.channels() == 1)
                    cv::cvtColor(image8U, edgeImage, cv::COLOR_GRAY2BGR);
                else if (image8U.channels() == 3)
                    edgeImage = image8U.clone();
                else if (image8U.channels() == 4)
                    cv::cvtColor(image8U, edgeImage, cv::COLOR_BGRA2BGR);
                else
                    edgeImage = cv::Mat::zeros(imageSize, CV_8UC3);

                if (edgeImage.size() != imageSize)
                    cv::resize(edgeImage, edgeImage, imageSize, 0.0, 0.0, cv::INTER_LINEAR);
            }

            return edgeImage;
        }

        void DrawCurve(cv::Mat &image, const BezierCurve &curve, const cv::Scalar &color)
        {
            if (curve.sampledPoints.empty())
                return;

            if (curve.sampledPoints.size() == 1)
            {
                cv::circle(image, cv::Point(cvRound(curve.sampledPoints.front().x()), cvRound(curve.sampledPoints.front().y())), 2, color, cv::FILLED, cv::LINE_AA);
                return;
            }

            for (size_t i = 1; i < curve.sampledPoints.size(); ++i)
            {
                cv::Point first(cvRound(curve.sampledPoints[i - 1].x()), cvRound(curve.sampledPoints[i - 1].y()));
                cv::Point second(cvRound(curve.sampledPoints[i].x()), cvRound(curve.sampledPoints[i].y()));
                if (cv::clipLine(image.size(), first, second))
                    cv::line(image, first, second, color, 2, cv::LINE_AA);
            }
        }

        cv::Scalar MatchColor(const size_t matchIndex)
        {
            // Golden-ratio hue steps keep adjacent matches visually distinct.
            const int hue = static_cast<int>((matchIndex * 67) % 180);
            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 220, 255));
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
            const cv::Vec3b color = bgr.at<cv::Vec3b>(0, 0);
            return cv::Scalar(color[0], color[1], color[2]);
        }

        cv::Scalar MapCurveColor(const MapCurve *pMapCurve)
        {
            return MatchColor(pMapCurve ? static_cast<size_t>(pMapCurve->mnId) : 0);
        }

        void DrawProjectedMapCurves(cv::Mat &image,
                                    const std::vector<MapCurve *> &mapCurves,
                                    const Sophus::SE3f &Tcw,
                                    GeometricCamera *pCamera)
        {
            if (!pCamera)
                return;

            std::unordered_set<MapCurve *> processedMapCurves;
            constexpr float projectionMargin = 8.0f;
            for (MapCurve *pMapCurve : mapCurves)
            {
                if (!pMapCurve || pMapCurve->isBad() ||
                    !processedMapCurves.insert(pMapCurve).second)
                {
                    continue;
                }

                const std::vector<Eigen::Vector3f> worldPoints =
                    pMapCurve->GetWorldPoints();
                if (worldPoints.size() < 2)
                    continue;

                const cv::Scalar color = MapCurveColor(pMapCurve);
                bool hasPreviousPoint = false;
                cv::Point previousPoint;
                for (const Eigen::Vector3f &worldPoint : worldPoints)
                {
                    const Eigen::Vector3f cameraPoint = Tcw * worldPoint;
                    if (cameraPoint.z() <= 0.0f)
                    {
                        hasPreviousPoint = false;
                        continue;
                    }

                    const Eigen::Vector2f projectedPoint =
                        pCamera->project(cameraPoint);
                    if (!projectedPoint.allFinite() ||
                        projectedPoint.x() < -projectionMargin ||
                        projectedPoint.x() > image.cols + projectionMargin ||
                        projectedPoint.y() < -projectionMargin ||
                        projectedPoint.y() > image.rows + projectionMargin)
                    {
                        hasPreviousPoint = false;
                        continue;
                    }

                    const cv::Point currentPoint(cvRound(projectedPoint.x()),
                                                 cvRound(projectedPoint.y()));
                    if (hasPreviousPoint)
                    {
                        cv::Point first = previousPoint;
                        cv::Point second = currentPoint;
                        if (cv::clipLine(image.size(), first, second))
                            cv::line(image, first, second, color, 2, cv::LINE_AA);
                    }

                    previousPoint = currentPoint;
                    hasPreviousPoint = true;
                }
            }
        }

        constexpr double kInvalidCurveCost = 1e6;
        constexpr double kUnmatchedCurveCost = 6.0;
        constexpr double kFragmentCurveCost = 6.5;

        struct ProjectedMapCurve
        {
            MapCurve *pMapBezier = nullptr;
            std::vector<Eigen::Vector2f> points;
            cv::Rect2f bounds;
        };

        struct CurveSimilarity
        {
            double cost = kInvalidCurveCost;
            double meanNormalOffset = 0.0;
            double normalOffsetStdDev = 0.0;
            int minMapIndex = -1;
            int maxMapIndex = -1;
            bool hasNormalOffset = false;
            bool valid = false;
        };

        struct AcceptedCurveFragment
        {
            size_t curveIndex = 0;
            CurveSimilarity similarity;
        };

        std::vector<Eigen::Vector2f> LimitCurveSamples(const std::vector<Eigen::Vector2f> &points, const size_t maxSamples = 96)
        {
            if (points.size() <= maxSamples)
                return points;

            std::vector<Eigen::Vector2f> samples;
            samples.reserve(maxSamples);
            for (size_t i = 0; i < maxSamples; ++i)
            {
                const size_t index = static_cast<size_t>(std::round(
                    static_cast<double>(i) * (points.size() - 1) / static_cast<double>(maxSamples - 1)));
                samples.push_back(points[index]);
            }
            return samples;
        }

        Eigen::Vector2f CurveTangentAt(const std::vector<Eigen::Vector2f> &points, const size_t index)
        {
            if (points.size() < 2)
                return Eigen::Vector2f::Zero();

            const size_t first = index == 0 ? 0 : index - 1;
            const size_t second = std::min(index + 1, points.size() - 1);
            Eigen::Vector2f tangent = points[second] - points[first];
            const float norm = tangent.norm();
            if (norm <= 1e-6f)
                return Eigen::Vector2f::Zero();
            return tangent / norm;
        }

        cv::Rect2f CurveBounds(const std::vector<Eigen::Vector2f> &points)
        {
            float minX = FLT_MAX;
            float minY = FLT_MAX;
            float maxX = -FLT_MAX;
            float maxY = -FLT_MAX;
            for (const Eigen::Vector2f &point : points)
            {
                minX = std::min(minX, point.x());
                minY = std::min(minY, point.y());
                maxX = std::max(maxX, point.x());
                maxY = std::max(maxY, point.y());
            }
            return cv::Rect2f(minX, minY, std::max(0.0f, maxX - minX), std::max(0.0f, maxY - minY));
        }

        bool BoundsOverlap(const cv::Rect2f &first, const cv::Rect2f &second, const float margin)
        {
            return first.x - margin <= second.x + second.width && second.x - margin <= first.x + first.width && first.y - margin <= second.y + second.height
                   && second.y - margin <= first.y + first.height;
        }

        double DtwLocalCost(const std::vector<Eigen::Vector2f> &mapPoints, const size_t mapIndex, const std::vector<Eigen::Vector2f> &curvePoints, const size_t curveIndex)
        {
            const double distance = (mapPoints[mapIndex] - curvePoints[curveIndex]).norm();
            const Eigen::Vector2f mapTangent = CurveTangentAt(mapPoints, mapIndex);
            const Eigen::Vector2f curveTangent = CurveTangentAt(curvePoints, curveIndex);
            double directionCost = 1.0;
            if (mapTangent.squaredNorm() > 0.0f && curveTangent.squaredNorm() > 0.0f)
            {
                directionCost = 1.0 - std::abs(static_cast<double>(mapTangent.dot(curveTangent)));
            }
            return distance + 4.0 * directionCost;
        }

        double SubsequenceDtwCost(const std::vector<Eigen::Vector2f> &mapPoints, const std::vector<Eigen::Vector2f> &curvePoints)
        {
            if (mapPoints.size() < 2 || curvePoints.size() < 2)
                return kInvalidCurveCost;

            std::vector<double> previous(mapPoints.size());
            std::vector<double> current(mapPoints.size(), kInvalidCurveCost);
            for (size_t mapIdx = 0; mapIdx < mapPoints.size(); ++mapIdx)
                previous[mapIdx] = DtwLocalCost(mapPoints, mapIdx, curvePoints, 0);

            constexpr size_t maxMapAdvance = 4;
            for (size_t curveIdx = 1; curveIdx < curvePoints.size(); ++curveIdx)
            {
                std::fill(current.begin(), current.end(), kInvalidCurveCost);
                for (size_t mapIdx = 0; mapIdx < mapPoints.size(); ++mapIdx)
                {
                    double previousBest = previous[mapIdx];
                    const size_t begin = mapIdx > maxMapAdvance
                                             ? mapIdx - maxMapAdvance
                                             : 0;
                    for (size_t previousMapIdx = begin;
                         previousMapIdx < mapIdx; ++previousMapIdx)
                    {
                        previousBest = std::min(previousBest, previous[previousMapIdx]);
                    }
                    current[mapIdx] = previousBest + DtwLocalCost(mapPoints, mapIdx, curvePoints, curveIdx);
                }
                previous.swap(current);
            }

            return *std::min_element(previous.begin(), previous.end()) / static_cast<double>(curvePoints.size());
        }

        CurveSimilarity ComputeCurveSimilarity(const ProjectedMapCurve &mapCurve, const BezierCurve &observedCurve)
        {
            CurveSimilarity result;
            std::vector<Eigen::Vector2f> mapPoints = LimitCurveSamples(mapCurve.points);
            std::vector<Eigen::Vector2f> curvePoints = LimitCurveSamples(observedCurve.sampledPoints);
            if (mapPoints.size() < 2 || curvePoints.size() < 2)
                return result;

            const cv::Rect2f observedBounds = CurveBounds(curvePoints);
            if (!BoundsOverlap(mapCurve.bounds, observedBounds, 8.0f))
                return result;

            const Eigen::Vector2f mapDirection = (mapPoints.back() - mapPoints.front()).normalized();
            const Eigen::Vector2f curveDirection = (curvePoints.back() - curvePoints.front()).normalized();
            if (mapDirection.allFinite() && curveDirection.allFinite() && std::abs(mapDirection.dot(curveDirection)) < 0.26f)
            {
                return result;
            }

            double distanceSum = 0.0;
            double directionCostSum = 0.0;
            double normalOffsetSum = 0.0;
            double normalOffsetSquaredSum = 0.0;
            size_t normalOffsetCount = 0;
            int minMapIndex = static_cast<int>(mapPoints.size());
            int maxMapIndex = -1;
            for (size_t curveIdx = 0; curveIdx < curvePoints.size(); ++curveIdx)
            {
                size_t nearestMapIdx = 0;
                float nearestDist2 = std::numeric_limits<float>::max();
                for (size_t mapIdx = 0; mapIdx < mapPoints.size(); ++mapIdx)
                {
                    const float dist2 = (curvePoints[curveIdx] - mapPoints[mapIdx]).squaredNorm();
                    if (dist2 < nearestDist2)
                    {
                        nearestDist2 = dist2;
                        nearestMapIdx = mapIdx;
                    }
                }

                distanceSum += std::sqrt(nearestDist2);
                const Eigen::Vector2f mapTangent = CurveTangentAt(mapPoints, nearestMapIdx);
                const Eigen::Vector2f curveTangent = CurveTangentAt(curvePoints, curveIdx);
                if (mapTangent.squaredNorm() > 0.0f && curveTangent.squaredNorm() > 0.0f)
                {
                    directionCostSum += 1.0 - std::abs(static_cast<double>(mapTangent.dot(curveTangent)));

                    const Eigen::Vector2f mapNormal(-mapTangent.y(), mapTangent.x());
                    const double normalOffset = static_cast<double>(
                        (curvePoints[curveIdx] - mapPoints[nearestMapIdx]).dot(mapNormal));
                    normalOffsetSum += normalOffset;
                    normalOffsetSquaredSum += normalOffset * normalOffset;
                    ++normalOffsetCount;
                }
                minMapIndex = std::min(minMapIndex, static_cast<int>(nearestMapIdx));
                maxMapIndex = std::max(maxMapIndex, static_cast<int>(nearestMapIdx));
            }

            const double meanDistance = distanceSum / curvePoints.size();
            const double meanDirectionCost = directionCostSum / curvePoints.size();
            if (meanDistance > 7.0 || meanDirectionCost > 0.35)
                return result;

            const double forwardDtw = SubsequenceDtwCost(mapPoints, curvePoints);
            std::reverse(curvePoints.begin(), curvePoints.end());
            const double reverseDtw = SubsequenceDtwCost(mapPoints, curvePoints);
            const double dtwCost = std::min(forwardDtw, reverseDtw);

            result.cost = 0.55 * dtwCost + 0.30 * meanDistance + 0.15 * (6.0 * meanDirectionCost);
            result.minMapIndex = minMapIndex;
            result.maxMapIndex = maxMapIndex;
            if (normalOffsetCount > 0)
            {
                result.meanNormalOffset = normalOffsetSum / normalOffsetCount;
                const double normalOffsetVariance = std::max(
                    0.0,
                    normalOffsetSquaredSum / normalOffsetCount -
                        result.meanNormalOffset * result.meanNormalOffset);
                result.normalOffsetStdDev = std::sqrt(normalOffsetVariance);
                result.hasNormalOffset = true;
            }
            result.valid = std::isfinite(result.cost) && result.cost < kFragmentCurveCost;
            return result;
        }

        std::vector<ProjectedMapCurve> BuildProjectedMapCurves(Frame &currentFrame, const std::vector<MapCurve *> &mapBeziers)
        {
            std::vector<ProjectedMapCurve> projectedCurves;
            std::unordered_set<MapCurve *> processed;
            const Sophus::SE3f Tcw = currentFrame.GetPose();
            for (MapCurve *pMapBezier : mapBeziers)
            {
                if (!pMapBezier || pMapBezier->isBad() || !processed.insert(pMapBezier).second)
                {
                    continue;
                }

                ProjectedMapCurve projected;
                projected.pMapBezier = pMapBezier;
                for (const Eigen::Vector3f &worldPoint : pMapBezier->GetWorldPoints())
                {
                    const Eigen::Vector3f cameraPoint = Tcw * worldPoint;
                    if (cameraPoint.z() <= 0.0f)
                        continue;
                    const Eigen::Vector2f uv = currentFrame.mpCamera->project(cameraPoint);
                    if (!uv.allFinite() || uv.x() < currentFrame.mnMinX - 8.0f || uv.x() > currentFrame.mnMaxX + 8.0f || uv.y() < currentFrame.mnMinY - 8.0f
                        || uv.y() > currentFrame.mnMaxY + 8.0f)
                    {
                        continue;
                    }
                    projected.points.push_back(uv);
                }

                if (projected.points.size() < 2)
                    continue;
                projected.points = LimitCurveSamples(projected.points);
                projected.bounds = CurveBounds(projected.points);
                projectedCurves.push_back(std::move(projected));
            }
            return projectedCurves;
        }

        bool CurveEndpointsAreContinuous(const BezierCurve &first,
                                         const BezierCurve &second,
                                         const float maxEndpointDistance)
        {
            if (first.sampledPoints.size() < 2 || second.sampledPoints.size() < 2)
                return false;

            const size_t firstEndpointIndices[2] = {0, first.sampledPoints.size() - 1};
            const size_t secondEndpointIndices[2] = {0, second.sampledPoints.size() - 1};
            float bestDistance2 = std::numeric_limits<float>::max();
            size_t bestFirstIndex = 0;
            size_t bestSecondIndex = 0;
            for (const size_t firstIndex : firstEndpointIndices)
            {
                for (const size_t secondIndex : secondEndpointIndices)
                {
                    const float distance2 =
                        (first.sampledPoints[firstIndex] -
                         second.sampledPoints[secondIndex])
                            .squaredNorm();
                    if (distance2 < bestDistance2)
                    {
                        bestDistance2 = distance2;
                        bestFirstIndex = firstIndex;
                        bestSecondIndex = secondIndex;
                    }
                }
            }

            if (bestDistance2 > maxEndpointDistance * maxEndpointDistance)
                return false;

            const Eigen::Vector2f firstTangent =
                CurveTangentAt(first.sampledPoints, bestFirstIndex);
            const Eigen::Vector2f secondTangent =
                CurveTangentAt(second.sampledPoints, bestSecondIndex);
            if (firstTangent.squaredNorm() <= 0.0f ||
                secondTangent.squaredNorm() <= 0.0f)
            {
                return false;
            }

            return std::abs(firstTangent.dot(secondTangent)) >= 0.75f;
        }

        bool FragmentGeometryIsCompatible(
            const CurveSimilarity &candidate,
            const BezierCurve &candidateCurve,
            const std::vector<AcceptedCurveFragment> &acceptedFragments,
            const std::vector<BezierCurve> &observedCurves,
            const size_t mapPointCount)
        {
            if (acceptedFragments.empty())
                return false;

            // Independent parallel curves tend to cover the same part of the
            // projected map curve. Genuine split fragments may share one or two
            // samples at their endpoints, but should not overlap substantially.
            for (const AcceptedCurveFragment &accepted : acceptedFragments)
            {
                const CurveSimilarity &acceptedSimilarity = accepted.similarity;
                const int overlapBegin = std::max(candidate.minMapIndex,
                                                  acceptedSimilarity.minMapIndex);
                const int overlapEnd = std::min(candidate.maxMapIndex,
                                                acceptedSimilarity.maxMapIndex);
                const int overlap = std::max(0, overlapEnd - overlapBegin + 1);
                const int candidateSpan = std::max(
                    1, candidate.maxMapIndex - candidate.minMapIndex + 1);
                const int acceptedSpan = std::max(
                    1, acceptedSimilarity.maxMapIndex -
                           acceptedSimilarity.minMapIndex + 1);
                const int shorterSpan = std::min(candidateSpan, acceptedSpan);
                const int allowedOverlap = std::max(
                    1, static_cast<int>(std::ceil(0.15 * shorterSpan)));
                if (overlap > allowedOverlap)
                    return false;
            }

            constexpr float maxEndpointDistance = 12.0f;
            constexpr double baseNormalOffsetTolerance = 2.5;
            const int maxGap = std::max(
                4, static_cast<int>(0.25f * mapPointCount));
            for (const AcceptedCurveFragment &accepted : acceptedFragments)
            {
                const CurveSimilarity &acceptedSimilarity = accepted.similarity;
                int gap = 0;
                if (candidate.maxMapIndex < acceptedSimilarity.minMapIndex)
                    gap = acceptedSimilarity.minMapIndex - candidate.maxMapIndex;
                else if (acceptedSimilarity.maxMapIndex < candidate.minMapIndex)
                    gap = candidate.minMapIndex - acceptedSimilarity.maxMapIndex;
                if (gap > maxGap)
                    continue;

                if (candidate.hasNormalOffset && acceptedSimilarity.hasNormalOffset)
                {
                    const double noiseAllowance = std::min(
                        1.5,
                        candidate.normalOffsetStdDev +
                            acceptedSimilarity.normalOffsetStdDev);
                    if (std::abs(candidate.meanNormalOffset -
                                 acceptedSimilarity.meanNormalOffset) >
                        baseNormalOffsetTolerance + noiseAllowance)
                    {
                        continue;
                    }
                }

                if (accepted.curveIndex >= observedCurves.size() ||
                    !CurveEndpointsAreContinuous(
                        candidateCurve,
                        observedCurves[accepted.curveIndex],
                        maxEndpointDistance))
                {
                    continue;
                }

                return true;
            }

            return false;
        }

        int MatchCurvesWithHungarian(Frame &currentFrame, const std::vector<MapCurve *> &mapBeziers, std::vector<MapCurve *> &matches)
        {
            const size_t curveCount = currentFrame.NC;
            if (curveCount == 0 || !currentFrame.mpCamera)
                return 0;

            std::vector<ProjectedMapCurve> projectedCurves = BuildProjectedMapCurves(currentFrame, mapBeziers);
            if (projectedCurves.empty())
                return 0;

            std::vector<size_t> rowToCurveIndex;
            for (size_t curveIdx = 0; curveIdx < curveCount; ++curveIdx)
            {
                if (!matches[curveIdx] && currentFrame.mvBezierCurves[curveIdx].sampledPoints.size() >= 2)
                {
                    rowToCurveIndex.push_back(curveIdx);
                }
            }
            if (rowToCurveIndex.empty())
                return 0;

            const size_t rowCount = rowToCurveIndex.size();
            const size_t realColumnCount = projectedCurves.size();
            std::vector<std::vector<CurveSimilarity>> similarities(rowCount, std::vector<CurveSimilarity>(realColumnCount));
            std::vector<std::vector<double>> costMatrix(rowCount, std::vector<double>(realColumnCount + rowCount, kInvalidCurveCost));

            for (size_t row = 0; row < rowCount; ++row)
            {
                const BezierCurve &curve = currentFrame.mvBezierCurves[rowToCurveIndex[row]];
                for (size_t column = 0; column < realColumnCount; ++column)
                {
                    similarities[row][column] = ComputeCurveSimilarity(projectedCurves[column], curve);
                    if (similarities[row][column].valid)
                    {
                        costMatrix[row][column] = similarities[row][column].cost;
                    }
                }
                costMatrix[row][realColumnCount + row] = kUnmatchedCurveCost;
            }

            HungarianAlgorithm hungarian;
            std::vector<int> assignment;
            hungarian.Solve(costMatrix, assignment);

            int nMatches = 0;
            std::vector<bool> anchoredMaps(realColumnCount, false);
            std::vector<std::vector<AcceptedCurveFragment>> acceptedFragments(realColumnCount);
            std::vector<bool> matchedRows(rowCount, false);

            // Existing matches are also valid anchors. This is important in
            // local-map tracking, where a map curve may already be associated
            // with one fragment before additional split fragments are searched.
            for (size_t curveIdx = 0; curveIdx < curveCount; ++curveIdx)
            {
                MapCurve *pMatchedMapCurve = matches[curveIdx];
                if (!pMatchedMapCurve ||
                    currentFrame.mvBezierCurves[curveIdx].sampledPoints.size() < 2)
                {
                    continue;
                }

                for (size_t column = 0; column < realColumnCount; ++column)
                {
                    if (projectedCurves[column].pMapBezier != pMatchedMapCurve)
                        continue;

                    const CurveSimilarity similarity = ComputeCurveSimilarity(
                        projectedCurves[column], currentFrame.mvBezierCurves[curveIdx]);
                    if (similarity.valid)
                    {
                        anchoredMaps[column] = true;
                        acceptedFragments[column].push_back(
                            {curveIdx, similarity});
                    }
                    break;
                }
            }

            for (size_t row = 0; row < assignment.size(); ++row)
            {
                const int column = assignment[row];
                if (column < 0 || column >= static_cast<int>(realColumnCount) || !similarities[row][column].valid
                    || similarities[row][column].cost >= kUnmatchedCurveCost)
                {
                    continue;
                }

                double alternativeCost = kInvalidCurveCost;
                for (size_t alternative = 0;
                     alternative < realColumnCount; ++alternative)
                {
                    if (alternative == static_cast<size_t>(column) || !similarities[row][alternative].valid)
                    {
                        continue;
                    }
                    alternativeCost = std::min(alternativeCost, similarities[row][alternative].cost);
                }
                if (alternativeCost < kInvalidCurveCost && similarities[row][column].cost > 0.9 * alternativeCost)
                {
                    continue;
                }

                const size_t curveIdx = rowToCurveIndex[row];
                matches[curveIdx] = projectedCurves[column].pMapBezier;
                matchedRows[row] = true;
                anchoredMaps[column] = true;
                acceptedFragments[column].push_back(
                    {curveIdx, similarities[row][column]});
                ++nMatches;
            }

            // Hungarian anchors are one-to-one. Attach remaining compatible
            // short fragments to an anchored map curve to recover many-to-one.
            for (size_t row = 0; row < rowCount; ++row)
            {
                if (matchedRows[row])
                    continue;

                int bestColumn = -1;
                double bestCost = kFragmentCurveCost;
                double secondBestCost = kInvalidCurveCost;
                for (size_t column = 0; column < realColumnCount; ++column)
                {
                    const CurveSimilarity &similarity = similarities[row][column];
                    const size_t curveIdx = rowToCurveIndex[row];
                    if (!anchoredMaps[column] || !similarity.valid ||
                        !FragmentGeometryIsCompatible(
                            similarity,
                            currentFrame.mvBezierCurves[curveIdx],
                            acceptedFragments[column],
                            currentFrame.mvBezierCurves,
                            projectedCurves[column].points.size()))
                    {
                        continue;
                    }

                    if (similarity.cost < bestCost)
                    {
                        secondBestCost = bestCost;
                        bestCost = similarity.cost;
                        bestColumn = static_cast<int>(column);
                    }
                    else if (similarity.cost < secondBestCost)
                    {
                        secondBestCost = similarity.cost;
                    }
                }

                if (bestColumn < 0 || (secondBestCost < kInvalidCurveCost && bestCost > 0.9 * secondBestCost))
                {
                    continue;
                }

                const size_t curveIdx = rowToCurveIndex[row];
                matches[curveIdx] = projectedCurves[bestColumn].pMapBezier;
                acceptedFragments[bestColumn].push_back(
                    {curveIdx, similarities[row][bestColumn]});
                ++nMatches;
            }

            return nMatches;
        }

        void EnsureOutputDirectory()
        {
            if (mkdir(kBezierMatchOutputDirectory, 0755) != 0 && errno != EEXIST)
            {
                std::cerr << "Could not create Bezier match output directory '"
                          << kBezierMatchOutputDirectory << "': errno " << errno
                          << std::endl;
            }
        }

        void VisualizeBezierMatches(KeyFrame *pKF, const Frame &currentFrame, const std::vector<MapCurve *> &vpMapBezierMatches)
        {
            if (!pKF || currentFrame.mImgLeft.empty())
                return;

            const cv::Size imageSize = currentFrame.mImgLeft.size();
            cv::Mat referenceEdges = PrepareFrameImage(pKF->mImgLeft, imageSize);
            cv::Mat currentEdges = PrepareFrameImage(currentFrame.mImgLeft, imageSize);
            const std::vector<MapCurve *> referenceMatches = pKF->GetMapCurveMatches();

            struct MatchGroup
            {
                cv::Scalar color;
            };
            std::unordered_map<MapCurve *, MatchGroup> matchGroups;
            size_t fragmentMatchCount = 0;
            for (size_t currentIndex = 0; currentIndex < vpMapBezierMatches.size()
                                          && currentIndex < currentFrame.mvBezierCurves.size(); ++currentIndex)
            {
                MapCurve *pMatch = vpMapBezierMatches[currentIndex];
                if (!pMatch)
                    continue;

                auto groupIt = matchGroups.find(pMatch);
                if (groupIt == matchGroups.end())
                {
                    const cv::Scalar color = MapCurveColor(pMatch);
                    bool hasReferenceCurve = false;

                    // A reference keyframe may itself contain several curve
                    // fragments associated with the same map edge. Draw all of
                    // them using the same group color.
                    const size_t referenceCount = std::min(referenceMatches.size(), pKF->mvBezierCurves.size());
                    for (size_t referenceIndex = 0;
                         referenceIndex < referenceCount; ++referenceIndex)
                    {
                        if (referenceMatches[referenceIndex] != pMatch)
                            continue;

                        const BezierCurve &referenceCurve = pKF->mvBezierCurves[referenceIndex];
                        DrawCurve(referenceEdges, referenceCurve, color);
                        hasReferenceCurve = hasReferenceCurve || !referenceCurve.sampledPoints.empty();
                    }

                    if (!hasReferenceCurve)
                        continue;

                    groupIt = matchGroups.emplace(pMatch, MatchGroup{color}).first;
                }

                const MatchGroup &group = groupIt->second;
                const BezierCurve &currentCurve = currentFrame.mvBezierCurves[currentIndex];
                DrawCurve(currentEdges, currentCurve, group.color);

                ++fragmentMatchCount;
            }

            // The two additional panels show every detected edge exclusively as
            // a sampled Bezier curve; raw discrete edge pixels are not drawn.
            cv::Mat referenceAllCurves = PrepareFrameImage(pKF->mImgLeft, imageSize);
            cv::Mat currentAllCurves = PrepareFrameImage(currentFrame.mImgLeft, imageSize);
            for (size_t curveIdx = 0;
                 curveIdx < pKF->mvBezierCurves.size(); ++curveIdx)
            {
                DrawCurve(referenceAllCurves, pKF->mvBezierCurves[curveIdx], MatchColor(curveIdx));
            }
            for (size_t curveIdx = 0;
                 curveIdx < currentFrame.mvBezierCurves.size(); ++curveIdx)
            {
                DrawCurve(currentAllCurves, currentFrame.mvBezierCurves[curveIdx], MatchColor(curveIdx));
            }

            cv::Mat referenceProjectedMapCurves = PrepareFrameImage(pKF->mImgLeft, imageSize);
            cv::Mat currentProjectedMapCurves = PrepareFrameImage(currentFrame.mImgLeft, imageSize);
            DrawProjectedMapCurves(referenceProjectedMapCurves, referenceMatches,
                                   pKF->GetPose(), pKF->mpCamera);
            DrawProjectedMapCurves(currentProjectedMapCurves, vpMapBezierMatches,
                                   currentFrame.GetPose(), currentFrame.mpCamera);

            const int headerHeight = 34;
            const int panelGap = 24;
            const int rowGap = 18;
            const int currentOffsetX = imageSize.width + panelGap;
            const int allCurvesHeaderY = headerHeight + imageSize.height + rowGap;
            const int allCurvesImageY = allCurvesHeaderY + headerHeight;
            const int projectionHeaderY = allCurvesImageY + imageSize.height + rowGap;
            const int projectionImageY = projectionHeaderY + headerHeight;
            cv::Mat visualization = cv::Mat::zeros(imageSize.height * 3 + headerHeight * 3 + rowGap * 2, imageSize.width * 2 + panelGap, CV_8UC3);
            referenceEdges.copyTo(visualization(cv::Rect(0, headerHeight, imageSize.width, imageSize.height)));
            currentEdges.copyTo(visualization(cv::Rect(currentOffsetX, headerHeight, imageSize.width, imageSize.height)));
            referenceAllCurves.copyTo(visualization(cv::Rect(0, allCurvesImageY, imageSize.width, imageSize.height)));
            currentAllCurves.copyTo(visualization(cv::Rect(currentOffsetX, allCurvesImageY, imageSize.width, imageSize.height)));
            referenceProjectedMapCurves.copyTo(visualization(cv::Rect(0, projectionImageY, imageSize.width, imageSize.height)));
            currentProjectedMapCurves.copyTo(visualization(cv::Rect(currentOffsetX, projectionImageY, imageSize.width, imageSize.height)));

            std::ostringstream referenceLabel;
            referenceLabel << "Reference KF " << pKF->mnId
                           << " - matched Bezier curves";
            std::ostringstream currentLabel;
            currentLabel << "Current frame " << currentFrame.mnId
                         << " - matched Bezier curves"
                         << " | map edges: " << matchGroups.size()
                         << " | fragments: " << fragmentMatchCount;
            cv::putText(visualization, referenceLabel.str(), cv::Point(8, 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(visualization, "Reference KF - all Bezier curves", cv::Point(8, allCurvesHeaderY + 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(visualization, "Current frame - all Bezier curves", cv::Point(currentOffsetX + 8, allCurvesHeaderY + 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(visualization, "Reference KF - projected map curves", cv::Point(8, projectionHeaderY + 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(visualization, "Current frame - projected map curves", cv::Point(currentOffsetX + 8, projectionHeaderY + 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(visualization, currentLabel.str(), cv::Point(currentOffsetX + 8, 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            EnsureOutputDirectory();
            std::ostringstream outputPath;
            outputPath << kBezierMatchOutputDirectory
                       << "/bezier_matches_kf_" << pKF->mnId
                       << "_frame_" << currentFrame.mnId << ".png";
            try
            {
                if (!cv::imwrite(outputPath.str(), visualization))
                    std::cerr << "Could not save Bezier match visualization to "
                              << outputPath.str() << std::endl;

                const char *display = std::getenv("DISPLAY");
                if (display && display[0] != '\0')
                {
                    cv::namedWindow(kBezierMatchWindow, cv::WINDOW_NORMAL);
                    cv::imshow(kBezierMatchWindow, visualization);
                    cv::waitKey(1);
                }
            }
            catch (const cv::Exception &exception)
            {
                std::cerr << "Bezier match visualization failed: "
                          << exception.what() << std::endl;
            }
        }

        void VisualizeBezierMatches(const Frame &lastFrame, const Frame &currentFrame)
        {
            if (currentFrame.mImgLeft.empty())
                return;

            const cv::Size imageSize = currentFrame.mImgLeft.size();
            cv::Mat lastImage = PrepareFrameImage(lastFrame.mImgLeft, imageSize);
            cv::Mat currentImage = PrepareFrameImage(currentFrame.mImgLeft, imageSize);

            struct MatchGroup
            {
                cv::Scalar color;
            };

            std::unordered_map<MapCurve *, MatchGroup> matchGroups;
            size_t fragmentMatchCount = 0;

            const size_t currentCount = std::min(currentFrame.mvBezierCurves.size(), currentFrame.mvpMapBeziers.size());
            const size_t lastCount = std::min(lastFrame.mvBezierCurves.size(), lastFrame.mvpMapBeziers.size());

            for (size_t currentIdx = 0; currentIdx < currentCount; ++currentIdx)
            {
                MapCurve *pMapBezier = currentFrame.mvpMapBeziers[currentIdx];
                if (!pMapBezier)
                    continue;

                auto groupIt = matchGroups.find(pMapBezier);
                if (groupIt == matchGroups.end())
                {
                    const cv::Scalar color = MapCurveColor(pMapBezier);
                    bool hasLastCurve = false;

                    // The same map edge may already be represented by several
                    // fragments in the last frame. Draw all of them in one color.
                    for (size_t lastIdx = 0; lastIdx < lastCount; ++lastIdx)
                    {
                        if (lastFrame.mvpMapBeziers[lastIdx] != pMapBezier)
                            continue;

                        const BezierCurve &lastCurve = lastFrame.mvBezierCurves[lastIdx];
                        DrawCurve(lastImage, lastCurve, color);
                        hasLastCurve = hasLastCurve || !lastCurve.sampledPoints.empty();
                    }

                    if (!hasLastCurve)
                        continue;

                    groupIt = matchGroups.emplace(pMapBezier, MatchGroup{color}).first;
                }

                const MatchGroup &group = groupIt->second;
                const BezierCurve &currentCurve = currentFrame.mvBezierCurves[currentIdx];
                DrawCurve(currentImage, currentCurve, group.color);

                ++fragmentMatchCount;
            }

            if (matchGroups.empty())
                return;

            cv::Mat lastAllCurves = PrepareFrameImage(lastFrame.mImgLeft, imageSize);
            cv::Mat currentAllCurves = PrepareFrameImage(currentFrame.mImgLeft, imageSize);
            for (size_t curveIdx = 0;
                 curveIdx < lastFrame.mvBezierCurves.size(); ++curveIdx)
            {
                DrawCurve(lastAllCurves, lastFrame.mvBezierCurves[curveIdx], MatchColor(curveIdx));
            }
            for (size_t curveIdx = 0;
                 curveIdx < currentFrame.mvBezierCurves.size(); ++curveIdx)
            {
                DrawCurve(currentAllCurves, currentFrame.mvBezierCurves[curveIdx], MatchColor(curveIdx));
            }

            cv::Mat lastProjectedMapCurves = PrepareFrameImage(lastFrame.mImgLeft, imageSize);
            cv::Mat currentProjectedMapCurves = PrepareFrameImage(currentFrame.mImgLeft, imageSize);
            DrawProjectedMapCurves(lastProjectedMapCurves, lastFrame.mvpMapBeziers,
                                   lastFrame.GetPose(), lastFrame.mpCamera);
            DrawProjectedMapCurves(currentProjectedMapCurves, currentFrame.mvpMapBeziers,
                                   currentFrame.GetPose(), currentFrame.mpCamera);

            const int headerHeight = 34;
            const int panelGap = 24;
            const int rowGap = 18;
            const int currentOffsetX = imageSize.width + panelGap;
            const int allCurvesHeaderY = headerHeight + imageSize.height + rowGap;
            const int allCurvesImageY = allCurvesHeaderY + headerHeight;
            const int projectionHeaderY = allCurvesImageY + imageSize.height + rowGap;
            const int projectionImageY = projectionHeaderY + headerHeight;
            cv::Mat visualization = cv::Mat::zeros(imageSize.height * 3 + headerHeight * 3 + rowGap * 2, imageSize.width * 2 + panelGap, CV_8UC3);
            lastImage.copyTo(visualization(cv::Rect(0, headerHeight, imageSize.width, imageSize.height)));
            currentImage.copyTo(visualization(cv::Rect(currentOffsetX, headerHeight, imageSize.width, imageSize.height)));
            lastAllCurves.copyTo(visualization(cv::Rect(0, allCurvesImageY, imageSize.width, imageSize.height)));
            currentAllCurves.copyTo(visualization(cv::Rect(currentOffsetX, allCurvesImageY, imageSize.width, imageSize.height)));
            lastProjectedMapCurves.copyTo(visualization(cv::Rect(0, projectionImageY, imageSize.width, imageSize.height)));
            currentProjectedMapCurves.copyTo(visualization(cv::Rect(currentOffsetX, projectionImageY, imageSize.width, imageSize.height)));

            std::ostringstream lastLabel;
            lastLabel << "Last frame " << lastFrame.mnId
                      << " - matched Bezier curves";
            std::ostringstream currentLabel;
            currentLabel << "Current frame " << currentFrame.mnId
                         << " - matched Bezier curves"
                         << " | map edges: " << matchGroups.size()
                         << " | fragments: " << fragmentMatchCount;
            cv::putText(visualization, lastLabel.str(), cv::Point(8, 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(visualization, "Last frame - all Bezier curves", cv::Point(8, allCurvesHeaderY + 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(visualization, "Current frame - all Bezier curves", cv::Point(currentOffsetX + 8, allCurvesHeaderY + 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(visualization, "Last frame - projected map curves", cv::Point(8, projectionHeaderY + 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(visualization, "Current frame - projected map curves", cv::Point(currentOffsetX + 8, projectionHeaderY + 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(visualization, currentLabel.str(), cv::Point(currentOffsetX + 8, 23), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            EnsureOutputDirectory();
            std::ostringstream outputPath;
            outputPath << kBezierMatchOutputDirectory
                       << "/bezier_matches_frame_" << lastFrame.mnId
                       << "_frame_" << currentFrame.mnId << ".png";
            try
            {
                if (!cv::imwrite(outputPath.str(), visualization))
                {
                    std::cerr << "Could not save Bezier match visualization to "
                              << outputPath.str() << std::endl;
                }

                const char *display = std::getenv("DISPLAY");
                if (display && display[0] != '\0')
                {
                    cv::namedWindow(kBezierMatchWindow, cv::WINDOW_NORMAL);
                    cv::imshow(kBezierMatchWindow, visualization);
                    cv::waitKey(1);
                }
            }
            catch (const cv::Exception &exception)
            {
                std::cerr << "Frame-to-frame Bezier visualization failed: "
                          << exception.what() << std::endl;
            }
        }
    } // namespace

    int BezierMatcher::SearchByProjection(Frame &currentFrame, const Frame &lastFrame)
    {
        if (currentFrame.NC <= 0 || currentFrame.mvBezierCurves.empty() || lastFrame.mvpMapBeziers.empty())
            return 0;

        if (currentFrame.mvpMapBeziers.size() != static_cast<size_t>(currentFrame.NC))
            currentFrame.mvpMapBeziers.resize(
                currentFrame.NC, static_cast<MapCurve *>(NULL));

        const int nMatches = MatchCurvesWithHungarian(currentFrame, lastFrame.mvpMapBeziers, currentFrame.mvpMapBeziers);

        if (nMatches > 0)
            VisualizeBezierMatches(lastFrame, currentFrame);

        return nMatches;
    }

    int BezierMatcher::SearchByProjection(KeyFrame *pKF, Frame &currentFrame, std::vector<MapCurve *> &vpMapBezierMatches)
    {
        if (currentFrame.NC <= 0)
        {
            vpMapBezierMatches.clear();
            return 0;
        }

        vpMapBezierMatches = vector<MapCurve *>(currentFrame.NC, static_cast<MapCurve *>(NULL));
        if (!pKF || pKF->isBad() || currentFrame.mvBezierCurves.empty())
            return 0;

        int nMatches = 0;
        // 参考帧中所包含的地图边缘
        const vector<MapCurve *> vpMapBeziersKF = pKF->GetMapCurveMatches();
        // 当前帧中的地图边缘匹配结果: 当前帧的第i条Bezier曲线对应的地图边缘, 如果没有匹配则为nullptr
        nMatches = MatchCurvesWithHungarian(currentFrame, vpMapBeziersKF, vpMapBezierMatches);

        if (nMatches > 0)
            VisualizeBezierMatches(pKF, currentFrame, vpMapBezierMatches);

        return nMatches;
    }

    int BezierMatcher::SearchByProjection(KeyFrame *pReferenceKF, KeyFrame *pTargetKF, std::vector<MapCurve *> &vpMapBezierMatches)
    {
        if (!pReferenceKF || !pTargetKF || pReferenceKF->isBad() || pTargetKF->isBad())
            return 0;

        const vector<MapCurve *> vpReferenceBeziers = pReferenceKF->GetMapCurveMatches();
        const vector<MapCurve *> vpTargetBeziers = pTargetKF->GetMapCurveMatches();
        const vector<BezierCurve> &vTargetCurves = pTargetKF->mvBezierCurves;

        vpMapBezierMatches = vector<MapCurve *>(vTargetCurves.size(), static_cast<MapCurve *>(NULL));
        if (vpReferenceBeziers.empty() || vTargetCurves.empty())
            return 0;

        const Sophus::SE3f TcwTarget = pTargetKF->GetPose();
        const float radius = 6.0f;
        const float radius2 = radius * radius;

        vector<int> bestVotes(vTargetCurves.size(), -1);
        vector<bool> fixedMatches(vTargetCurves.size(), false);
        for (size_t i = 0; i < vTargetCurves.size() && i < vpTargetBeziers.size(); ++i)
            fixedMatches[i] = (vpTargetBeziers[i] != static_cast<MapCurve *>(NULL));

        int nMatches = 0;
        for (size_t refIdx = 0; refIdx < vpReferenceBeziers.size(); ++refIdx)
        {
            MapCurve *pMapBezier = vpReferenceBeziers[refIdx];
            if (!pMapBezier || pMapBezier->isBad())
                continue;

            const vector<Eigen::Vector3f> worldPoints = pMapBezier->GetWorldPoints();
            if (worldPoints.size() < 2)
                continue;

            map<size_t, int> curveVoteMap;
            int validProjectedPoints = 0;

            for (const Eigen::Vector3f &worldPoint : worldPoints)
            {
                const Eigen::Vector3f cameraPoint = TcwTarget * worldPoint;
                if (cameraPoint.z() <= 0.0f)
                    continue;

                const Eigen::Vector2f uv = pTargetKF->mpCamera->project(cameraPoint);
                if (!uv.allFinite() || !pTargetKF->IsInImage(uv.x(), uv.y()))
                    continue;

                ++validProjectedPoints;

                size_t bestCurveIdx = vTargetCurves.size();
                float bestDist2 = radius2;
                for (size_t curveIdx = 0; curveIdx < vTargetCurves.size(); ++curveIdx)
                {
                    if (fixedMatches[curveIdx])
                        continue;

                    const BezierCurve &curve = vTargetCurves[curveIdx];
                    for (const Eigen::Vector2f &sample : curve.sampledPoints)
                    {
                        const float dist2 = (sample - uv).squaredNorm();
                        if (dist2 < bestDist2)
                        {
                            bestDist2 = dist2;
                            bestCurveIdx = curveIdx;
                        }
                    }
                }

                if (bestCurveIdx < vTargetCurves.size())
                    ++curveVoteMap[bestCurveIdx];
            }

            if (curveVoteMap.empty() || validProjectedPoints < 2)
                continue;

            const int minVotes = std::max(2, std::min(static_cast<int>(0.3f * worldPoints.size()), 8));
            for (map<size_t, int>::const_iterator mit = curveVoteMap.begin(), mend = curveVoteMap.end(); mit != mend; ++mit)
            {
                const size_t targetIdx = mit->first;
                const int votes = mit->second;
                const float voteRatio = static_cast<float>(votes) / static_cast<float>(validProjectedPoints);

                if (votes < minVotes || voteRatio < 0.3f)
                    continue;

                if (votes > bestVotes[targetIdx])
                {
                    if (!vpMapBezierMatches[targetIdx])
                        ++nMatches;

                    bestVotes[targetIdx] = votes;
                    vpMapBezierMatches[targetIdx] = pMapBezier;
                }
            }
        }

        return nMatches;
    }

    int BezierMatcher::SearchByProjection(Frame &F, const std::vector<MapCurve *> &vpMapBeziers)
    {
        if (F.NC <= 0 || F.mvBezierCurves.empty())
            return 0;

        if (F.mvpMapBeziers.size() != static_cast<size_t>(F.NC))
            F.mvpMapBeziers.resize(F.NC, static_cast<MapCurve *>(NULL));

        std::vector<MapCurve *> visibleMapBeziers;
        visibleMapBeziers.reserve(vpMapBeziers.size());
        for (MapCurve *pMapBezier : vpMapBeziers)
        {
            if (pMapBezier && !pMapBezier->isBad() && pMapBezier->mbTrackInView)
                visibleMapBeziers.push_back(pMapBezier);
        }

        return MatchCurvesWithHungarian(F, visibleMapBeziers, F.mvpMapBeziers);
    }

    int BezierMatcher::Fuse(KeyFrame *targetKF, const std::vector<MapCurve *> &vpMapBeziers)
    {
        if (!targetKF || targetKF->isBad() || vpMapBeziers.empty())
            return 0;

        const vector<BezierCurve> &targetCurves = targetKF->mvBezierCurves;
        if (targetCurves.empty())
            return 0;

        const Sophus::SE3f Tcw = targetKF->GetPose();
        const float radius = 6.0f;
        const float radius2 = radius * radius;
        const float maxMeanDist2 = 25.0f;

        int nFused = 0;
        std::set<MapCurve *> spProcessedBeziers;

        for (MapCurve *pMB : vpMapBeziers)
        {
            if (!pMB || pMB->isBad() || pMB->GetMap() != targetKF->GetMap())
                continue;

            if (!spProcessedBeziers.insert(pMB).second)
                continue;

            const vector<Eigen::Vector3f> worldPoints = pMB->GetWorldPoints();
            if (worldPoints.size() < 2)
                continue;

            map<size_t, CurveCandidate> candidates;
            int validProjectedPoints = 0;

            for (size_t mapIdx = 0; mapIdx < worldPoints.size(); ++mapIdx)
            {
                const Eigen::Vector3f cameraPoint = Tcw * worldPoints[mapIdx];
                if (cameraPoint.z() <= 0.0f)
                    continue;

                const Eigen::Vector2f uv = targetKF->mpCamera->project(cameraPoint);
                if (!uv.allFinite() || !targetKF->IsInImage(uv.x(), uv.y()))
                    continue;

                ++validProjectedPoints;

                size_t bestCurveIdx = targetCurves.size();
                int bestSampleIdx = -1;
                float bestDist2 = radius2;

                for (size_t curveIdx = 0; curveIdx < targetCurves.size(); ++curveIdx)
                {
                    const BezierCurve &curve = targetCurves[curveIdx];
                    for (size_t sampleIdx = 0; sampleIdx < curve.sampledPoints.size(); ++sampleIdx)
                    {
                        const float dist2 = (curve.sampledPoints[sampleIdx] - uv).squaredNorm();
                        if (dist2 < bestDist2)
                        {
                            bestDist2 = dist2;
                            bestCurveIdx = curveIdx;
                            bestSampleIdx = static_cast<int>(sampleIdx);
                        }
                    }
                }

                if (bestCurveIdx >= targetCurves.size())
                    continue;

                CurveCandidate &candidate = candidates[bestCurveIdx];
                candidate.votes++;
                candidate.dist2Sum += bestDist2;
                candidate.minMapIdx = std::min(candidate.minMapIdx, static_cast<int>(mapIdx));
                candidate.maxMapIdx = std::max(candidate.maxMapIdx, static_cast<int>(mapIdx));
                candidate.targetSampleIdxs.push_back(bestSampleIdx);
            }

            if (validProjectedPoints < 2 || candidates.empty())
                continue;

            const int minVotes = std::max(3, std::min(static_cast<int>(0.1f * worldPoints.size()), 8));
            vector<AcceptedCandidate> acceptedCandidates;

            for (map<size_t, CurveCandidate>::iterator mit = candidates.begin(), mend = candidates.end(); mit != mend; ++mit)
            {
                const size_t targetIdx = mit->first;
                CurveCandidate &candidate = mit->second;
                const BezierCurve &targetCurve = targetCurves[targetIdx];

                if (targetCurve.sampledPoints.size() < 2 || candidate.votes < minVotes)
                    continue;

                sort(candidate.targetSampleIdxs.begin(), candidate.targetSampleIdxs.end());
                candidate.targetSampleIdxs.erase(unique(candidate.targetSampleIdxs.begin(), candidate.targetSampleIdxs.end()), candidate.targetSampleIdxs.end());

                const float meanDist2 = candidate.dist2Sum / static_cast<float>(candidate.votes);
                const float mapCoverage = static_cast<float>(candidate.votes) / static_cast<float>(validProjectedPoints);
                const float targetCoverage = static_cast<float>(candidate.targetSampleIdxs.size()) / static_cast<float>(targetCurve.sampledPoints.size());
                const float mapSpanCoverage = static_cast<float>(candidate.maxMapIdx - candidate.minMapIdx + 1) / static_cast<float>(worldPoints.size());

                if (meanDist2 > maxMeanDist2)
                    continue;

                // A long map curve may be split into short observed curves, so
                // target coverage is enough for adding a segment observation.
                if (targetCoverage < 0.35f && mapCoverage < 0.20f)
                    continue;

                acceptedCandidates.push_back({targetIdx, candidate.votes, meanDist2, mapCoverage, targetCoverage, mapSpanCoverage});
            }

            sort(acceptedCandidates.begin(), acceptedCandidates.end(), [](const AcceptedCandidate &lhs, const AcceptedCandidate &rhs)
                 {
                     const float lhsScore = lhs.targetCoverage * lhs.votes / std::max(lhs.meanDist2, 1.0f);
                     const float rhsScore = rhs.targetCoverage * rhs.votes / std::max(rhs.meanDist2, 1.0f);
                     return lhsScore > rhsScore;
                 });

            for (const AcceptedCandidate &candidate : acceptedCandidates)
            {
                MapCurve *pExistingMB = targetKF->GetMapCurve(candidate.targetIdx);

                if (!pExistingMB)
                {
                    targetKF->AddMapCurve(pMB, candidate.targetIdx);
                    if (!pMB->IsInKeyFrame(targetKF))
                        pMB->AddObservation(targetKF, static_cast<int>(candidate.targetIdx));
                    nFused++;
                    continue;
                }

                if (pExistingMB == pMB || pExistingMB->isBad())
                    continue;

                const bool strongOverlap = candidate.mapSpanCoverage > 0.55f && candidate.targetCoverage > 0.55f;
                if (!strongOverlap)
                    continue;

                if (pExistingMB->Observations() >= pMB->Observations())
                    pMB->Replace(pExistingMB);
                else
                    pExistingMB->Replace(pMB);

                nFused++;
            }
        }

        return nFused;
    }
} // namespace ORB_SLAM3
