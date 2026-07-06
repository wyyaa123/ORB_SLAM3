#include "EdgeMatcher.h"

#include <algorithm>
#include <cerrno>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace ORB_SLAM3
{
    namespace
    {
        const char *kBezierMatchWindow = "Bezier edge matches";
        const char *kBezierMatchOutputDirectory = "output";

        cv::Mat DrawEdgeImage(const std::vector<Edge> &edges, const cv::Size &imageSize)
        {
            cv::Mat edgeImage = cv::Mat::zeros(imageSize, CV_8UC3);
            for (const Edge &edge : edges)
            {
                for (const orderedEdgePoint &point : edge.mvPoints)
                {
                    const cv::Point pixel(cvRound(point.x), cvRound(point.y));
                    if (pixel.x >= 0 && pixel.x < edgeImage.cols &&
                        pixel.y >= 0 && pixel.y < edgeImage.rows)
                    {
                        cv::circle(edgeImage, pixel, 1, cv::Scalar(180, 180, 180),
                                   cv::FILLED, cv::LINE_AA);
                    }
                }
            }
            return edgeImage;
        }

        bool CurveCenter(const BezierCurve &curve, cv::Point &center)
        {
            if (curve.sampledPoints.empty())
                return false;

            const Eigen::Vector2f &point =
                curve.sampledPoints[curve.sampledPoints.size() / 2];
            center = cv::Point(cvRound(point.x()), cvRound(point.y()));
            return true;
        }

        void DrawCurve(cv::Mat &image, const BezierCurve &curve,
                       const cv::Scalar &color)
        {
            if (curve.sampledPoints.empty())
                return;

            if (curve.sampledPoints.size() == 1)
            {
                cv::circle(image,
                           cv::Point(cvRound(curve.sampledPoints.front().x()),
                                     cvRound(curve.sampledPoints.front().y())),
                           2, color, cv::FILLED, cv::LINE_AA);
                return;
            }

            for (size_t i = 1; i < curve.sampledPoints.size(); ++i)
            {
                cv::Point first(cvRound(curve.sampledPoints[i - 1].x()),
                                cvRound(curve.sampledPoints[i - 1].y()));
                cv::Point second(cvRound(curve.sampledPoints[i].x()),
                                 cvRound(curve.sampledPoints[i].y()));
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

        void EnsureOutputDirectory()
        {
            if (mkdir(kBezierMatchOutputDirectory, 0755) != 0 && errno != EEXIST)
            {
                std::cerr << "Could not create Bezier match output directory '"
                          << kBezierMatchOutputDirectory << "': errno " << errno
                          << std::endl;
            }
        }

        void VisualizeBezierMatches(
            KeyFrame *pKF, const Frame &currentFrame,
            const std::vector<MapBezier *> &vpMapBezierMatches)
        {
            if (!pKF || currentFrame.mImgLeft.empty())
                return;

            const cv::Size imageSize = currentFrame.mImgLeft.size();
            cv::Mat referenceEdges = DrawEdgeImage(pKF->mvEdges, imageSize);
            cv::Mat currentEdges = DrawEdgeImage(currentFrame.mvEdges, imageSize);
            const std::vector<MapBezier *> referenceMatches =
                pKF->GetMapBezierMatches();

            struct MatchDrawing
            {
                cv::Point referenceCenter;
                cv::Point currentCenter;
                cv::Scalar color;
            };
            std::vector<MatchDrawing> matchDrawings;

            size_t matchIndex = 0;
            for (size_t currentIndex = 0;
                 currentIndex < vpMapBezierMatches.size() &&
                 currentIndex < currentFrame.mvBezierCurves.size();
                 ++currentIndex)
            {
                MapBezier *pMatch = vpMapBezierMatches[currentIndex];
                if (!pMatch)
                    continue;

                const auto referenceIt =
                    std::find(referenceMatches.begin(), referenceMatches.end(), pMatch);
                if (referenceIt == referenceMatches.end())
                    continue;

                const size_t referenceIndex =
                    static_cast<size_t>(std::distance(referenceMatches.begin(), referenceIt));
                if (referenceIndex >= pKF->mvBezierCurves.size())
                    continue;

                const cv::Scalar color = MatchColor(matchIndex++);
                const BezierCurve &referenceCurve =
                    pKF->mvBezierCurves[referenceIndex];
                const BezierCurve &currentCurve =
                    currentFrame.mvBezierCurves[currentIndex];
                DrawCurve(referenceEdges, referenceCurve, color);
                DrawCurve(currentEdges, currentCurve, color);

                cv::Point referenceCenter;
                cv::Point currentCenter;
                if (CurveCenter(referenceCurve, referenceCenter) &&
                    CurveCenter(currentCurve, currentCenter))
                {
                    matchDrawings.push_back(
                        {referenceCenter, currentCenter, color});
                }
            }

            const int headerHeight = 34;
            const int panelGap = 24;
            cv::Mat visualization = cv::Mat::zeros(
                imageSize.height + headerHeight,
                imageSize.width * 2 + panelGap, CV_8UC3);
            referenceEdges.copyTo(
                visualization(cv::Rect(0, headerHeight,
                                       imageSize.width, imageSize.height)));
            currentEdges.copyTo(
                visualization(cv::Rect(imageSize.width + panelGap, headerHeight,
                                       imageSize.width, imageSize.height)));

            const int currentOffsetX = imageSize.width + panelGap;
            for (const MatchDrawing &drawing : matchDrawings)
            {
                const cv::Point referencePoint(
                    drawing.referenceCenter.x,
                    drawing.referenceCenter.y + headerHeight);
                const cv::Point currentPoint(
                    drawing.currentCenter.x + currentOffsetX,
                    drawing.currentCenter.y + headerHeight);
                cv::line(visualization, referencePoint, currentPoint,
                         drawing.color, 1, cv::LINE_AA);
                cv::circle(visualization, referencePoint, 3, drawing.color,
                           cv::FILLED, cv::LINE_AA);
                cv::circle(visualization, currentPoint, 3, drawing.color,
                           cv::FILLED, cv::LINE_AA);
            }

            std::ostringstream referenceLabel;
            referenceLabel << "Reference KF " << pKF->mnId;
            std::ostringstream currentLabel;
            currentLabel << "Current frame " << currentFrame.mnId
                         << " | matches: " << matchIndex;
            cv::putText(visualization, referenceLabel.str(), cv::Point(8, 23),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255),
                        1, cv::LINE_AA);
            cv::putText(visualization, currentLabel.str(),
                        cv::Point(currentOffsetX + 8, 23),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255),
                        1, cv::LINE_AA);

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

                cv::namedWindow(kBezierMatchWindow, cv::WINDOW_NORMAL);
                cv::imshow(kBezierMatchWindow, visualization);
                cv::waitKey(1);
            }
            catch (const cv::Exception &exception)
            {
                std::cerr << "Bezier match visualization failed: "
                          << exception.what() << std::endl;
            }
        }
    } // namespace

    BezierMatcher::BezierMatcher()
    {
    }

    int BezierMatcher::SearchByProjection(KeyFrame *pKF, Frame &currentFrame, std::vector<MapBezier *> &vpMapBezierMatches)
    {
        const vector<MapBezier *> vpMapBeziersKF = pKF->GetMapBezierMatches();
        vpMapBezierMatches = vector<MapBezier *>(currentFrame.NB, static_cast<MapBezier *>(NULL));

        if (currentFrame.NB == 0 || currentFrame.mMatSearch.empty())
        {
            VisualizeBezierMatches(pKF, currentFrame, vpMapBezierMatches);
            return 0;
        }

        // MapBezier stores points in world coordinates, so project them with Tcw.
        const Sophus::SE3f Tcw = currentFrame.GetPose();
        const float radius = 6.0f;
        std::vector<int> bestVotes(currentFrame.NB, -1);

        assert(vpMapBeziersKF.size() == pKF->mvBezierCurves.size());

        for (size_t i = 0; i < vpMapBeziersKF.size(); ++i)
        {
            MapBezier *pMapBezier = vpMapBeziersKF[i];
            if (!pMapBezier || pMapBezier->isBad() || i >= pKF->mvBezierCurves.size())
                continue;

            std::vector<Eigen::Vector3f> worldPoints = pMapBezier->GetWorldPoints();
            const BezierCurve &queryCurve = pKF->mvBezierCurves[i];
            const size_t numPoints = std::min(worldPoints.size(), queryCurve.sampledPoints.size());
            if (numPoints == 0)
                continue;

            std::map<int, int> edgeVoteMapTotal;
            const int thresholdValue =
                std::min(static_cast<int>(numPoints * 0.3f), 5);

            for (size_t j = 0; j < numPoints; ++j)
            {
                const Eigen::Vector3f currentPoint = Tcw * worldPoints[j];
                if (currentPoint.z() <= 0.0f)
                    continue;

                const float u = currentFrame.fx * currentPoint.x() / currentPoint.z() + currentFrame.cx;
                const float v = currentFrame.fy * currentPoint.y() / currentPoint.z() + currentFrame.cy;
                if (u < 0.0f || u >= currentFrame.mImgLeft.cols ||
                    v < 0.0f || v >= currentFrame.mImgLeft.rows)
                {
                    continue;
                }

                std::vector<orderedEdgePoint> neighborsPoints;
                currentFrame.searchRadius(u, v, radius, neighborsPoints);

                std::unordered_map<int, int> edgeVoteMap;
                for (const orderedEdgePoint &neighbor : neighborsPoints)
                {
                    ++edgeVoteMap[neighbor.frame_edge_ID];
                }

                if (!edgeVoteMap.empty())
                {
                    const auto maxVote = std::max_element(
                        edgeVoteMap.begin(), edgeVoteMap.end(),
                        [](const auto &lhs, const auto &rhs)
                        {
                            if (lhs.second != rhs.second)
                                return lhs.second < rhs.second;
                            return lhs.first > rhs.first;
                        });
                    ++edgeVoteMapTotal[maxVote->first];
                }
            }

            std::unordered_set<int> validEdgeIds;
            for (const auto &[edgeId, votes] : edgeVoteMapTotal)
            {
                if (votes > thresholdValue)
                    validEdgeIds.insert(edgeId);
            }

            if (validEdgeIds.empty())
                continue;

            // The vote uses raw edge IDs; the output vector is indexed by Bezier curve.
            for (size_t curveIdx = 0; curveIdx < currentFrame.mvBezierCurves.size(); ++curveIdx)
            {
                const int edgeId = currentFrame.mvBezierCurves[curveIdx].edge_ID;
                const auto voteIt = edgeVoteMapTotal.find(edgeId);
                if (validEdgeIds.count(edgeId) != 0 &&
                    voteIt != edgeVoteMapTotal.end() &&
                    voteIt->second > bestVotes[curveIdx])
                {
                    bestVotes[curveIdx] = voteIt->second;
                    vpMapBezierMatches[curveIdx] = pMapBezier;
                }
            }
        }

        const int matchCount = static_cast<int>(std::count_if(
            vpMapBezierMatches.begin(), vpMapBezierMatches.end(),
            [](const MapBezier *pMapBezier)
            {
                return pMapBezier != nullptr;
            }));
        VisualizeBezierMatches(pKF, currentFrame, vpMapBezierMatches);
        return matchCount;
    }
} // namespace ORB_SLAM3
