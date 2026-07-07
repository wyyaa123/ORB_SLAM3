#include "EdgeMatcher.h"

#include <algorithm>
#include <cerrno>
#include <iostream>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <unordered_map>

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

    int BezierMatcher::SearchByProjection(KeyFrame *pKF, Frame &currentFrame, std::vector<MapBezier *> &vpMapBezierMatches)
    {
        int nMatches = 0;
        // 参考帧中所包含的地图边缘
        const vector<MapBezier *> vpMapBeziersKF = pKF->GetMapBezierMatches();
        // 当前帧中的地图边缘匹配结果: 当前帧的第i条Bezier曲线对应的地图边缘, 如果没有匹配则为nullptr
        vpMapBezierMatches = vector<MapBezier *>(currentFrame.NB, static_cast<MapBezier *>(NULL));

        // MapBezier stores points in world coordinates, so project them with Tcw.
        const Sophus::SE3f Tcw = currentFrame.GetPose();
        const float radius = 6.0f;
        std::vector<int> bestVotes(currentFrame.NB, -1);

        assert(vpMapBeziersKF.size() == pKF->mvBezierCurves.size());

        for (size_t i = 0; i < vpMapBeziersKF.size(); ++i)
        {
            //  参考帧中的地图边缘
            MapBezier *pMapBezier = vpMapBeziersKF[i];
            if (!pMapBezier || pMapBezier->isBad() || i >= pKF->mvBezierCurves.size())
                continue;

            // 参考帧中的地图边缘所对应的点
            std::vector<Eigen::Vector3f> worldPoints = pMapBezier->GetWorldPoints();
            assert(pKF->mvBezierCurves[i].sampledPoints.size() == worldPoints.size());
            const size_t numPoints = worldPoints.size();
            if (numPoints == 0)
                continue;

            std::map<int, int> edgeVoteMapTotal;
            const int thresholdValue = std::min(static_cast<int>(numPoints * 0.3f), 5);

            for (size_t j = 0; j < numPoints; ++j)
            {
                const Eigen::Vector3f currentPoint = Tcw * worldPoints[j];
                if (currentPoint.z() <= 0.0f)
                    continue;

                const float u = currentFrame.fx * currentPoint.x() / currentPoint.z() + currentFrame.cx;
                const float v = currentFrame.fy * currentPoint.y() / currentPoint.z() + currentFrame.cy;
                if (u < 0.0f || u >= currentFrame.mImgLeft.cols || v < 0.0f || v >= currentFrame.mImgLeft.rows)
                    continue;

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
                if (validEdgeIds.count(edgeId) != 0 && voteIt != edgeVoteMapTotal.end() && voteIt->second > bestVotes[curveIdx])
                {
                    bestVotes[curveIdx] = voteIt->second;
                    vpMapBezierMatches[curveIdx] = pMapBezier;
                    ++nMatches;
                }
            }
        }

        return nMatches;
    }

    int BezierMatcher::SearchByProjection(KeyFrame *pReferenceKF, KeyFrame *pTargetKF, std::vector<MapBezier *> &vpMapBezierMatches)
    {
        if (!pReferenceKF || !pTargetKF || pReferenceKF->isBad() || pTargetKF->isBad())
            return 0;

        const vector<MapBezier *> vpReferenceBeziers = pReferenceKF->GetMapBezierMatches();
        const vector<MapBezier *> vpTargetBeziers = pTargetKF->GetMapBezierMatches();
        const vector<BezierCurve> &vTargetCurves = pTargetKF->mvBezierCurves;

        vpMapBezierMatches = vector<MapBezier *>(vTargetCurves.size(), static_cast<MapBezier *>(NULL));
        if (vpReferenceBeziers.empty() || vTargetCurves.empty())
            return 0;

        const Sophus::SE3f TcwTarget = pTargetKF->GetPose();
        const float radius = 6.0f;
        const float radius2 = radius * radius;

        vector<int> bestVotes(vTargetCurves.size(), -1);
        vector<bool> fixedMatches(vTargetCurves.size(), false);
        for (size_t i = 0; i < vTargetCurves.size() && i < vpTargetBeziers.size(); ++i)
            fixedMatches[i] = (vpTargetBeziers[i] != static_cast<MapBezier *>(NULL));

        int nMatches = 0;
        for (size_t refIdx = 0; refIdx < vpReferenceBeziers.size(); ++refIdx)
        {
            MapBezier *pMapBezier = vpReferenceBeziers[refIdx];
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

    int BezierMatcher::SearchByProjection(Frame &F, const std::vector<MapBezier *> &vpMapBeziers)
    {
        if (F.NB <= 0 || F.mvBezierCurves.empty())
            return 0;

        if (F.mvpMapBeziers.size() != static_cast<size_t>(F.NB))
            F.mvpMapBeziers.resize(F.NB, static_cast<MapBezier *>(NULL));

        const Sophus::SE3f Tcw = F.GetPose();
        const float radius = 6.0f;

        std::vector<int> bestVotes(F.NB, -1);
        std::vector<bool> fixedMatches(F.NB, false);
        for (int i = 0; i < F.NB; ++i)
            fixedMatches[i] = (F.mvpMapBeziers[i] != static_cast<MapBezier *>(NULL));

        int nMatches = 0;
        for (MapBezier *pMapBezier : vpMapBeziers)
        {
            if (!pMapBezier || pMapBezier->isBad() || !pMapBezier->mbTrackInView)
                continue;

            const std::vector<Eigen::Vector3f> worldPoints = pMapBezier->GetWorldPoints();
            const size_t numPoints = worldPoints.size();
            if (numPoints < 2)
                continue;

            std::map<int, int> edgeVoteMapTotal;
            const int thresholdValue = std::min(static_cast<int>(numPoints * 0.3f), 5);

            for (const Eigen::Vector3f &worldPoint : worldPoints)
            {
                const Eigen::Vector3f currentPoint = Tcw * worldPoint;
                if (currentPoint.z() <= 0.0f)
                    continue;

                const Eigen::Vector2f uv = F.mpCamera->project(currentPoint);
                if (uv.x() < F.mnMinX || uv.x() > F.mnMaxX ||
                    uv.y() < F.mnMinY || uv.y() > F.mnMaxY)
                {
                    continue;
                }

                std::vector<orderedEdgePoint> neighborsPoints;
                F.searchRadius(uv.x(), uv.y(), radius, neighborsPoints);

                std::unordered_map<int, int> edgeVoteMap;
                for (const orderedEdgePoint &neighbor : neighborsPoints)
                    ++edgeVoteMap[neighbor.frame_edge_ID];

                if (edgeVoteMap.empty())
                    continue;

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

            if (edgeVoteMapTotal.empty())
                continue;

            std::unordered_set<int> validEdgeIds;
            for (const auto &[edgeId, votes] : edgeVoteMapTotal)
            {
                if (votes > thresholdValue)
                    validEdgeIds.insert(edgeId);
            }

            if (validEdgeIds.empty())
                continue;

            for (size_t curveIdx = 0; curveIdx < F.mvBezierCurves.size() && curveIdx < F.NB; ++curveIdx)
            {
                if (fixedMatches[curveIdx])
                    continue;

                const int edgeId = F.mvBezierCurves[curveIdx].edge_ID;
                const auto voteIt = edgeVoteMapTotal.find(edgeId);
                if (validEdgeIds.count(edgeId) == 0 || voteIt == edgeVoteMapTotal.end())
                    continue;

                if (voteIt->second > bestVotes[curveIdx])
                {
                    if (bestVotes[curveIdx] < 0)
                        ++nMatches;

                    bestVotes[curveIdx] = voteIt->second;
                    F.mvpMapBeziers[curveIdx] = pMapBezier;
                }
            }
        }

        return nMatches;
    }

    int BezierMatcher::Fuse(KeyFrame *targetKF, const std::vector<MapBezier *> &vpMapBeziers)
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
        std::set<MapBezier *> spProcessedBeziers;

        for (MapBezier *pMB : vpMapBeziers)
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
                candidate.targetSampleIdxs.erase(unique(candidate.targetSampleIdxs.begin(), candidate.targetSampleIdxs.end()),
                                                 candidate.targetSampleIdxs.end());

                const float meanDist2 = candidate.dist2Sum / static_cast<float>(candidate.votes);
                const float mapCoverage = static_cast<float>(candidate.votes) / static_cast<float>(validProjectedPoints);
                const float targetCoverage = static_cast<float>(candidate.targetSampleIdxs.size()) /
                                             static_cast<float>(targetCurve.sampledPoints.size());
                const float mapSpanCoverage = static_cast<float>(candidate.maxMapIdx - candidate.minMapIdx + 1) /
                                              static_cast<float>(worldPoints.size());

                if (meanDist2 > maxMeanDist2)
                    continue;

                // A long map curve may be split into short observed curves, so
                // target coverage is enough for adding a segment observation.
                if (targetCoverage < 0.35f && mapCoverage < 0.20f)
                    continue;

                acceptedCandidates.push_back({targetIdx, candidate.votes, meanDist2,
                                              mapCoverage, targetCoverage, mapSpanCoverage});
            }

            sort(acceptedCandidates.begin(), acceptedCandidates.end(),
                 [](const AcceptedCandidate &lhs, const AcceptedCandidate &rhs)
                 {
                     const float lhsScore = lhs.targetCoverage * lhs.votes / std::max(lhs.meanDist2, 1.0f);
                     const float rhsScore = rhs.targetCoverage * rhs.votes / std::max(rhs.meanDist2, 1.0f);
                     return lhsScore > rhsScore;
                 });

            for (const AcceptedCandidate &candidate : acceptedCandidates)
            {
                MapBezier *pExistingMB = targetKF->GetMapBezier(candidate.targetIdx);

                if (!pExistingMB)
                {
                    targetKF->AddMapBezier(pMB, candidate.targetIdx);
                    if (!pMB->IsInKeyFrame(targetKF))
                        pMB->AddObservation(targetKF, static_cast<int>(candidate.targetIdx));
                    nFused++;
                    continue;
                }

                if (pExistingMB == pMB || pExistingMB->isBad())
                    continue;

                const bool strongOverlap =
                    candidate.mapSpanCoverage > 0.55f && candidate.targetCoverage > 0.55f;
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
