#ifndef EDGEMATCHER_H
#define EDGEMATCHER_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "MapCurve.h"

#include <unordered_set>

namespace ORB_SLAM3
{
    class KeyFrame;
    class Map;
    class Frame;
    class MapCurve;

    struct CurveCandidate
    {
        int votes = 0;
        float dist2Sum = 0.0f;
        int minMapIdx = DBL_MAX;
        int maxMapIdx = -1;
        vector<int> targetSampleIdxs;
    };

    struct AcceptedCandidate
    {
        size_t targetIdx = 0;
        int votes = 0;
        float meanDist2 = 0.0f;
        float mapCoverage = 0.0f;
        float targetCoverage = 0.0f;
        float mapSpanCoverage = 0.0f;
    };

    class BezierMatcher
    {
    public:
        BezierMatcher() {};

        int SearchByProjection(Frame &currentFrame, const Frame &lastFrame);

        int SearchByProjection(KeyFrame *pKF, Frame &currentFrame, std::vector<MapCurve *> &vpMapBezierMatches);

        int SearchByProjection(KeyFrame *pReferenceKF, KeyFrame *pTargetKF, std::vector<MapCurve *> &vpMapBezierMatches);

        int SearchByProjection(Frame &F, const std::vector<MapCurve *> &vpMapBeziers);

        int SearchByProjection(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapCurve *> &vpBeziers, std::vector<MapCurve *> &vpMatched);

        int Fuse(KeyFrame *targetKF, const std::vector<MapCurve *> &vpMapBeziers);

        int Fuse(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapCurve *> &vpBeziers, float th, vector<MapCurve *> &vpReplaceBeziers);
    };
} // namespace ORB_SLAM3

#endif // EDGEMATCHER_H
