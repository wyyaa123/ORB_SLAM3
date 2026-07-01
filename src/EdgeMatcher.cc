#include "EdgeMatcher.h"

namespace ORB_SLAM3
{
    BezierMatcher::BezierMatcher()
    {
    }

    int BezierMatcher::SearchByProjection(Frame &currentFrame, const std::vector<MapBezier *> &candidates)
    {
        return 0;
    }

    int BezierMatcher::SearchFromLastFrame(Frame &currentFrame, const Frame &lastFrame)
    {
        return 0;
    }

    int BezierMatcher::Fuse(KeyFrame *targetKF, const std::vector<MapBezier *> &candidates)
    {
        return 0;
    }
} // namespace ORB_SLAM3
