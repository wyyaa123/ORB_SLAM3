#ifndef EDGEMATCHER_H
#define EDGEMATCHER_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include "MapBezier.h"

namespace ORB_SLAM3
{
    class KeyFrame;
    class Map;
    class Frame;
    class MapBezier;

    class BezierMatcher
    {
    public:
        BezierMatcher();

        int SearchByProjection(
            Frame &currentFrame,
            const std::vector<MapBezier *> &candidates);

        int SearchFromLastFrame(
            Frame &currentFrame,
            const Frame &lastFrame);

        int Fuse(
            KeyFrame *targetKF,
            const std::vector<MapBezier *> &candidates);
    };
} // namespace ORB_SLAM3

#endif // EDGEMATCHER_H