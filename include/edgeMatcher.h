#ifndef EDGE_MATCHER_H
#define EDGE_MATCHER_H

#include <edgeExtracter.h>
#include "Frame.h"
#include "KeyFrame.h"
#include <opencv2/opencv.hpp>
#include <utility>

namespace ORB_SLAM3
{
    class EdgeMatcher
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeMatcher(float _radius = 5.0, bool enableVisualization = false)
            : mradius(_radius), mbEnableVisualization(enableVisualization) {}

        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const Sophus::SE3f &Tcl);

        /**
         * Project curves from pKF2 into pKF1 and find all supported curve pairs.
         *
         * matches12 is indexed by the curve index in pKF2. Each map entry uses a
         * curve index in pKF1 as key and stores the pKF2 sampled-point indices
         * which support that relation. More than one entry may exist on either
         * side, so the result represents a many-to-many relation.
         *
         * @return number of accepted curve-to-curve relations.
         */
        int SearchByProjection(KeyFrame *pKF1, KeyFrame *pKF2,
                               std::vector<std::map<int, std::vector<int>>> &matches12);

        /**
         * Build two visualization images. The returned pair is ordered as
         * {previous/neighbor keyframe, current keyframe}.
         */
        std::pair<cv::Mat, cv::Mat> DrawCurveMatches(
            const KeyFrame *pCurrentKF, const KeyFrame *pPreviousKF,
            const std::vector<std::map<int, std::vector<int>>> &matches12) const;

        void SetVisualizationEnabled(bool enabled) { mbEnableVisualization = enabled; }

        void ShowCurveMatches(
            const KeyFrame *pCurrentKF, const KeyFrame *pPreviousKF,
            const std::vector<std::map<int, std::vector<int>>> &matches12) const;

    private:
        float mradius;
        bool mbEnableVisualization;
    };
}

#endif
