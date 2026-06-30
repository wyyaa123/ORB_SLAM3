#include "edgeMatcher.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace ORB_SLAM3
{
    int EdgeMatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const Sophus::SE3f &Tcl)
    {
        return 0;
    }

    // Project the curves in pKF2 into pKF1 and retain every supported pair.
    int EdgeMatcher::SearchByProjection(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<std::map<int, std::vector<int>>> &matches12)
    {

    }
}
