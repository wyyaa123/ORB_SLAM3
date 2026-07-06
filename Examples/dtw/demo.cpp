#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

extern "C"
{
#include "dd_dtw.h"
}

void compareSequences(const char *firstName,
                      std::vector<double> &first,
                      const char *secondName,
                      std::vector<double> &second,
                      DTWSettings &settings)
{
    const idx_t firstLength = static_cast<idx_t>(first.size());
    const idx_t secondLength = static_cast<idx_t>(second.size());
    const idx_t maxPathLength = firstLength + secondLength;

    std::vector<idx_t> firstIndices(maxPathLength);
    std::vector<idx_t> secondIndices(maxPathLength);
    idx_t pathLength = 0;

    const double distance = dtw_warping_path(
        first.data(), firstLength,
        second.data(), secondLength,
        firstIndices.data(), secondIndices.data(),
        &pathLength, &settings);

    firstIndices.resize(pathLength);
    secondIndices.resize(pathLength);

    // DTAIDistanceC 返回的路径方向是终点到起点。
    std::reverse(firstIndices.begin(), firstIndices.end());
    std::reverse(secondIndices.begin(), secondIndices.end());

    std::cout << firstName << " <-> " << secondName << '\n'
              << "  DTW distance: " << distance << '\n'
              << "  path length: " << pathLength << '\n'
              << "  RMS distance: "
              << distance / std::sqrt(static_cast<double>(pathLength))
              << "\n  path:";

    for (idx_t k = 0; k < pathLength; ++k)
    {
        std::cout << " (" << firstIndices[k]
                  << ", " << secondIndices[k] << ')';
    }
    std::cout << "\n\n";
}

int main()
{

    double s1[] = {0, 0, 0, 1, 2, 1, 0, 1, 0, 0};
    double s2[] = {0, 0, 2, 1, 0, 1, 0, .5, 0, 0};
    DTWSettings settings = dtw_settings_default();
    seq_t d = dtw_distance_ndim(s1, 5, s2, 5, 2, &settings);

    std::vector<double> s1 = {0, 0.1, 2, 3, 4};
    std::vector<double> s2 = {1, 2, 3, 4, 5, 6, 7};
    std::vector<double> s3 = {0, 1, 4, 9};

    DTWSettings settings = dtw_settings_default();

    compareSequences("s1", s1, "s2", s2, settings);
    compareSequences("s1", s1, "s3", s3, settings);
    compareSequences("s2", s2, "s3", s3, settings);

    return 0;
}
