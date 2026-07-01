#ifndef MAPEDGE_H
#define MAPEDGE_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"
#include <mutex>

namespace ORB_SLAM3
{
    class KeyFrame;
    class Map;
    class Frame;

    class MapBezier
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        MapBezier(const std::vector<Eigen::Vector3f> &vWorldPoints,
                  KeyFrame *pRefKF,
                  Map *pMap);

        void AddObservation(KeyFrame *pKF, int idx);
        void EraseObservation(KeyFrame *pKF);

        std::map<KeyFrame *, int> GetObservations();
        int Observations();

        int GetIndexInKeyFrame(KeyFrame *pKF);
        bool IsInKeyFrame(KeyFrame *pKF);

        std::vector<Eigen::Vector3f> GetWorldPoints();
        void SetWorldPoints(const std::vector<Eigen::Vector3f> &vWorldPoints);

        bool isBad();
        void SetBadFlag();

        void Replace(MapBezier *pMB);
        MapBezier *GetReplaced();

        void IncreaseVisible(int n = 1);
        void IncreaseFound(int n = 1);
        float GetFoundRatio();

        Map *GetMap();

    public:
        static long unsigned int nNextId;
        long unsigned int mnId;

        static std::mutex mGlobalMutex;

        int nObs;

    protected:
        std::vector<Eigen::Vector3f> mvWorldPoints;
        std::map<KeyFrame *, int> mObservations;

        KeyFrame *mpRefKF;
        Map *mpMap;

        bool mbBad = false;
        MapBezier *mpReplaced;

        long int mnFirstKFid;
        long int mnFirstFrame;

        // Tracking counters
        int mnVisible;
        int mnFound;

        unsigned int mnOriginMapId;

        std::mutex mMutexFeatures;
        std::mutex mMutexPos;
        std::mutex mMutexMap;
    };
}

#endif // MAPEDGE_H