/**
 * This file adds a map-level Bezier edge landmark for ORB-SLAM3.
 *
 * A MapEdge is analogous to MapPoint/MapLine, but stores a Bezier curve in
 * world coordinates through its 3D control points.
 */
#ifndef MAPEDGE_H
#define MAPEDGE_H

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "bezierCurve.h"

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <map>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <vector>

namespace ORB_SLAM3
{

    class Frame;
    class KeyFrame;
    class Map;

    typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> EdgeControlPoints;

    class MapEdge
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        MapEdge();
        MapEdge(const EdgeControlPoints &controlPoints, KeyFrame *pRefKF, Map *pMap, int cls = -1);
        MapEdge(const EdgeControlPoints &controlPoints, Map *pMap, Frame *pFrame, const int &idxF);
        MapEdge(KeyFrame *pRefKF, Map *pMap, const int &idxKF);
        MapEdge(Map *pMap, Frame *pFrame, const int &idxF);

        void SetWorldControlPoints(const EdgeControlPoints &controlPoints);
        EdgeControlPoints GetWorldControlPoints();

        void SetWorldSampledPoints(const EdgeControlPoints &sampledPoints);
        EdgeControlPoints GetWorldSampledPoints();

        Eigen::Vector3f GetCenter();
        Eigen::Vector3f GetNormal();
        void SetNormalVector(const Eigen::Vector3f &normal);

        int GetSemanticClass();
        int GetCurveOrder();

        KeyFrame *GetReferenceKeyFrame();

        std::map<KeyFrame *, size_t> GetObservations();
        int Observations();

        void AddObservation(KeyFrame *pKF, size_t idx);
        void EraseObservation(KeyFrame *pKF);

        int GetIndexInKeyFrame(KeyFrame *pKF);
        bool IsInKeyFrame(KeyFrame *pKF);

        void SetBadFlag();
        bool isBad();

        void Replace(MapEdge *pME);
        MapEdge *GetReplaced();

        void IncreaseVisible(int n = 1);
        void IncreaseFound(int n = 1);
        float GetFoundRatio();
        inline int GetFound()
        {
            return mnFound;
        }

        void FuseWorldGeometry(const EdgeControlPoints &controlPoints, const EdgeControlPoints &sampledPoints);
        void UpdateAverageDirAndDepth();

        float GetMinDistanceInvariance();
        float GetMaxDistanceInvariance();
        int PredictScale(const float &currentDist, KeyFrame *pKF);
        int PredictScale(const float &currentDist, Frame *pF);

        Map *GetMap();
        void UpdateMap(Map *pMap);

    public:
        long unsigned int mnId;
        static long unsigned int nNextId;
        long int mnFirstKFid;
        long int mnFirstFrame;
        int nObs;

        // Variables used by tracking.
        std::vector<cv::Point2f> mvTrackProjections;
        bool mbTrackInView;
        int mnTrackScaleLevel;
        float mTrackViewCos;
        long unsigned int mnTrackReferenceForFrame;
        long unsigned int mnLastFrameSeen;

        // Variables used by local mapping.
        long unsigned int mnBALocalForKF;
        long unsigned int mnFuseCandidateForKF;

        // Variables used by loop closing / global BA.
        long unsigned int mnLoopEdgeForKF;
        long unsigned int mnCorrectedByKF;
        long unsigned int mnCorrectedReference;
        EdgeControlPoints mvControlPointsGBA;
        long unsigned int mnBAGlobalForKF;

        static std::mutex mGlobalMutex;

    protected:
        EdgeControlPoints mvWorldControlPoints;
        EdgeControlPoints mvWorldSampledPoints;

        std::map<KeyFrame *, size_t> mObservations;

        Eigen::Vector3f mNormalVector;

        KeyFrame *mpRefKF;

        int mnSemanticClass;
        int mnCurveOrder;

        int mnVisible;
        int mnFound;

        bool mbBad;
        MapEdge *mpReplaced;

        float mfMinDistance;
        float mfMaxDistance;

        Map *mpMap;

        std::mutex mMutexPos;
        std::mutex mMutexFeatures;
        std::mutex mMutexMap;
    };

} // namespace ORB_SLAM3

#endif // MAPEDGE_H
