#include "MapEdge.h"

#include <cmath>
#include <limits>

namespace ORB_SLAM3
{

    long unsigned int MapEdge::nNextId = 0;
    std::mutex MapEdge::mGlobalMutex;

    namespace
    {
        constexpr float kMinValidDepth = 0.02f;
        constexpr float kMaxValidDepth = 5.0f;

        bool IsValidEdgePoint3D(const orderedEdgePoint &point)
        {
            return point.depth > kMinValidDepth && point.depth < kMaxValidDepth &&
                   std::isfinite(point.x_3d) && std::isfinite(point.y_3d) && std::isfinite(point.z_3d);
        }

        Eigen::Vector3f ToCameraPoint(const orderedEdgePoint &point)
        {
            return Eigen::Vector3f(static_cast<float>(point.x_3d),
                                   static_cast<float>(point.y_3d),
                                   static_cast<float>(point.z_3d));
        }

        Eigen::Vector3f ToWorldPoint(const Eigen::Vector3f &pointCamera, const Sophus::SE3f &Tcw)
        {
            return Tcw.inverse() * pointCamera;
        }

        EdgeControlPoints ExtractWorldPoints(const std::vector<orderedEdgePoint> &points, const Sophus::SE3f &Tcw)
        {
            EdgeControlPoints worldPoints;
            worldPoints.reserve(points.size());

            for (const orderedEdgePoint &point : points)
            {
                if (!IsValidEdgePoint3D(point))
                    continue;

                worldPoints.push_back(ToWorldPoint(ToCameraPoint(point), Tcw));
            }

            return worldPoints;
        }

        Eigen::Vector3f ComputeCenterNoLock(const EdgeControlPoints &points)
        {
            Eigen::Vector3f center = Eigen::Vector3f::Zero();
            if (points.empty())
                return center;

            for (const Eigen::Vector3f &point : points)
                center += point;

            return center / static_cast<float>(points.size());
        }

        float CurveLength(const EdgeControlPoints &points)
        {
            if (points.size() < 2)
                return 0.0f;

            float length = 0.0f;
            for (size_t i = 1; i < points.size(); ++i)
                length += (points[i] - points[i - 1]).norm();

            return length;
        }

        int ClampScaleLevel(int scaleLevel, int nScaleLevels)
        {
            if (nScaleLevels <= 0)
                return 0;

            if (scaleLevel < 0)
                return 0;

            if (scaleLevel >= nScaleLevels)
                return nScaleLevels - 1;

            return scaleLevel;
        }
    }

    MapEdge::MapEdge()
        : mnId(0), mnFirstKFid(0), mnFirstFrame(0), nObs(0), mbTrackInView(false), mnTrackScaleLevel(0),
          mTrackViewCos(0.0f), mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0),
          mnFuseCandidateForKF(0), mnLoopEdgeForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0),
          mnBAGlobalForKF(0), mNormalVector(Eigen::Vector3f::Zero()), mpRefKF(static_cast<KeyFrame *>(NULL)),
          mnSemanticClass(-1), mnCurveOrder(0), mnVisible(1), mnFound(1), mbBad(false),
          mpReplaced(static_cast<MapEdge *>(NULL)), mfMinDistance(0.0f), mfMaxDistance(0.0f),
          mpMap(static_cast<Map *>(NULL))
    {
    }

    MapEdge::MapEdge(const EdgeControlPoints &controlPoints, KeyFrame *pRefKF, Map *pMap, int cls)
        : mnFirstKFid(pRefKF ? pRefKF->mnId : -1), mnFirstFrame(pRefKF ? pRefKF->mnFrameId : -1),
          nObs(0), mbTrackInView(false), mnTrackScaleLevel(0), mTrackViewCos(0.0f),
          mnTrackReferenceForFrame(0), mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0),
          mnLoopEdgeForKF(0), mnCorrectedByKF(0), mnCorrectedReference(0), mnBAGlobalForKF(0),
          mvWorldControlPoints(controlPoints), mNormalVector(Eigen::Vector3f::Zero()), mpRefKF(pRefKF),
          mnSemanticClass(cls), mnCurveOrder(static_cast<int>(controlPoints.size()) - 1), mnVisible(1),
          mnFound(1), mbBad(false), mpReplaced(static_cast<MapEdge *>(NULL)), mfMinDistance(0.0f),
          mfMaxDistance(0.0f), mpMap(pMap)
    {
        if (pRefKF && !controlPoints.empty())
        {
            Eigen::Vector3f normal = ComputeCenterNoLock(controlPoints) - pRefKF->GetCameraCenter();
            if (normal.norm() > std::numeric_limits<float>::epsilon())
                mNormalVector = normal.normalized();
            UpdateAverageDirAndDepth();
        }

        if (mpMap)
        {
            std::unique_lock<std::mutex> lock(mpMap->mMutexEdgeCreation);
            mnId = nNextId++;
        }
        else
        {
            std::unique_lock<std::mutex> lock(mGlobalMutex);
            mnId = nNextId++;
        }
    }

    MapEdge::MapEdge(const EdgeControlPoints &controlPoints, Map *pMap, Frame *pFrame, const int &idxF)
        : MapEdge(controlPoints, static_cast<KeyFrame *>(NULL), pMap, -1)
    {
        mnFirstKFid = -1;
        mnFirstFrame = pFrame ? pFrame->mnId : -1;
        mpRefKF = static_cast<KeyFrame *>(NULL);

        if (pFrame && idxF >= 0 && idxF < static_cast<int>(pFrame->mvBezierCurves.size()))
        {
            const BezierCurve &curve = pFrame->mvBezierCurves[idxF];
            mnSemanticClass = curve.cls;
            mnCurveOrder = static_cast<int>(curve.controlPoints.size()) - 1;

            Eigen::Vector3f normal = ComputeCenterNoLock(mvWorldControlPoints) - pFrame->GetCameraCenter();
            if (normal.norm() > std::numeric_limits<float>::epsilon())
                mNormalVector = normal.normalized();

            const float dist = normal.norm();
            mfMaxDistance = dist;
            mfMinDistance = dist;
        }
    }

    MapEdge::MapEdge(KeyFrame *pRefKF, Map *pMap, const int &idxKF)
        : MapEdge(EdgeControlPoints(), pRefKF, pMap, -1)
    {
        if (!pRefKF || idxKF < 0 || idxKF >= static_cast<int>(pRefKF->mvBezierCurves.size()))
            return;

        const BezierCurve &curve = pRefKF->mvBezierCurves[idxKF];
        mnSemanticClass = curve.cls;
        mnCurveOrder = static_cast<int>(curve.controlPoints.size()) - 1;
        mvWorldControlPoints = ExtractWorldPoints(curve.controlPoints, pRefKF->GetPose());
        mvWorldSampledPoints = ExtractWorldPoints(curve.sampledPoints, pRefKF->GetPose());
        AddObservation(pRefKF, static_cast<size_t>(idxKF));
        pRefKF->AddMapEdge(this, static_cast<size_t>(idxKF));
        if (mpMap)
            mpMap->AddMapEdge(this);
        UpdateAverageDirAndDepth();
    }

    MapEdge::MapEdge(Map *pMap, Frame *pFrame, const int &idxF)
        : mnFirstKFid(-1), mnFirstFrame(pFrame ? pFrame->mnId : -1), nObs(0), mbTrackInView(false),
          mnTrackScaleLevel(0), mTrackViewCos(0.0f), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
          mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopEdgeForKF(0), mnCorrectedByKF(0),
          mnCorrectedReference(0), mnBAGlobalForKF(0), mNormalVector(Eigen::Vector3f::Zero()),
          mpRefKF(static_cast<KeyFrame *>(NULL)), mnSemanticClass(-1), mnCurveOrder(0), mnVisible(1),
          mnFound(1), mbBad(false), mpReplaced(static_cast<MapEdge *>(NULL)), mfMinDistance(0.0f),
          mfMaxDistance(0.0f), mpMap(pMap)
    {
        if (pFrame && idxF >= 0 && idxF < static_cast<int>(pFrame->mvBezierCurves.size()))
        {
            const BezierCurve &curve = pFrame->mvBezierCurves[idxF];
            mnSemanticClass = curve.cls;
            mnCurveOrder = static_cast<int>(curve.controlPoints.size()) - 1;
            mvWorldControlPoints = ExtractWorldPoints(curve.controlPoints, pFrame->GetPose());
            mvWorldSampledPoints = ExtractWorldPoints(curve.sampledPoints, pFrame->GetPose());

            Eigen::Vector3f normal = ComputeCenterNoLock(mvWorldControlPoints) - pFrame->GetCameraCenter();
            if (normal.norm() > std::numeric_limits<float>::epsilon())
                mNormalVector = normal.normalized();

            const float dist = normal.norm();
            mfMaxDistance = dist;
            mfMinDistance = dist;
        }

        if (mpMap)
        {
            std::unique_lock<std::mutex> lock(mpMap->mMutexEdgeCreation);
            mnId = nNextId++;
            mpMap->AddMapEdge(this);
        }
        else
        {
            std::unique_lock<std::mutex> lock(mGlobalMutex);
            mnId = nNextId++;
        }
    }

    void MapEdge::SetWorldControlPoints(const EdgeControlPoints &controlPoints)
    {
        std::unique_lock<std::mutex> lock2(mGlobalMutex);
        std::unique_lock<std::mutex> lock(mMutexPos);
        mvWorldControlPoints = controlPoints;
        mnCurveOrder = static_cast<int>(mvWorldControlPoints.size()) - 1;
    }

    EdgeControlPoints MapEdge::GetWorldControlPoints()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mvWorldControlPoints;
    }

    void MapEdge::SetWorldSampledPoints(const EdgeControlPoints &sampledPoints)
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        mvWorldSampledPoints = sampledPoints;
    }

    EdgeControlPoints MapEdge::GetWorldSampledPoints()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mvWorldSampledPoints;
    }

    Eigen::Vector3f MapEdge::GetCenter()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return ComputeCenterNoLock(mvWorldControlPoints);
    }

    Eigen::Vector3f MapEdge::GetNormal()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mNormalVector;
    }

    void MapEdge::SetNormalVector(const Eigen::Vector3f &normal)
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        mNormalVector = normal;
    }

    int MapEdge::GetSemanticClass()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mnSemanticClass;
    }

    int MapEdge::GetCurveOrder()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mnCurveOrder;
    }

    KeyFrame *MapEdge::GetReferenceKeyFrame()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    void MapEdge::AddObservation(KeyFrame *pKF, size_t idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if (!pKF || mObservations.count(pKF))
            return;

        mObservations[pKF] = idx;
        nObs++;
    }

    void MapEdge::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        {
            std::unique_lock<std::mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF))
            {
                mObservations.erase(pKF);
                nObs--;

                if (mpRefKF == pKF)
                    mpRefKF = mObservations.empty() ? static_cast<KeyFrame *>(NULL) : mObservations.begin()->first;

                if (nObs <= 2)
                    bBad = true;
            }
        }

        if (bBad)
            SetBadFlag();
    }

    std::map<KeyFrame *, size_t> MapEdge::GetObservations()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mObservations;
    }

    int MapEdge::Observations()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return nObs;
    }

    int MapEdge::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return static_cast<int>(mObservations[pKF]);

        return -1;
    }

    bool MapEdge::IsInKeyFrame(KeyFrame *pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mObservations.count(pKF);
    }

    void MapEdge::SetBadFlag()
    {
        std::map<KeyFrame *, size_t> obs;
        {
            std::unique_lock<std::mutex> lock1(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            mbBad = true;
            obs = mObservations;
            mObservations.clear();
        }

        for (std::map<KeyFrame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; ++mit)
        {
            KeyFrame *pKF = mit->first;
            if (pKF)
                pKF->EraseMapEdgeMatch(static_cast<int>(mit->second));
        }

        Map *pMap = GetMap();
        if (pMap)
            pMap->EraseMapEdge(this);
    }

    bool MapEdge::isBad()
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures, std::defer_lock);
        std::unique_lock<std::mutex> lock2(mMutexPos, std::defer_lock);
        std::lock(lock1, lock2);
        return mbBad;
    }

    void MapEdge::Replace(MapEdge *pME)
    {
        if (!pME || pME->mnId == mnId)
            return;

        int nvisible, nfound;
        std::map<KeyFrame *, size_t> obs;
        {
            std::unique_lock<std::mutex> lock1(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            obs = mObservations;
            mObservations.clear();
            mbBad = true;
            nvisible = mnVisible;
            nfound = mnFound;
            mpReplaced = pME;
        }

        for (std::map<KeyFrame *, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; ++mit)
        {
            KeyFrame *pKF = mit->first;
            if (!pKF)
                continue;

            if (!pME->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapEdgeMatch(static_cast<int>(mit->second), pME);
                pME->AddObservation(pKF, mit->second);
            }
            else
            {
                pKF->EraseMapEdgeMatch(static_cast<int>(mit->second));
            }
        }

        pME->IncreaseFound(nfound);
        pME->IncreaseVisible(nvisible);

        Map *pMap = GetMap();
        if (pMap)
            pMap->EraseMapEdge(this);
    }

    MapEdge *MapEdge::GetReplaced()
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapEdge::IncreaseVisible(int n)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    void MapEdge::IncreaseFound(int n)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float MapEdge::GetFoundRatio()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if (mnVisible == 0)
            return 0.0f;

        return static_cast<float>(mnFound) / static_cast<float>(mnVisible);
    }

    void MapEdge::FuseWorldGeometry(const EdgeControlPoints &controlPoints, const EdgeControlPoints &sampledPoints)
    {
        std::unique_lock<std::mutex> lock(mMutexPos);

        auto fusePoints = [](EdgeControlPoints &target, const EdgeControlPoints &source)
        {
            if (source.size() < 2)
                return;

            if (target.size() != source.size())
            {
                if (target.size() < 2)
                    target = source;
                return;
            }

            for (size_t i = 0; i < target.size(); ++i)
                target[i] = 0.8f * target[i] + 0.2f * source[i];
        };

        fusePoints(mvWorldControlPoints, controlPoints);
        fusePoints(mvWorldSampledPoints, sampledPoints);
    }

    void MapEdge::UpdateAverageDirAndDepth()
    {
        std::map<KeyFrame *, size_t> observations;
        KeyFrame *pRefKF;
        EdgeControlPoints controlPoints;
        {
            std::unique_lock<std::mutex> lock1(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            if (mbBad)
                return;

            observations = mObservations;
            pRefKF = mpRefKF;
            controlPoints = mvWorldControlPoints;
        }

        if (controlPoints.empty())
            return;

        const Eigen::Vector3f center = ComputeCenterNoLock(controlPoints);
        Eigen::Vector3f normal = Eigen::Vector3f::Zero();
        int n = 0;

        for (std::map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; ++mit)
        {
            KeyFrame *pKF = mit->first;
            if (!pKF || pKF->isBad())
                continue;

            Eigen::Vector3f normali = center - pKF->GetCameraCenter();
            if (normali.norm() <= std::numeric_limits<float>::epsilon())
                continue;

            normal += normali.normalized();
            n++;
        }

        if (n == 0 && pRefKF)
        {
            Eigen::Vector3f normali = center - pRefKF->GetCameraCenter();
            if (normali.norm() > std::numeric_limits<float>::epsilon())
            {
                normal += normali.normalized();
                n++;
            }
        }

        float dist = 0.0f;
        if (pRefKF)
            dist = (center - pRefKF->GetCameraCenter()).norm();

        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            if (n > 0)
                mNormalVector = normal / static_cast<float>(n);

            if (dist > 0.0f)
            {
                mfMaxDistance = dist;
                mfMinDistance = dist;
            }
        }
    }

    float MapEdge::GetMinDistanceInvariance()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return 0.8f * mfMinDistance;
    }

    float MapEdge::GetMaxDistanceInvariance()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return 1.2f * mfMaxDistance;
    }

    int MapEdge::PredictScale(const float &currentDist, KeyFrame *pKF)
    {
        if (!pKF || currentDist <= 0.0f)
            return 0;

        float ratio;
        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        const int nScale = static_cast<int>(std::ceil(std::log(ratio) / pKF->mfLogScaleFactor));
        return ClampScaleLevel(nScale, pKF->mnScaleLevels);
    }

    int MapEdge::PredictScale(const float &currentDist, Frame *pF)
    {
        if (!pF || currentDist <= 0.0f)
            return 0;

        float ratio;
        {
            std::unique_lock<std::mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        const int nScale = static_cast<int>(std::ceil(std::log(ratio) / pF->mfLogScaleFactor));
        return ClampScaleLevel(nScale, pF->mnScaleLevels);
    }

    Map *MapEdge::GetMap()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mpMap;
    }

    void MapEdge::UpdateMap(Map *pMap)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mpMap = pMap;
    }

} // namespace ORB_SLAM3
