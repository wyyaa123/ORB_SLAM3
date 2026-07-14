#include "MapCurve.h"

namespace ORB_SLAM3
{

    long unsigned int MapCurve::nNextId = 0;
    std::mutex MapCurve::mGlobalMutex;

    MapCurve::MapCurve(const std::vector<Eigen::Vector3f> &vWorldPoints, KeyFrame *pRefKF, Map *pMap) : mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mbTrackInView(false), mnTrackReferenceForFrame(0), mbBad(false),
                                                                                                          mpRefKF(pRefKF), mpMap(pMap), mnOriginMapId(pMap->GetId()), mpReplaced(nullptr),
                                                                                                          mnVisible(1), mnFound(1), mnLastFrameSeen(0)
    {
        SetWorldPoints(vWorldPoints);

        NC = vWorldPoints.size();

        std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    MapCurve::MapCurve(const std::vector<Eigen::Vector3f> &vWorldPoints, Map *pMap, Frame *pFrame) : mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mbTrackInView(false), mnTrackReferenceForFrame(0), mbBad(false),
                                                                                                      mpRefKF(static_cast<KeyFrame *>(NULL)), mpMap(pMap), mnOriginMapId(pMap->GetId()), mpReplaced(nullptr),
                                                                                                      mnVisible(1), mnFound(1), mnLastFrameSeen(0)
    {
        SetWorldPoints(vWorldPoints);

        NC = vWorldPoints.size();

        std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    void MapCurve::AddObservation(KeyFrame *pKF, int idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);

        // 这个KeyFrame是否已经观察过该Bezier？
        if (mObservations.count(pKF))
            return;

        mObservations[pKF] = idx;
        nObs++;
    }

    void MapCurve::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF))
            {
                int idx = mObservations[pKF];
                if (idx != -1)
                    nObs--;

                mObservations.erase(pKF);

                if (mpRefKF == pKF)
                    mpRefKF = mObservations.empty() ? nullptr : mObservations.begin()->first;

                if (!nObs)
                    bBad = true;
            }
        }

        if (bBad)
            SetBadFlag();
    }

    std::map<KeyFrame *, int> MapCurve::GetObservations()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mObservations;
    }

    int MapCurve::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    bool MapCurve::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }

    std::vector<Eigen::Vector3f> MapCurve::GetWorldPoints()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mvWorldPoints;
    }

    void MapCurve::SetWorldPoints(const std::vector<Eigen::Vector3f> &vWorldPoints)
    {
        std::unique_lock<std::mutex> lock2(mGlobalMutex);
        std::unique_lock<std::mutex> lock(mMutexPos);
        mvWorldPoints = vWorldPoints;
    }

    bool MapCurve::isBad()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mbBad;
    }

    void MapCurve::SetBadFlag()
    {
        map<KeyFrame *, int> obs;
        {
            std::unique_lock<std::mutex> lock1(mMutexFeatures);
            std::unique_lock<std::mutex> lock2(mMutexPos);
            mbBad = true;
            obs = mObservations;
            mObservations.clear();
        }

        for (map<KeyFrame *, int>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            pKF->EraseMapCurveMatch(this);
        }
        mpMap->EraseMapCurve(this);
    }

    void MapCurve::Replace(MapCurve *pMB)
    {
        if (pMB->mnId == this->mnId)
            return;

        int nvisible, nfound;
        map<KeyFrame *, int> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            obs = mObservations;
            mObservations.clear();
            mbBad = true;
            nvisible = mnVisible;
            nfound = mnFound;
            mpReplaced = pMB;
        }

        for (map<KeyFrame *, int>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
        {
            // Replace measurement in keyframe
            KeyFrame *pKF = mit->first;

            int indexes = mit->second;

            if (!pMB->IsInKeyFrame(pKF))
            {
                if (indexes != -1)
                {
                    pKF->ReplaceMapCurveMatch(this, pMB);
                    pMB->AddObservation(pKF, indexes);
                }
            }
            else
            {
                if (indexes != -1)
                {
                    pKF->EraseMapCurveMatch(this);
                }
            }
        }
        pMB->IncreaseFound(nfound);
        pMB->IncreaseVisible(nvisible);
        // pMB->ComputeDistinctiveDescriptors();

        mpMap->EraseMapCurve(this);
    }

    MapCurve *MapCurve::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapCurve::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    void MapCurve::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float MapCurve::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }

    Map *MapCurve::GetMap()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mpMap;
    }

    int MapCurve::Observations()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return nObs;
    }
}
