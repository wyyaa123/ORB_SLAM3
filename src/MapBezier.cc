#include "MapBezier.h"

namespace ORB_SLAM3
{

    long unsigned int MapBezier::nNextId = 0;
    std::mutex MapBezier::mGlobalMutex;

    MapBezier::MapBezier(const std::vector<Eigen::Vector3f> &vWorldPoints, KeyFrame *pRefKF, Map *pMap) : mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mbBad(false),
                                                                                                          mpRefKF(pRefKF), mpMap(pMap), mnOriginMapId(pMap->GetId()), mpReplaced(nullptr), 
                                                                                                          mnVisible(1), mnFound(1)
    {
        SetWorldPoints(vWorldPoints);

        std::unique_lock<std::mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    void MapBezier::AddObservation(KeyFrame *pKF, int idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);

        if (mObservations.count(pKF))
            return;

        mObservations[pKF] = idx;
        nObs++;

        assert(nObs == static_cast<int>(mObservations.size()));
    }

    void MapBezier::EraseObservation(KeyFrame *pKF)
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
                    mpRefKF = mObservations.begin()->first;

                if (!nObs)
                    bBad = true;
            }
        }

        if (bBad)
            SetBadFlag();
    }

    std::map<KeyFrame *, int> MapBezier::GetObservations()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mObservations;
    }

    int MapBezier::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    bool MapBezier::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }

    std::vector<Eigen::Vector3f> MapBezier::GetWorldPoints()
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mvWorldPoints;
    }

    void MapBezier::SetWorldPoints(const std::vector<Eigen::Vector3f> &vWorldPoints)
    {
        std::unique_lock<std::mutex> lock2(mGlobalMutex);
        std::unique_lock<std::mutex> lock(mMutexPos);
        mvWorldPoints = vWorldPoints;
    }

    bool MapBezier::isBad()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mbBad;
    }

    void MapBezier::SetBadFlag()
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
            int idx = mit->second;
            if (idx != -1)
                pKF->EraseMapBezierMatch(idx);
        }
        mpMap->EraseMapBezier(this);
    }

    void MapBezier::Replace(MapBezier *pMB)
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
                    pKF->ReplaceMapBezierMatch(indexes, pMB);
                    pMB->AddObservation(pKF, indexes);
                }
            }
            else
            {
                if (indexes != -1)
                {
                    pKF->EraseMapBezierMatch(indexes);
                }
            }
        }
        pMB->IncreaseFound(nfound);
        pMB->IncreaseVisible(nvisible);
        // pMB->ComputeDistinctiveDescriptors();

        mpMap->EraseMapBezier(this);
    }

    MapBezier *MapBezier::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapBezier::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    void MapBezier::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float MapBezier::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }

    Map *MapBezier::GetMap()
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mpMap;
    }

    int MapBezier::Observations()
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return nObs;
    }
}
