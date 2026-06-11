/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "MapPoint.h"
#include "ORBmatcher.h"

#include <mutex>

namespace ORB_SLAM3
{

    long unsigned int MapPoint::nNextId = 0;
    mutex MapPoint::mGlobalMutex;

    MapPoint::MapPoint() : mnFirstKFid(0), mnFirstFrame(0), nObs(0), mnTrackReferenceForFrame(0),
                           mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
                           mnCorrectedReference(0), mnBAGlobalForKF(0), mnVisible(1), mnFound(1), mbBad(false),
                           mpReplaced(static_cast<MapPoint *>(NULL))
    {
        mpReplaced = static_cast<MapPoint *>(NULL);
    }

    MapPoint::MapPoint(const Eigen::Vector3f &Pos, KeyFrame *pRefKF, Map *pMap) : mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
                                                                                  mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
                                                                                  mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
                                                                                  mpReplaced(static_cast<MapPoint *>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
                                                                                  mnOriginMapId(pMap->GetId())
    {
        SetWorldPos(Pos);

        mNormalVector.setZero();

        mbTrackInViewR = false;
        mbTrackInView = false;

        // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    MapPoint::MapPoint(const double invDepth, cv::Point2f uv_init, KeyFrame *pRefKF, KeyFrame *pHostKF, Map *pMap) : mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
                                                                                                                     mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
                                                                                                                     mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
                                                                                                                     mpReplaced(static_cast<MapPoint *>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap),
                                                                                                                     mnOriginMapId(pMap->GetId())
    {
        mInvDepth = invDepth;
        mInitU = (double)uv_init.x;
        mInitV = (double)uv_init.y;
        mpHostKF = pHostKF;

        mNormalVector.setZero();

        // Worldpos is not set
        // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    MapPoint::MapPoint(const Eigen::Vector3f &Pos, Map *pMap, Frame *pFrame, const int &idxF) : mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
                                                                                                mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
                                                                                                mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame *>(NULL)), mnVisible(1),
                                                                                                mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap), mnOriginMapId(pMap->GetId())
    {
        SetWorldPos(Pos);

        Eigen::Vector3f Ow;
        if (pFrame->Nleft == -1 || idxF < pFrame->Nleft)
        {
            Ow = pFrame->GetCameraCenter();
        }
        else
        {
            Eigen::Matrix3f Rwl = pFrame->GetRwc();
            Eigen::Vector3f tlr = pFrame->GetRelativePoseTlr().translation();
            Eigen::Vector3f twl = pFrame->GetOw();

            Ow = Rwl * tlr + twl;
        }
        mNormalVector = mWorldPos - Ow;
        mNormalVector = mNormalVector / mNormalVector.norm();

        Eigen::Vector3f PC = mWorldPos - Ow;
        const float dist = PC.norm();
        const int level = (pFrame->Nleft == -1)    ? pFrame->mvKeysUn[idxF].octave
                          : (idxF < pFrame->Nleft) ? pFrame->mvKeys[idxF].octave
                                                   : pFrame->mvKeysRight[idxF].octave;
        const float levelScaleFactor = pFrame->mvScaleFactors[level];
        const int nLevels = pFrame->mnScaleLevels;

        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

        pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

        // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
        unique_lock<mutex> lock(mpMap->mMutexPointCreation);
        mnId = nNextId++;
    }

    void MapPoint::SetWorldPos(const Eigen::Vector3f &Pos)
    {
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        mWorldPos = Pos;
    }

    Eigen::Vector3f MapPoint::GetWorldPos()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldPos;
    }

    Eigen::Vector3f MapPoint::GetNormal()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mNormalVector;
    }

    KeyFrame *MapPoint::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    void MapPoint::AddObservation(KeyFrame *pKF, int idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        tuple<int, int> indexes;

        if (mObservations.count(pKF))
        {
            indexes = mObservations[pKF];
        }
        else
        {
            indexes = tuple<int, int>(-1, -1);
        }

        if (pKF->NLeft != -1 && idx >= pKF->NLeft)
        {
            get<1>(indexes) = idx;
        }
        else
        {
            get<0>(indexes) = idx;
        }

        mObservations[pKF] = indexes;

        if (!pKF->mpCamera2 && pKF->mvuRight[idx] >= 0)
            nObs += 2;
        else
            nObs++;
    }

    void MapPoint::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if (mObservations.count(pKF))
            {
                tuple<int, int> indexes = mObservations[pKF];
                int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

                if (leftIndex != -1)
                {
                    if (!pKF->mpCamera2 && pKF->mvuRight[leftIndex] >= 0)
                        nObs -= 2;
                    else
                        nObs--;
                }
                if (rightIndex != -1)
                {
                    nObs--;
                }

                mObservations.erase(pKF);

                if (mpRefKF == pKF)
                    mpRefKF = mObservations.begin()->first;

                // If only 2 observations or less, discard point
                if (nObs <= 2)
                    bBad = true;
            }
        }

        if (bBad)
            SetBadFlag();
    }

    // 同一个特征点在哪些关键帧中被观测到，tuple<int, int>（左右）是观测到的特征点在关键帧中的索引
    std::map<KeyFrame *, std::tuple<int, int>> MapPoint::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }
    // 这个 MapPoint 被多少关键帧观测到
    int MapPoint::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    void MapPoint::SetBadFlag()
    {
        map<KeyFrame *, tuple<int, int>> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad = true;
            obs = mObservations;
            mObservations.clear();
        }
        for (map<KeyFrame *, tuple<int, int>>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            int leftIndex = get<0>(mit->second), rightIndex = get<1>(mit->second);
            if (leftIndex != -1)
            {
                pKF->EraseMapPointMatch(leftIndex);
            }
            if (rightIndex != -1)
            {
                pKF->EraseMapPointMatch(rightIndex);
            }
        }

        mpMap->EraseMapPoint(this);
    }

    MapPoint *MapPoint::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapPoint::Replace(MapPoint *pMP)
    {
        if (pMP->mnId == this->mnId)
            return;

        int nvisible, nfound;
        map<KeyFrame *, tuple<int, int>> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            obs = mObservations;
            mObservations.clear();
            mbBad = true;
            nvisible = mnVisible;
            nfound = mnFound;
            mpReplaced = pMP;
        }

        for (map<KeyFrame *, tuple<int, int>>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
        {
            // Replace measurement in keyframe
            KeyFrame *pKF = mit->first;

            tuple<int, int> indexes = mit->second;
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if (!pMP->IsInKeyFrame(pKF))
            {
                if (leftIndex != -1)
                {
                    pKF->ReplaceMapPointMatch(leftIndex, pMP);
                    pMP->AddObservation(pKF, leftIndex);
                }
                if (rightIndex != -1)
                {
                    pKF->ReplaceMapPointMatch(rightIndex, pMP);
                    pMP->AddObservation(pKF, rightIndex);
                }
            }
            else
            {
                if (leftIndex != -1)
                {
                    pKF->EraseMapPointMatch(leftIndex);
                }
                if (rightIndex != -1)
                {
                    pKF->EraseMapPointMatch(rightIndex);
                }
            }
        }
        pMP->IncreaseFound(nfound);
        pMP->IncreaseVisible(nvisible);
        pMP->ComputeDistinctiveDescriptors();

        mpMap->EraseMapPoint(this);
    }

    bool MapPoint::isBad()
    {
        unique_lock<mutex> lock1(mMutexFeatures, std::defer_lock);
        unique_lock<mutex> lock2(mMutexPos, std::defer_lock);
        lock(lock1, lock2);

        return mbBad;
    }

    void MapPoint::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    void MapPoint::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float MapPoint::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound) / mnVisible;
    }

    /**

    * @brief 为当前 MapPoint 选择最具代表性的描述子。
    *
    * 一个 MapPoint 可能被多个关键帧观测到，因此会对应多个 ORB 描述子。
    * 该函数从所有有效观测中收集该地图点对应的描述子，并通过描述子之间的
    * 汉明距离选择一个最稳定、最具代表性的描述子，作为当前 MapPoint 的
    * mDescriptor，用于后续的地图点匹配、投影搜索和重定位等过程。
    *
    * 该函数通过以下步骤实现描述子选择：
    * 1. 加锁读取当前 MapPoint 的所有观测关键帧，并在地图点被标记为 bad 时直接返回。
    * 2. 遍历所有有效关键帧，提取该 MapPoint 在左图或右图中对应的 ORB 描述子。
    * 3. 计算所有候选描述子之间的两两汉明距离，构建描述子距离矩阵。
    * 4. 对每个候选描述子，将其到其他描述子的距离排序，并取其中位数作为该描述子的代表距离。
    * 5. 选择中位距离最小的描述子，认为其与其他观测描述子整体最相似。
    * 6. 将该描述子拷贝并保存为当前 MapPoint 的代表描述子 mDescriptor。
    *
    * @note 该方法使用中位距离而不是平均距离，可以降低异常观测或错误匹配对描述子选择的影响。
    * @return void
      */
    void MapPoint::ComputeDistinctiveDescriptors()
    {
        // Retrieve all observed descriptors
        vector<cv::Mat> vDescriptors;

        map<KeyFrame *, tuple<int, int>> observations;

        {
            unique_lock<mutex> lock1(mMutexFeatures);
            if (mbBad)
                return;
            observations = mObservations;
        }

        if (observations.empty())
            return;

        vDescriptors.reserve(observations.size());

        for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;

            if (!pKF->isBad())
            {
                tuple<int, int> indexes = mit->second;
                int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

                if (leftIndex != -1)
                {
                    vDescriptors.push_back(pKF->mDescriptors.row(leftIndex));
                }
                if (rightIndex != -1)
                {
                    vDescriptors.push_back(pKF->mDescriptors.row(rightIndex));
                }
            }
        }

        if (vDescriptors.empty())
            return;

        // Compute distances between them
        const size_t N = vDescriptors.size();

        float Distances[N][N];
        for (size_t i = 0; i < N; i++)
        {
            Distances[i][i] = 0;
            for (size_t j = i + 1; j < N; j++)
            {
                int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for (size_t i = 0; i < N; i++)
        {
            vector<int> vDists(Distances[i], Distances[i] + N);
            sort(vDists.begin(), vDists.end());
            int median = vDists[0.5 * (N - 1)];

            if (median < BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }

        {
            unique_lock<mutex> lock(mMutexFeatures);
            mDescriptor = vDescriptors[BestIdx].clone();
        }
    }

    cv::Mat MapPoint::GetDescriptor()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mDescriptor.clone();
    }

    tuple<int, int> MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mObservations.count(pKF))
            return mObservations[pKF];
        else
            return tuple<int, int>(-1, -1);
    }

    bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }

    /**
    * @brief 更新当前 MapPoint 的平均观测方向和有效深度范围。
    *
    * 一个 MapPoint 可能被多个关键帧从不同视角观测到。该函数根据所有有效观测
    * 计算该地图点的平均法向量 mNormalVector，并根据参考关键帧中的尺度层级
    * 估计该地图点在后续匹配时允许的最大和最小观测距离。
    *
    * 该函数通过以下步骤实现更新：
    * 1. 加锁读取当前 MapPoint 的观测信息、参考关键帧和世界坐标位置。
    * 2. 若地图点被标记为 bad，或当前没有任何观测，则直接返回。
    * 3. 遍历所有观测该 MapPoint 的关键帧，分别处理左目和右目观测。
    * 4. 对每一次有效观测，计算相机中心指向 MapPoint 的单位方向向量。
    * 5. 将所有单位方向向量累加，并取平均值作为该 MapPoint 的平均观测方向。
    * 6. 根据 MapPoint 到参考关键帧相机中心的距离，结合其所在图像金字塔层级，
    * 计算该地图点的最大可观测距离 mfMaxDistance。
    * 7. 根据图像金字塔的最大尺度因子，进一步计算最小可观测距离 mfMinDistance。
    * 8. 加锁更新 mNormalVector、mfMaxDistance 和 mfMinDistance。
    *
    * @note mNormalVector 用于判断后续观测方向是否合理，mfMaxDistance 和
    * ```
        mfMinDistance 用于限制地图点在不同尺度下的匹配搜索范围。
      ```
    *
    * @return void
      */
    void MapPoint::UpdateNormalAndDepth()
    {
        map<KeyFrame *, tuple<int, int>> observations;
        KeyFrame *pRefKF;
        Eigen::Vector3f Pos;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            if (mbBad)
                return;
            observations = mObservations;
            pRefKF = mpRefKF;
            Pos = mWorldPos;
        }

        if (observations.empty())
            return;

        Eigen::Vector3f normal;
        normal.setZero();
        int n = 0;
        for (map<KeyFrame *, tuple<int, int>>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;

            tuple<int, int> indexes = mit->second;
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);

            if (leftIndex != -1)
            {
                Eigen::Vector3f Owi = pKF->GetCameraCenter();
                Eigen::Vector3f normali = Pos - Owi;
                normal = normal + normali / normali.norm();
                n++;
            }
            if (rightIndex != -1)
            {
                Eigen::Vector3f Owi = pKF->GetRightCameraCenter();
                Eigen::Vector3f normali = Pos - Owi;
                normal = normal + normali / normali.norm();
                n++;
            }
        }

        Eigen::Vector3f PC = Pos - pRefKF->GetCameraCenter();
        const float dist = PC.norm();

        tuple<int, int> indexes = observations[pRefKF];
        int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
        int level;
        if (pRefKF->NLeft == -1)
        {
            level = pRefKF->mvKeysUn[leftIndex].octave;
        }
        else if (leftIndex != -1)
        {
            level = pRefKF->mvKeys[leftIndex].octave;
        }
        else
        {
            level = pRefKF->mvKeysRight[rightIndex - pRefKF->NLeft].octave;
        }

        // const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
        const float levelScaleFactor = pRefKF->mvScaleFactors[level];
        const int nLevels = pRefKF->mnScaleLevels;

        {
            unique_lock<mutex> lock3(mMutexPos);
            mfMaxDistance = dist * levelScaleFactor;
            mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
            mNormalVector = normal / n;
        }
    }

    void MapPoint::SetNormalVector(const Eigen::Vector3f &normal)
    {
        unique_lock<mutex> lock3(mMutexPos);
        mNormalVector = normal;
    }

    float MapPoint::GetMinDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 0.8f * mfMinDistance;
    }

    float MapPoint::GetMaxDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 1.2f * mfMaxDistance;
    }

    int MapPoint::PredictScale(const float &currentDist, KeyFrame *pKF)
    {
        float ratio;
        {
            unique_lock<mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pKF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pKF->mnScaleLevels)
            nScale = pKF->mnScaleLevels - 1;

        return nScale;
    }

    int MapPoint::PredictScale(const float &currentDist, Frame *pF)
    {
        float ratio;
        {
            unique_lock<mutex> lock(mMutexPos);
            ratio = mfMaxDistance / currentDist;
        }

        int nScale = ceil(log(ratio) / pF->mfLogScaleFactor);
        if (nScale < 0)
            nScale = 0;
        else if (nScale >= pF->mnScaleLevels)
            nScale = pF->mnScaleLevels - 1;

        return nScale;
    }

    void MapPoint::PrintObservations()
    {
        cout << "MP_OBS: MP " << mnId << endl;
        for (map<KeyFrame *, tuple<int, int>>::iterator mit = mObservations.begin(), mend = mObservations.end(); mit != mend; mit++)
        {
            KeyFrame *pKFi = mit->first;
            tuple<int, int> indexes = mit->second;
            int leftIndex = get<0>(indexes), rightIndex = get<1>(indexes);
            cout << "--OBS in KF " << pKFi->mnId << " in map " << pKFi->GetMap()->GetId() << endl;
        }
    }

    Map *MapPoint::GetMap()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mpMap;
    }

    void MapPoint::UpdateMap(Map *pMap)
    {
        unique_lock<mutex> lock(mMutexMap);
        mpMap = pMap;
    }

    void MapPoint::PreSave(set<KeyFrame *> &spKF, set<MapPoint *> &spMP)
    {
        mBackupReplacedId = -1;
        if (mpReplaced && spMP.find(mpReplaced) != spMP.end())
            mBackupReplacedId = mpReplaced->mnId;

        mBackupObservationsId1.clear();
        mBackupObservationsId2.clear();
        // Save the id and position in each KF who view it
        for (std::map<KeyFrame *, std::tuple<int, int>>::const_iterator it = mObservations.begin(), end = mObservations.end(); it != end; ++it)
        {
            KeyFrame *pKFi = it->first;
            if (spKF.find(pKFi) != spKF.end())
            {
                mBackupObservationsId1[it->first->mnId] = get<0>(it->second);
                mBackupObservationsId2[it->first->mnId] = get<1>(it->second);
            }
            else
            {
                EraseObservation(pKFi);
            }
        }

        // Save the id of the reference KF
        if (spKF.find(mpRefKF) != spKF.end())
        {
            mBackupRefKFId = mpRefKF->mnId;
        }
    }

    void MapPoint::PostLoad(map<long unsigned int, KeyFrame *> &mpKFid, map<long unsigned int, MapPoint *> &mpMPid)
    {
        mpRefKF = mpKFid[mBackupRefKFId];
        if (!mpRefKF)
        {
            cout << "ERROR: MP without KF reference " << mBackupRefKFId << "; Num obs: " << nObs << endl;
        }
        mpReplaced = static_cast<MapPoint *>(NULL);
        if (mBackupReplacedId >= 0)
        {
            map<long unsigned int, MapPoint *>::iterator it = mpMPid.find(mBackupReplacedId);
            if (it != mpMPid.end())
                mpReplaced = it->second;
        }

        mObservations.clear();

        for (map<long unsigned int, int>::const_iterator it = mBackupObservationsId1.begin(), end = mBackupObservationsId1.end(); it != end; ++it)
        {
            KeyFrame *pKFi = mpKFid[it->first];
            map<long unsigned int, int>::const_iterator it2 = mBackupObservationsId2.find(it->first);
            std::tuple<int, int> indexes = tuple<int, int>(it->second, it2->second);
            if (pKFi)
            {
                mObservations[pKFi] = indexes;
            }
        }

        mBackupObservationsId1.clear();
        mBackupObservationsId2.clear();
    }

} // namespace ORB_SLAM
