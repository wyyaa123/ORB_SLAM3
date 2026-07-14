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

#include "OptimizableTypes.h"

#include <cassert>
#include <cmath>

namespace ORB_SLAM3
{
    bool EdgeSE3ProjectXYZOnlyPose::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZOnlyPose::write(std::ostream &os) const
    {

        for (int i = 0; i < 2; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    void EdgeSE3ProjectXYZOnlyPose::linearizeOplus()
    {
        g2o::VertexSE3Expmap *vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Vector3d xyz_trans = vi->estimate().map(Xw);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        Eigen::Matrix<double, 3, 6> SE3deriv;
        SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f,
            -z, 0.f, x, 0.f, 1.f, 0.f,
            y, -x, 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXi = -pCamera->projectJac(xyz_trans) * SE3deriv;
    }

    double EdgeSE3ProjectCurveOnlyPose::projectedCurvature(const g2o::SE3Quat &Tcw) const
    {
        const Eigen::Vector2d previous = pCamera->project(Tcw.map(XwPrevious));
        const Eigen::Vector2d current = pCamera->project(Tcw.map(Xw));
        const Eigen::Vector2d next = pCamera->project(Tcw.map(XwNext));

        const Eigen::Vector2d first = current - previous;
        const Eigen::Vector2d second = next - current;
        const Eigen::Vector2d chord = next - previous;
        const double denominator = first.norm() * second.norm() * chord.norm();

        if (denominator <= 1e-12)
            return 0.0;

        const double cross = first.x() * second.y() - first.y() * second.x();
        return 2.0 * std::fabs(cross) / denominator;
    }

    void EdgeSE3ProjectCurveOnlyPose::computeError()
    {
        assert(_vertices[0] != nullptr);
        assert(pCamera != nullptr);

        const g2o::VertexSE3Expmap *vSE3 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        const g2o::SE3Quat &Tcw = vSE3->estimate();
        const Eigen::Vector3d previousCameraPoint = Tcw.map(XwPrevious);
        const Eigen::Vector3d cameraPoint = Tcw.map(Xw);
        const Eigen::Vector3d nextCameraPoint = Tcw.map(XwNext);
        if (previousCameraPoint.z() <= 1e-6 || cameraPoint.z() <= 1e-6 || nextCameraPoint.z() <= 1e-6)
        {
            _error.setConstant(1e3);
            return;
        }

        const Eigen::Vector2d projected = pCamera->project(cameraPoint);

        const double normalNorm = normal.norm();
        if (normalNorm <= 1e-12)
        {
            _error.setZero();
            return;
        }

        const Eigen::Vector2d unitNormal = normal / normalNorm;
        // Normal-position residual and curvature-consistency residual.
        _error[0] = unitNormal.dot(projected - _measurement);
        _error[1] = curvatureScale * (projectedCurvature(Tcw) - observedCurvature);
    }

    bool EdgeSE3ProjectCurveOnlyPose::isDepthPositive()
    {
        const g2o::VertexSE3Expmap *vSE3 = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        const g2o::SE3Quat &Tcw = vSE3->estimate();
        return Tcw.map(XwPrevious).z() > 0.0 &&
               Tcw.map(Xw).z() > 0.0 &&
               Tcw.map(XwNext).z() > 0.0;
    }

    void EdgeSE3ProjectCurveOnlyPose::linearizeOplus()
    {
        assert(_vertices[0] != nullptr);
        assert(pCamera != nullptr);

        g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        const g2o::SE3Quat Tcw = vSE3->estimate();
        const Eigen::Vector3d cameraPoint = Tcw.map(Xw);

        if (!isDepthPositive())
        {
            _jacobianOplusXi.setZero();
            return;
        }

        const double x = cameraPoint.x();
        const double y = cameraPoint.y();
        const double z = cameraPoint.z();

        Eigen::Matrix<double, 3, 6> SE3deriv;
        SE3deriv << 0.0, z, -y, 1.0, 0.0, 0.0,
            -z, 0.0, x, 0.0, 1.0, 0.0,
            y, -x, 0.0, 0.0, 0.0, 1.0;

        const double normalNorm = normal.norm();
        if (normalNorm <= 1e-12)
        {
            _jacobianOplusXi.setZero();
            return;
        }

        const Eigen::Vector2d unitNormal = normal / normalNorm;
        _jacobianOplusXi.row(0) = unitNormal.transpose() * pCamera->projectJac(cameraPoint) * SE3deriv;

        constexpr double delta = 1e-6;
        for (int i = 0; i < 6; ++i)
        {
            Eigen::Matrix<double, 6, 1> update = Eigen::Matrix<double, 6, 1>::Zero();
            update[i] = delta;
            const g2o::SE3Quat positive = g2o::SE3Quat::exp(update) * Tcw;

            update[i] = -delta;
            const g2o::SE3Quat negative = g2o::SE3Quat::exp(update) * Tcw;

            _jacobianOplusXi(1, i) = curvatureScale *
                                      (projectedCurvature(positive) - projectedCurvature(negative)) /
                                      (2.0 * delta);
        }
    }

    bool EdgeSE3ProjectXYZOnlyPoseToBody::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZOnlyPoseToBody::write(std::ostream &os) const
    {

        for (int i = 0; i < 2; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    void EdgeSE3ProjectXYZOnlyPoseToBody::linearizeOplus()
    {
        g2o::VertexSE3Expmap *vi = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T_lw(vi->estimate());
        Eigen::Vector3d X_l = T_lw.map(Xw);
        Eigen::Vector3d X_r = mTrl.map(T_lw.map(Xw));

        double x_w = X_l[0];
        double y_w = X_l[1];
        double z_w = X_l[2];

        Eigen::Matrix<double, 3, 6> SE3deriv;
        SE3deriv << 0.f, z_w, -y_w, 1.f, 0.f, 0.f,
            -z_w, 0.f, x_w, 0.f, 1.f, 0.f,
            y_w, -x_w, 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXi = -pCamera->projectJac(X_r) * mTrl.rotation().toRotationMatrix() * SE3deriv;
    }

    EdgeSE3ProjectXYZ::EdgeSE3ProjectXYZ() : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>()
    {
    }

    bool EdgeSE3ProjectXYZ::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZ::write(std::ostream &os) const
    {

        for (int i = 0; i < 2; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    void EdgeSE3ProjectXYZ::linearizeOplus()
    {
        g2o::VertexSE3Expmap *vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
        g2o::SE3Quat T(vj->estimate());
        g2o::VertexSBAPointXYZ *vi = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
        Eigen::Vector3d xyz = vi->estimate();
        Eigen::Vector3d xyz_trans = T.map(xyz);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        auto projectJac = -pCamera->projectJac(xyz_trans);

        _jacobianOplusXi = projectJac * T.rotation().toRotationMatrix();

        Eigen::Matrix<double, 3, 6> SE3deriv;
        SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f,
            -z, 0.f, x, 0.f, 1.f, 0.f,
            y, -x, 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXj = projectJac * SE3deriv;
    }

    EdgeSE3ProjectXYZToBody::EdgeSE3ProjectXYZToBody() : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>()
    {
    }

    bool EdgeSE3ProjectXYZToBody::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }
        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSE3ProjectXYZToBody::write(std::ostream &os) const
    {

        for (int i = 0; i < 2; i++)
        {
            os << measurement()[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    void EdgeSE3ProjectXYZToBody::linearizeOplus()
    {
        g2o::VertexSE3Expmap *vj = static_cast<g2o::VertexSE3Expmap *>(_vertices[1]);
        g2o::SE3Quat T_lw(vj->estimate());
        g2o::SE3Quat T_rw = mTrl * T_lw;
        g2o::VertexSBAPointXYZ *vi = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
        Eigen::Vector3d X_w = vi->estimate();
        Eigen::Vector3d X_l = T_lw.map(X_w);
        Eigen::Vector3d X_r = mTrl.map(T_lw.map(X_w));

        _jacobianOplusXi = -pCamera->projectJac(X_r) * T_rw.rotation().toRotationMatrix();

        double x = X_l[0];
        double y = X_l[1];
        double z = X_l[2];

        Eigen::Matrix<double, 3, 6> SE3deriv;
        SE3deriv << 0.f, z, -y, 1.f, 0.f, 0.f,
            -z, 0.f, x, 0.f, 1.f, 0.f,
            y, -x, 0.f, 0.f, 0.f, 1.f;

        _jacobianOplusXj = -pCamera->projectJac(X_r) * mTrl.rotation().toRotationMatrix() * SE3deriv;
    }

    VertexSim3Expmap::VertexSim3Expmap() : BaseVertex<7, g2o::Sim3>()
    {
        _marginalized = false;
        _fix_scale = false;
    }

    bool VertexSim3Expmap::read(std::istream &is)
    {
        g2o::Vector7d cam2world;
        for (int i = 0; i < 6; i++)
        {
            is >> cam2world[i];
        }
        is >> cam2world[6];

        float nextParam;
        for (size_t i = 0; i < pCamera1->size(); i++)
        {
            is >> nextParam;
            pCamera1->setParameter(nextParam, i);
        }

        for (size_t i = 0; i < pCamera2->size(); i++)
        {
            is >> nextParam;
            pCamera2->setParameter(nextParam, i);
        }

        setEstimate(g2o::Sim3(cam2world).inverse());
        return true;
    }

    bool VertexSim3Expmap::write(std::ostream &os) const
    {
        g2o::Sim3 cam2world(estimate().inverse());
        g2o::Vector7d lv = cam2world.log();
        for (int i = 0; i < 7; i++)
        {
            os << lv[i] << " ";
        }

        for (size_t i = 0; i < pCamera1->size(); i++)
        {
            os << pCamera1->getParameter(i) << " ";
        }

        for (size_t i = 0; i < pCamera2->size(); i++)
        {
            os << pCamera2->getParameter(i) << " ";
        }

        return os.good();
    }

    EdgeSim3ProjectXYZ::EdgeSim3ProjectXYZ() : g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, VertexSim3Expmap>()
    {
    }

    bool EdgeSim3ProjectXYZ::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeSim3ProjectXYZ::write(std::ostream &os) const
    {
        for (int i = 0; i < 2; i++)
        {
            os << _measurement[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }

    EdgeInverseSim3ProjectXYZ::EdgeInverseSim3ProjectXYZ() : g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSBAPointXYZ, VertexSim3Expmap>()
    {
    }

    bool EdgeInverseSim3ProjectXYZ::read(std::istream &is)
    {
        for (int i = 0; i < 2; i++)
        {
            is >> _measurement[i];
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                is >> information()(i, j);
                if (i != j)
                    information()(j, i) = information()(i, j);
            }
        return true;
    }

    bool EdgeInverseSim3ProjectXYZ::write(std::ostream &os) const
    {
        for (int i = 0; i < 2; i++)
        {
            os << _measurement[i] << " ";
        }

        for (int i = 0; i < 2; i++)
            for (int j = i; j < 2; j++)
            {
                os << " " << information()(i, j);
            }
        return os.good();
    }
}
