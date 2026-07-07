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

    EdgeSE3ProjectBezierPointOnlyPose::EdgeSE3ProjectBezierPointOnlyPose()
        : Xw(Eigen::Vector3d::Zero()),
          normal(Eigen::Vector2d::Zero()),
          pCamera(nullptr)
    {
        information().setIdentity();
    }

    void EdgeSE3ProjectBezierPointOnlyPose::computeError()
    {
        assert(_vertices[0] != nullptr);
        assert(pCamera != nullptr);

        const auto *v =
            static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        const Eigen::Vector3d Xc = v->estimate().map(Xw);

        const double normalNorm = normal.norm();
        assert(normalNorm > 1e-12);
        if (normalNorm <= 1e-12)
        {
            _error.setZero();
            return;
        }

        const Eigen::Vector2d projected = pCamera->project(Xc);
        const Eigen::Vector2d unitNormal = normal / normalNorm;

        // _measurement is the closest point on the observed image curve.
        _error[0] = unitNormal.dot(projected - _measurement);
    }

    bool EdgeSE3ProjectBezierPointOnlyPose::isDepthPositive() const
    {
        assert(_vertices[0] != nullptr);

        const auto *v =
            static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        return v->estimate().map(Xw).z() > 0.0;
    }

    void EdgeSE3ProjectBezierPointOnlyPose::linearizeOplus()
    {
        assert(_vertices[0] != nullptr);
        assert(pCamera != nullptr);

        const auto *v =
            static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        const Eigen::Vector3d Xc = v->estimate().map(Xw);

        const double x = Xc.x();
        const double y = Xc.y();
        const double z = Xc.z();

        Eigen::Matrix<double, 3, 6> Jse3;
        Jse3 << 0, z, -y, 1, 0, 0,
            -z, 0, x, 0, 1, 0,
            y, -x, 0, 0, 0, 1;

        const double normalNorm = normal.norm();
        assert(normalNorm > 1e-12);
        if (normalNorm <= 1e-12)
        {
            _jacobianOplusXi.setZero();
            return;
        }

        const Eigen::Vector2d unitNormal = normal / normalNorm;
        _jacobianOplusXi =
            unitNormal.transpose() * pCamera->projectJac(Xc) * Jse3;
    }

    bool EdgeSE3ProjectBezierPointOnlyPose::read(std::istream &is)
    {
        is >> _measurement[0] >> _measurement[1]
           >> information()(0, 0)
           >> Xw[0] >> Xw[1] >> Xw[2]
           >> normal[0] >> normal[1];

        // pCamera cannot be serialized and must be restored by the caller.
        return static_cast<bool>(is);
    }

    bool EdgeSE3ProjectBezierPointOnlyPose::write(std::ostream &os) const
    {
        os << measurement()[0] << " "
           << measurement()[1] << " "
           << information()(0, 0) << " "
           << Xw[0] << " " << Xw[1] << " " << Xw[2] << " "
           << normal[0] << " " << normal[1];
        return os.good();
    }

    EdgeSE3ProjectBezierPoint::EdgeSE3ProjectBezierPoint()
        : normal(Eigen::Vector2d::Zero()),
          pCamera(nullptr)
    {
        information().setIdentity();
    }

    void EdgeSE3ProjectBezierPoint::computeError()
    {
        assert(_vertices[0] != nullptr);
        assert(_vertices[1] != nullptr);
        assert(pCamera != nullptr);

        const auto *vPoint =
            static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        const auto *vPose =
            static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);

        const double normalNorm = normal.norm();
        if (normalNorm <= 1e-12)
        {
            _error.setZero();
            return;
        }

        const Eigen::Vector2d projected =
            pCamera->project(vPose->estimate().map(vPoint->estimate()));
        const Eigen::Vector2d unitNormal = normal / normalNorm;

        _error[0] = unitNormal.dot(projected - _measurement);
    }

    bool EdgeSE3ProjectBezierPoint::isDepthPositive() const
    {
        assert(_vertices[0] != nullptr);
        assert(_vertices[1] != nullptr);

        const auto *vPoint =
            static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        const auto *vPose =
            static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);

        return vPose->estimate().map(vPoint->estimate()).z() > 0.0;
    }

    void EdgeSE3ProjectBezierPoint::linearizeOplus()
    {
        assert(_vertices[0] != nullptr);
        assert(_vertices[1] != nullptr);
        assert(pCamera != nullptr);

        const auto *vPoint =
            static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        const auto *vPose =
            static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);

        const g2o::SE3Quat T(vPose->estimate());
        const Eigen::Vector3d Xw = vPoint->estimate();
        const Eigen::Vector3d Xc = T.map(Xw);

        const double normalNorm = normal.norm();
        if (normalNorm <= 1e-12)
        {
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }

        const Eigen::Vector2d unitNormal = normal / normalNorm;
        const Eigen::Matrix<double, 2, 3> projectJac = pCamera->projectJac(Xc);

        _jacobianOplusXi =
            unitNormal.transpose() * projectJac * T.rotation().toRotationMatrix();

        const double x = Xc.x();
        const double y = Xc.y();
        const double z = Xc.z();

        Eigen::Matrix<double, 3, 6> Jse3;
        Jse3 << 0, z, -y, 1, 0, 0,
            -z, 0, x, 0, 1, 0,
            y, -x, 0, 0, 0, 1;

        _jacobianOplusXj =
            unitNormal.transpose() * projectJac * Jse3;
    }

    bool EdgeSE3ProjectBezierPoint::read(std::istream &is)
    {
        is >> _measurement[0] >> _measurement[1]
           >> information()(0, 0)
           >> normal[0] >> normal[1];

        return static_cast<bool>(is);
    }

    bool EdgeSE3ProjectBezierPoint::write(std::ostream &os) const
    {
        os << measurement()[0] << " "
           << measurement()[1] << " "
           << information()(0, 0) << " "
           << normal[0] << " " << normal[1];
        return os.good();
    }

}
