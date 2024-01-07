#include <clothpose/optimizer/Energy.hpp>
#include <clothpose/optimizer/Optimizer.hpp>
namespace cp
{
ARAPEnergy::ARAPEnergy(const TriMesh& restMesh, double weight,
                       double boundaryWeight)
    : restMesh{restMesh}, weight{weight}, boundaryWeight{boundaryWeight}
{
}

void ARAPEnergy::computeEnergy(Optimizer& optimizer, double& energy)
{
    auto& deformedMesh = *optimizer.meshObject;
    for (int vi = 0; vi < restMesh.vertices.rows(); vi++)
    {
        for (auto vj : restMesh.neighbours[vi])
        {
            Eigen::Vector3d pi = restMesh.vertices.row(vi);
            Eigen::Vector3d pj = restMesh.vertices.row(vj);
            Eigen::Vector3d qi = deformedMesh.vertices.row(vi);
            Eigen::Vector3d qj = deformedMesh.vertices.row(vj);
            auto Ri = rotations[vi];
            double w = restMesh.weights(vi, vj) * weight;
            auto& bv = deformedMesh.boundaryVertices;
            if ((bv.find(vi) != bv.end()) || (bv.find(vj) != bv.end()))
                w *= boundaryWeight;
            energy += w * ((qi - qj) - Ri * (pi - pj)).squaredNorm();
        }
    }
}

void ARAPEnergy::computeGradient(Optimizer& optimizer,
                                 Eigen::VectorXd& gradient)
{
    auto& deformedMesh = *optimizer.meshObject;
    if (weight == 0) return;
    for (int vi = 0; vi < restMesh.vertices.rows(); vi++)
    {
        for (auto vj : restMesh.neighbours[vi])
        {
            Eigen::Vector3d pi = restMesh.vertices.row(vi);
            Eigen::Vector3d pj = restMesh.vertices.row(vj);
            Eigen::Vector3d qi = deformedMesh.vertices.row(vi);
            Eigen::Vector3d qj = deformedMesh.vertices.row(vj);
            auto Ri = rotations[vi];
            double w = restMesh.weights(vi, vj) * weight;
            auto& bv = deformedMesh.boundaryVertices;
            if ((bv.find(vi) != bv.end()) || (bv.find(vj) != bv.end()))
                w *= boundaryWeight;
            Eigen::Vector3d v = w * ((qi - qj) - Ri * (pi - pj)).transpose();
            // increase the gradient
            for (int offset = 0; offset < 3; offset++)
            {
                gradient(3 * vi + offset) += v(offset);
                gradient(3 * vj + offset) -= v(offset);
            }
        }
    }
}

void ARAPEnergy::computeHessian(Optimizer& optimizer, LinearSolver& hessian)
{
    auto& deformedMesh = *optimizer.meshObject;

    for (int vi = 0; vi < restMesh.vertices.rows(); vi++)
    {
        for (auto vj : restMesh.neighbours[vi])
        {
            auto w = restMesh.weights(vi, vj) * weight;
            auto& bv = deformedMesh.boundaryVertices;
            if ((bv.find(vi) != bv.end()) || (bv.find(vj) != bv.end()))
                w *= boundaryWeight;
            for (int x = 0; x < 3; x++)
            {
                hessian.ptr->addCoeff(3 * vi + x, 3 * vi + x, 2 * w);
                hessian.ptr->addCoeff(3 * vj + x, 3 * vj + x, 2 * w);
                hessian.ptr->addCoeff(3 * vi + x, 3 * vj + x, -2 * w);
                hessian.ptr->addCoeff(3 * vj + x, 3 * vi + x, -2 * w);
            }
        }
    }
}

void ARAPEnergy::estimateLocalRotation(Optimizer& optimizer)
{
    auto& resultMesh = *optimizer.meshObject;
    // TODO: parallel computing
    rotations.resize(resultMesh.nVerts());
    for (int vi = 0; vi < restMesh.vertices.rows(); vi++)
    {
        Eigen::MatrixXd Pi, Qi, Ri, Di, Si;
        int Ni = restMesh.neighbours[vi].size();
        Pi.resize(Ni, 3);
        Qi.resize(Ni, 3);
        Di = Eigen::MatrixXd::Zero(Ni, Ni);
        for (int j = 0; j < Ni; j++)
        {
            int vj = restMesh.neighbours[vi][j];
            Di(j, j) = restMesh.weights(vi, vj);
            Pi(j, Eigen::indexing::all) =
                restMesh.vertices.row(vi) - restMesh.vertices.row(vj);
            Qi(j, Eigen::indexing::all) =
                resultMesh.vertices.row(vi) - resultMesh.vertices.row(vj);
        }
        Si = Pi.transpose() * Di * Qi;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            Si, Eigen::ComputeThinU | Eigen::ComputeThinV);
        auto V = svd.matrixV();
        auto U = svd.matrixU();
        if (V.determinant() < 0) V(Eigen::indexing::all, 2) *= -1;
        if (U.determinant() < 0) U(Eigen::indexing::all, 2) *= -1;
        Ri = V * U.transpose();  // svd.matrixV() * svd.matrixU().transpose();
        rotations[vi] = Ri;
    }
}
}