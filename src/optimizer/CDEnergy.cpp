#include <tbb/tbb.h>

#include <clothpose/optimizer/Energy.hpp>
#include <clothpose/optimizer/Optimizer.hpp>
namespace cp
{
MeshToPointsCDEnergy::MeshToPointsCDEnergy(double weight) : weight{weight} {}

void MeshToPointsCDEnergy::computeEnergy(Optimizer& optimizer, double& energy)
{
    auto& pcd = optimizer.pcd;
    if (!pcd.has_value())
        throw std::runtime_error(
            "[MeshToPointsCDEnergy]\tPCD is not set before computing energy!");
    auto& meshObject = optimizer.meshObject;
    energies.resize(meshObject->nVerts());
    tbb::parallel_for(
        0, (int)meshObject->vertices.rows(), 1,
        [&pointInds = nearestPointInds, weight = weight, &energies = energies,
         &pcd = *pcd.value(), &vertices = meshObject->vertices](int vi)
        {
            double minDist2 = std::numeric_limits<double>::max();
            int ind;
            for (int pi = 0; pi < pcd.rows(); pi++)
            {
                double dist2 = (pcd.row(pi) - vertices.row(vi)).squaredNorm();
                if (dist2 < minDist2)
                {
                    minDist2 = dist2;
                    ind = pi;
                }
            }
            pointInds[vi] = ind;
            energies[vi] = minDist2 * weight;
        });
    for (auto e : energies) energy += e;
}

void MeshToPointsCDEnergy::computeGradient(Optimizer& optimizer,
                                           Eigen::VectorXd& gradient)
{
    auto& pcd = optimizer.pcd;
    if (!pcd.has_value())
        throw std::runtime_error(
            "[MeshToPointsCDEnergy]\tPCD is not set before computing "
            "gradient!");
    auto& meshObject = optimizer.meshObject;
    nearestPointInds.resize(optimizer.meshObject->nVerts());
    tbb::parallel_for(
        0, (int)meshObject->nVerts(), 1,
        [&inds = nearestPointInds, &vertices = meshObject->vertices,
         &pcd = *pcd.value()](int vi)
        {
            double minDist2 = std::numeric_limits<double>::max();
            int ind;
            for (int pi = 0; pi < pcd.rows(); pi++)
            {
                double dist2 = (pcd.row(pi) - vertices.row(vi)).squaredNorm();
                if (dist2 < minDist2)
                {
                    minDist2 = dist2;
                    ind = pi;
                }
            }
            inds[vi] = ind;
        });
    for (std::size_t vi = 0; vi < meshObject->nVerts(); vi++)
    {
        int ind = nearestPointInds[vi];
        auto gi =
            2 * weight * (meshObject->vertices.row(vi) - pcd.value()->row(ind));
        for (int d = 0; d < 3; d++) gradient[3 * vi + d] += gi[d];
    }
}

void MeshToPointsCDEnergy::computeHessian(Optimizer& optimizer,
                                          LinearSolver& hessian)
{
    auto& pcd = optimizer.pcd;
    if (!pcd.has_value())
        throw std::runtime_error(
            "[MeshToPointsCDEnergy]\tPCD is not set before computing hessian!");
    auto& meshObject = optimizer.meshObject;
    for (int vi = 0; vi < meshObject->vertices.rows(); vi++)
        for (int d = 0; d < 3; d++)
            hessian.ptr->addCoeff(3 * vi + d, 3 * vi + d, 2 * weight);
}

PointsToMeshCDEnergy::PointsToMeshCDEnergy(double weight) : weight{weight} {}

void PointsToMeshCDEnergy::computeEnergy(Optimizer& optimizer, double& energy)
{
    auto& pcd = optimizer.pcd;
    if (!pcd.has_value())
        throw std::runtime_error(
            "[MeshToPointsCDEnergy]\tPCD is not set before computing energy!");
    auto& meshObject = optimizer.meshObject;
    energies.resize(pcd.value()->rows());
    tbb::parallel_for(
        0, (int)pcd.value()->rows(), 1,
        [&inds = nearestVertInds, weight = weight, &energies = energies,
         &pcd = *pcd.value(), &vertices = meshObject->vertices](int pi)
        {
            double minDist2 = std::numeric_limits<double>::max();
            int ind;
            for (int vi = 0; vi < vertices.rows(); vi++)
            {
                double dist2 = (vertices.row(vi) - pcd.row(pi)).squaredNorm();
                if (dist2 < minDist2)
                {
                    minDist2 = dist2;
                    ind = vi;
                }
            }
            inds[pi] = ind;
            energies[pi] = minDist2 * weight;
        });
    for (auto e : energies) energy += e;
}

void PointsToMeshCDEnergy::computeGradient(Optimizer& optimizer,
                                           Eigen::VectorXd& gradient)
{
    auto& pcd = optimizer.pcd;
    if (!pcd.has_value())
        throw std::runtime_error(
            "[MeshToPointsCDEnergy]\tPCD is not set before computing "
            "gradient!");
    nearestVertInds.resize(pcd.value()->rows());
    auto& meshObject = optimizer.meshObject;
    tbb::parallel_for(
        0, (int)pcd.value()->rows(), 1,
        [&inds = nearestVertInds, &vertices = meshObject->vertices,
         &pcd = *pcd.value()](int pi)
        {
            double minDist2 = std::numeric_limits<double>::max();
            int ind;
            for (int vi = 0; vi < vertices.rows(); vi++)
            {
                double dist2 = (vertices.row(vi) - pcd.row(pi)).squaredNorm();
                if (dist2 < minDist2)
                {
                    minDist2 = dist2;
                    ind = vi;
                }
            }
            inds[pi] = ind;
        });
    for (int pi = 0; pi < pcd.value()->rows(); pi++)
    {
        int ind = nearestVertInds[pi];
        auto gi =
            2 * weight * (meshObject->vertices.row(ind) - pcd.value()->row(pi));
        for (int d = 0; d < 3; d++) gradient[3 * ind + d] += gi[d];
    }
}

void PointsToMeshCDEnergy::computeHessian(Optimizer& optimizer,
                                          LinearSolver& hessian)
{
    // should be called after computeGradient
    auto& pcd = optimizer.pcd;
    if (!pcd.has_value())
        throw std::runtime_error(
            "[MeshToPointsCDEnergy]\tPCD is not set before computing hessian!");
    for (std::size_t pi = 0; pi < pcd.value()->rows(); pi++)
    {
        auto vi = nearestVertInds[pi];
        for (int d = 0; d < 3; d++)
            hessian.ptr->addCoeff(3 * vi + d, 3 * vi + d, 2 * weight);
    }
}
}  // namespace cp