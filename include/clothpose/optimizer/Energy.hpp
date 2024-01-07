#pragma once
#include <clothpose/IPC/LinSysSolver/LinSysSolver.hpp>
#include <clothpose/meshio/TriMesh.hpp>
#include <optional>
#include <variant>

namespace cp
{
class Optimizer;
class LinearSolver;

template <class Derived>
class BaseEnergy
{
    using type = Derived;

public:
    BaseEnergy() = default;
    virtual void computeEnergy(Optimizer& optimizer, double& energy) = 0;
    virtual void computeGradient(Optimizer& optimizer,
                                 Eigen::VectorXd& gradient) = 0;
    virtual void computeHessian(Optimizer& optimizer,
                                LinearSolver& hessian) = 0;
};

class ARAPEnergy : public BaseEnergy<ARAPEnergy>
{
    TriMesh restMesh;
    std::vector<Eigen::Matrix3d> rotations;
    double weight;
    double boundaryWeight;

public:
    ARAPEnergy(const TriMesh& restMesh, double weight, double boundaryWeight);
    void computeEnergy(Optimizer& optimizer, double& energy) override;
    void computeGradient(Optimizer& optimizer,
                         Eigen::VectorXd& gradient) override;
    void computeHessian(Optimizer& optimizer, LinearSolver& hessian) override;
    void estimateLocalRotation(Optimizer& optimizer);
};

class L2Energy : public BaseEnergy<L2Energy>
{
    std::vector<std::size_t> coarseVertInds;
    double weight;

public:
    L2Energy(std::vector<std::size_t> coarseVertInds, double weight = 1.0);
    void computeEnergy(Optimizer& optimizer, double& energy) override;
    void computeGradient(Optimizer& optimizer,
                         Eigen::VectorXd& gradient) override;
    void computeHessian(Optimizer& optimizer, LinearSolver& hessian) override;
};

class MeshToPointsCDEnergy : public BaseEnergy<MeshToPointsCDEnergy>
{
    std::vector<std::size_t> nearestPointInds;
    std::vector<double> energies;
    double weight;

public:
    MeshToPointsCDEnergy(double weight = 1.0);
    void computeEnergy(Optimizer& optimizer, double& energy) override;
    void computeGradient(Optimizer& optimizer,
                         Eigen::VectorXd& gradient) override;
    void computeHessian(Optimizer& optimizer, LinearSolver& hessian) override;
};

class PointsToMeshCDEnergy : public BaseEnergy<PointsToMeshCDEnergy>
{
    std::vector<std::size_t> nearestVertInds;
    std::vector<double> energies;
    double weight;

public:
    PointsToMeshCDEnergy(double weight = 1.0);
    void computeEnergy(Optimizer& optimizer, double& energy) override;
    void computeGradient(Optimizer& optimizer,
                         Eigen::VectorXd& gradient) override;
    void computeHessian(Optimizer& optimizer, LinearSolver& hessian) override;
};

class BarrierEnergy : public BaseEnergy<BarrierEnergy>
{
    double dHat, weight;

public:
    BarrierEnergy(double dHat, double weight);
    void computeEnergy(Optimizer& optimizer, double& energy) override;
    void computeGradient(Optimizer& optimizer,
                         Eigen::VectorXd& gradient) override;
    void computeHessian(Optimizer& optimizer, LinearSolver& hessian) override;
};

using Energy = std::variant<ARAPEnergy, L2Energy, MeshToPointsCDEnergy,
                            PointsToMeshCDEnergy, BarrierEnergy>;
}