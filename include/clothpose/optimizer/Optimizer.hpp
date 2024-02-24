#pragma once
#include <clothpose/IPC/SpatialHash.hpp>
#include <clothpose/IPC/Utils/MeshCollisionUtils.hpp>
#include <clothpose/meshio/TriMesh.hpp>
#include <clothpose/optimizer/Energy.hpp>
#include <vector>
namespace cp
{
struct LinearSolver
{
    using LinearSolver_t = IPC::LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>;
    LinearSolver_t* ptr = nullptr;

    LinearSolver();
    LinearSolver(const LinearSolver& other) = delete;
    LinearSolver& operator=(const LinearSolver& other) = delete;
    LinearSolver(LinearSolver&& other);
    LinearSolver& operator=(LinearSolver&& other);
    ~LinearSolver();
};
class Optimizer
{
private:
    double newtonDirRes, dHat2;

    MeshObject meshObject;
    std::optional<PointsObject> pcd;
    std::optional<MeshObject> coarseMesh;
    std::size_t maxIterations, estimateRotationInterval;
    std::vector<Energy> energies;  // built-in: barrier, ARAP

    IPC::SpatialHash sh;
    std::vector<int> activeSet;
    std::vector<IPC::MMCVID> MMActiveSet, paraEEMMCVIDSet;
    std::vector<std::pair<int, int>> paraEEeIeJSet, MMActiveSet_CCD;
    Eigen::VectorXd searchDir, gradient;
    LinearSolver linSysSolver{};
    std::vector<std::set<int>> vNeighbor_IP;

public:
    Optimizer(const TriMesh& restMesh, MeshObject meshObject,
              double relDHat = 3e-3, double barrierWeight = 1e5,
              double ARAPWeight = 10.0, double ARAPBoundaryWeight = 1.0,
              std::vector<Energy> energyTerms = {},
              double relNewtonDirRes = 1e-2);
    Optimizer(Optimizer&& optimizer) = default;
    Optimizer& operator=(Optimizer&& optimizer) = default;
    void precompute();
    void setupIgnoredCollisionPairs(); 
    void clearIgnoredCollisionPairs(); 
    void solve(std::size_t maxIterations = 100,
               std::size_t estimateRotationInterval = 3);
    void step(bool doLineSearch = false);
    void updateSearchDir();
    void computeGradient(TriMesh& data, Eigen::VectorXd& gradient);
    void computePrecondMtr(TriMesh& data, LinearSolver& p_linSysSolver);
    double computeEnergyVal();
    void stepForward(Eigen::MatrixXd& dataV0, TriMesh& data, double stepSize);
    void lineSearch(double stepSize, double armijoParam = 1e-4,
                    double lowerBound = 0.0);
    void largestFeasibleStepSize_CCD(double slackness, double& stepSize,
                                     int maxIter = 128);
    void setPCD(PointsObject pcd);
    void setCoarseMesh(MeshObject mesh);

    friend class BarrierEnergy;
    friend class ARAPEnergy;
    friend class MeshToPointsCDEnergy;
    friend class PointsToMeshCDEnergy;
    friend class L2Energy;
};
}  // namespace cp