#include <clothpose/optimizer/Energy.hpp>
#include <clothpose/optimizer/Optimizer.hpp>
namespace cp
{
L2Energy::L2Energy(std::vector<std::size_t> coarseVertInds, double weight)
    : coarseVertInds{coarseVertInds}, weight{weight}
{
}

void L2Energy::computeEnergy(Optimizer& optimizer, double& energy)
{
    if (weight == 0) return;
    if (!optimizer.coarseMesh.has_value())
        throw std::runtime_error(
            "[L2Energy] no coarse mesh set when computing energy!");
    auto& coarseVerts = optimizer.coarseMesh.value()->vertices;
    auto& meshObject = optimizer.meshObject;
    for (std::size_t i = 0; i < meshObject->nVerts(); i++)
    {
        energy += weight * (meshObject->vertices.row(i) -
                            coarseVerts.row(coarseVertInds.at(i)))
                               .squaredNorm();
    }
}

void L2Energy::computeGradient(Optimizer& optimizer, Eigen::VectorXd& gradient)
{
    if (weight == 0) return;
    if (!optimizer.coarseMesh.has_value())
        throw std::runtime_error(
            "[L2Energy] no coarse mesh set when computing gradient!");
    auto& coarseVerts = optimizer.coarseMesh.value()->vertices;
    auto& meshObject = optimizer.meshObject;
    for (std::size_t i = 0; i < meshObject->nVerts(); i++)
    {
        auto gi = 2 *
                  (meshObject->vertices.row(i) -
                   coarseVerts.row(coarseVertInds.at(i))) *
                  weight;
        gradient[3 * i + 0] += gi[0];
        gradient[3 * i + 1] += gi[1];
        gradient[3 * i + 2] += gi[2];
    }
}

void L2Energy::computeHessian(Optimizer& optimizer, LinearSolver& hessian)
{
    if (weight == 0) return;
    if (!optimizer.coarseMesh.has_value())
        throw std::runtime_error(
            "[L2Energy] no coarse mesh set when computing gradient!");
    auto& meshObject = optimizer.meshObject;
    for (std::size_t i = 0; i < meshObject->nVerts(); i++)
    {
        hessian.ptr->addCoeff(3 * i, 3 * i, 2 * weight);
        hessian.ptr->addCoeff(3 * i + 1, 3 * i + 1, 2 * weight);
        hessian.ptr->addCoeff(3 * i + 2, 3 * i + 2, 2 * weight);
    }
}
}  // namespace cp