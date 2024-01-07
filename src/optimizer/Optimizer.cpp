#include <clothpose/IPC/Constraints.hpp>
#include <clothpose/optimizer/Optimizer.hpp>
#include <variant>
namespace cp
{
LinearSolver::LinearSolver() { ptr = LinearSolver_t::create(); }

LinearSolver::LinearSolver(LinearSolver&& other) { *this = std::move(other); }

LinearSolver& LinearSolver::operator=(LinearSolver&& other)
{
    std::swap(other.ptr, ptr);
    return *this;
}

LinearSolver::~LinearSolver()
{
    if (ptr != nullptr) delete ptr;
}

Optimizer::Optimizer(const TriMesh& restMesh, MeshObject meshObject,
                     double relDHat, double barrierWeight, double ARAPWeight,
                     double ARAPBoundaryWeight, std::vector<Energy> energyTerms,
                     double relNewtonDirRes)
    : meshObject{std::move(meshObject)}, energies{std::move(energyTerms)}
{
    auto bboxDiagSize2 = restMesh.matSpaceBBoxSize2();
    auto targetGRes =
        std::sqrt(bboxDiagSize2 * relNewtonDirRes * relNewtonDirRes);
    dHat2 = bboxDiagSize2 * relDHat * relDHat;
    energies.push_back(BarrierEnergy{dHat2, barrierWeight});
    energies.push_back(ARAPEnergy{restMesh, ARAPWeight, ARAPBoundaryWeight});
}

void Optimizer::precompute()
{
    sh.build(*meshObject, meshObject->avgEdgeLen() / 3.);
    computeConstraintSet(*meshObject, sh, dHat2, MMActiveSet, paraEEMMCVIDSet,
                         paraEEeIeJSet, true, MMActiveSet_CCD);
}

void Optimizer::solve(std::size_t maxIterations,
                      std::size_t estimateRotationInterval)
{
    for (int iterI = 0; iterI < maxIterations; iterI++)
    {
        // std::cout << "iteration " << iterI << std::endl;
        if (iterI % estimateRotationInterval == 0)
        {
            for (auto& energy : energies)
            {
                std::visit(
                    [this](auto& energy)
                    {
                        using EnergyType = std::decay_t<decltype(energy)>;
                        if constexpr (std::is_same_v<EnergyType, ARAPEnergy>)
                            energy.estimateLocalRotation(*this);
                    },
                    energy);
            }
        }
        step();
        if (searchDir.norm() < newtonDirRes) break;
    }
}

void Optimizer::step(bool doLineSearch)
{
    precompute();  // for barrier energy computation
    double alpha = 1.0, slackness = 0.8;
    updateSearchDir();

    // for barrier energy computation
    // TODO: if alpha_CFL is not tiny, use CFL instead of full CCD
    sh.build(*meshObject, searchDir, alpha, meshObject->avgEdgeLen() / 3.);
    largestFeasibleStepSize_CCD(slackness, alpha);

    if (doLineSearch)
        lineSearch(alpha);
    else
        stepForward(meshObject->vertices, *meshObject, alpha);
}

void Optimizer::updateSearchDir()
{
    computeGradient(*meshObject, gradient);
    computePrecondMtr(*meshObject, linSysSolver);

    // debug
    computeEnergyVal();
    linSysSolver.ptr->factorize();
    Eigen::VectorXd minusG = -gradient;
    linSysSolver.ptr->solve(minusG, searchDir);
}

void Optimizer::computeGradient(TriMesh& result, Eigen::VectorXd& gradient)
{
    gradient = Eigen::VectorXd::Zero(meshObject->nVerts() * 3);
    for (auto& e : energies)
        std::visit([this, &gradient](auto& e)
                   { e.computeGradient(*this, gradient); },
                   e);
}

void Optimizer::computePrecondMtr(TriMesh& result, LinearSolver& linSysSolver)
{
    std::vector<std::set<int>> vNeighbor_IP_new = meshObject->vNeighbor;
    augmentConnectivity(result, MMActiveSet, vNeighbor_IP_new);
    augmentConnectivity(result, paraEEMMCVIDSet, paraEEeIeJSet,
                        vNeighbor_IP_new);
    if (MMActiveSet.size() + paraEEeIeJSet.size())
    {
        if (vNeighbor_IP_new != vNeighbor_IP)
        {
            vNeighbor_IP = vNeighbor_IP_new;
            linSysSolver.ptr->set_pattern(vNeighbor_IP);
            linSysSolver.ptr->analyze_pattern();
        }
    }
    else if (vNeighbor_IP != meshObject->vNeighbor)
    {
        // no extra connectivity in this iteration but there is in the last
        // iteration
        vNeighbor_IP = meshObject->vNeighbor;
        linSysSolver.ptr->set_pattern(vNeighbor_IP);
        linSysSolver.ptr->analyze_pattern();
    }
    linSysSolver.ptr->setZero();

    for (auto& e : energies)
        std::visit([this, &linSysSolver](auto& e)
                   { e.computeHessian(*this, linSysSolver); },
                   e);
}

double Optimizer::computeEnergyVal()
{
    double energy = 0;
    for (auto& e : energies)
        std::visit([this, &energy](auto& e) { e.computeEnergy(*this, energy); },
                   e);
    return energy;
}

void Optimizer::stepForward(Eigen::MatrixXd& dataV0, TriMesh& data,
                            double stepSize)
{
    for (int vI = 0; vI < data.vertices.rows(); vI++)
        data.vertices.row(vI) =
            dataV0.row(vI) +
            stepSize * searchDir.segment<3>(vI * 3).transpose();
}

void Optimizer::lineSearch(double stepSize, double armijoParam,
                           double lowerBound)
{
    // for barrier
    sh.build(*meshObject, meshObject->avgEdgeLen() / 3.0);
    computeConstraintSet(*meshObject, sh, dHat2, MMActiveSet, paraEEMMCVIDSet,
                         paraEEeIeJSet, true, MMActiveSet_CCD);

    auto lastEnergy = computeEnergyVal();
    double c1m = 0.0;
    if (armijoParam > 0.0) c1m = armijoParam * searchDir.dot(gradient);
    Eigen::MatrixXd resultV0 = meshObject->vertices;
    stepForward(resultV0, *meshObject, stepSize);

    // for barrier
    sh.build(*meshObject, meshObject->avgEdgeLen() / 3.0);
    computeConstraintSet(*meshObject, sh, dHat2, MMActiveSet, paraEEMMCVIDSet,
                         paraEEeIeJSet, true, MMActiveSet_CCD);

    double testEnergy = computeEnergyVal();

    double LFStepSize = stepSize;
    while ((testEnergy > lastEnergy + stepSize * c1m) &&  // Armijo condition
           (stepSize > lowerBound))
    {
        stepSize /= 2.0;
        if (stepSize < 1e-6) break;
        stepForward(resultV0, *meshObject, stepSize);

        sh.build(*meshObject, meshObject->avgEdgeLen() / 3.0);
        computeConstraintSet(*meshObject, sh, dHat2, MMActiveSet,
                             paraEEMMCVIDSet, paraEEeIeJSet, true,
                             MMActiveSet_CCD);
        testEnergy = computeEnergyVal();
    }
}

void Optimizer::largestFeasibleStepSize_CCD(double slackness, double& stepSize,
                                            int maxIter)
{
    const int dim = 3;
    const TriMesh& mesh = *meshObject;
    const double CCDDistRatio = 1.0 - slackness;

    // point-point,edge,triangle
    Eigen::VectorXd largestAlphasPPET(mesh.vertices.rows());
    tbb::parallel_for(
        0, (int)mesh.vertices.rows(), 1,
        [&](int vI)
        {
            largestAlphasPPET[vI] = stepSize;

            std::unordered_set<int> vInds, edgeInds, triInds;

            sh.queryPointForTriangles(vI, triInds);
            // point-triangle
            for (const auto& sfI : triInds)
            {
                const Eigen::RowVector3i& sfVInd = mesh.triangles.row(sfI);
                if (!(vI == sfVInd[0] || vI == sfVInd[1] || vI == sfVInd[2]))
                {
                    double d_sqrt;
                    IPC::computePointTriD(mesh.vertices.row(vI),
                                          mesh.vertices.row(sfVInd[0]),
                                          mesh.vertices.row(sfVInd[1]),
                                          mesh.vertices.row(sfVInd[2]), d_sqrt);
                    d_sqrt = std::sqrt(d_sqrt);

                    double largestAlpha = stepSize;
                    if (CTCD::vertexFaceCTCD(
                            mesh.vertices.row(vI).transpose(),
                            mesh.vertices.row(sfVInd[0]).transpose(),
                            mesh.vertices.row(sfVInd[1]).transpose(),
                            mesh.vertices.row(sfVInd[2]).transpose(),
                            mesh.vertices.row(vI).transpose() +
                                searchDir.segment<dim>(vI * dim),
                            mesh.vertices.row(sfVInd[0]).transpose() +
                                searchDir.segment<dim>(sfVInd[0] * dim),
                            mesh.vertices.row(sfVInd[1]).transpose() +
                                searchDir.segment<dim>(sfVInd[1] * dim),
                            mesh.vertices.row(sfVInd[2]).transpose() +
                                searchDir.segment<dim>(sfVInd[2] * dim),
                            CCDDistRatio * d_sqrt, largestAlpha))
                    {
                        if (largestAlpha < 1.0e-6)
                        {
                            // use ACCD
                            std::cout << "PT CCD failed: " << vI << " "
                                      << sfVInd[0] << " " << sfVInd[1] << " "
                                      << sfVInd[2] << std::endl;
                            std::cout << "PT CCD failed positions: "
                                      << mesh.vertices.row(vI) << ";"
                                      << mesh.vertices.row(sfVInd[0]) << ";"
                                      << mesh.vertices.row(sfVInd[1]) << ";"
                                      << mesh.vertices.row(sfVInd[2])
                                      << std::endl;
                            std::cout << "PT CCD failed directions: "
                                      << searchDir.segment<dim>(vI * dim) << ";"
                                      << searchDir.segment<dim>(sfVInd[0] * dim)
                                      << ";"
                                      << searchDir.segment<dim>(sfVInd[1] * dim)
                                      << ";"
                                      << searchDir.segment<dim>(sfVInd[2] * dim)
                                      << std::endl;
                            std::cout
                                << "PT CCD failed grad: "
                                << gradient.segment<dim>(vI * dim) << ";"
                                << gradient.segment<dim>(sfVInd[0] * dim) << ";"
                                << gradient.segment<dim>(sfVInd[1] * dim) << ";"
                                << gradient.segment<dim>(sfVInd[2] * dim)
                                << std::endl;
                            double cur_t = 0;
                            Eigen::RowVector3d pv = mesh.vertices.row(vI);
                            Eigen::RowVector3d pt0 =
                                mesh.vertices.row(sfVInd[0]);
                            Eigen::RowVector3d pt1 =
                                mesh.vertices.row(sfVInd[1]);
                            Eigen::RowVector3d pt2 =
                                mesh.vertices.row(sfVInd[2]);

                            Eigen::RowVector3d dpv =
                                searchDir.segment<dim>(vI * dim);
                            Eigen::RowVector3d dpt0 =
                                searchDir.segment<dim>(sfVInd[0] * dim);
                            Eigen::RowVector3d dpt1 =
                                searchDir.segment<dim>(sfVInd[1] * dim);
                            Eigen::RowVector3d dpt2 =
                                searchDir.segment<dim>(sfVInd[2] * dim);

                            auto meanDir = (dpv + dpt0 + dpt1 + dpt2) * 0.25;
                            dpv -= meanDir;
                            dpt0 -= meanDir;
                            dpt1 -= meanDir;
                            dpt2 -= meanDir;

                            double relD =
                                std::max(std::max(dpt0.norm(), dpt1.norm()),
                                         dpt2.norm()) +
                                dpv.norm();
                            std::cout << "relD: " << relD << std::endl;
                            double dist;
                            IPC::computePointTriD(pv, pt0, pt1, pt2, dist);
                            std::cout << "dist: " << dist << std::endl;
                            double thresh = (1 - slackness) * dist;
                            double dt = thresh / relD;
                            int iter = 0;
                            while (iter++ < maxIter)
                            {
                                cur_t += dt;
                                pv += dt * dpv;
                                pt0 += dt * dpt0;
                                pt1 += dt * dpt1;
                                pt2 += dt * dpt2;
                                IPC::computePointTriD(pv, pt0, pt1, pt2, dist);

                                if (dist < thresh)
                                {
                                    if (cur_t > stepSize) cur_t = stepSize;
                                    break;
                                }
                                if (cur_t > stepSize)
                                {
                                    cur_t = stepSize;
                                    break;
                                }
                                dt = (1 - slackness) * dist / relD;
                            }
                            largestAlpha = cur_t;
                            std::cout << "pt accd largest alpha: " << cur_t
                                      << std::endl;
                        }
                        if (largestAlpha < largestAlphasPPET[vI])
                            largestAlphasPPET[vI] = largestAlpha;
                    }
                }
            }
        });
    stepSize = std::min(stepSize, largestAlphasPPET.minCoeff());

    // edge-edge
    Eigen::VectorXd largestAlphasEE(mesh.edges.size());
    tbb::parallel_for(
        0, (int)mesh.edges.size(), 1,
        [&](int eI)
        {
            const auto& meshEI = mesh.edges[eI];

            largestAlphasEE[eI] = stepSize;

            std::unordered_set<int> sEdgeInds;
            sh.queryEdgeForEdges(eI, sEdgeInds);
            for (const auto& eJ : sEdgeInds)
            {
                const auto& meshEJ = mesh.edges[eJ];
                if (!(meshEI.first == meshEJ.first ||
                      meshEI.first == meshEJ.second ||
                      meshEI.second == meshEJ.first ||
                      meshEI.second == meshEJ.second || eI > eJ))
                {
                    double d_sqrt;
                    IPC::computeEdgeEdgeD(mesh.vertices.row(meshEI.first),
                                          mesh.vertices.row(meshEI.second),
                                          mesh.vertices.row(meshEJ.first),
                                          mesh.vertices.row(meshEJ.second),
                                          d_sqrt);
                    d_sqrt = std::sqrt(d_sqrt);

                    double largestAlpha = 1.0;
                    if (CTCD::edgeEdgeCTCD(
                            mesh.vertices.row(meshEI.first).transpose(),
                            mesh.vertices.row(meshEI.second).transpose(),
                            mesh.vertices.row(meshEJ.first).transpose(),
                            mesh.vertices.row(meshEJ.second).transpose(),
                            mesh.vertices.row(meshEI.first).transpose() +
                                searchDir.segment<dim>(meshEI.first * dim),
                            mesh.vertices.row(meshEI.second).transpose() +
                                searchDir.segment<dim>(meshEI.second * dim),
                            mesh.vertices.row(meshEJ.first).transpose() +
                                searchDir.segment<dim>(meshEJ.first * dim),
                            mesh.vertices.row(meshEJ.second).transpose() +
                                searchDir.segment<dim>(meshEJ.second * dim),
                            CCDDistRatio * d_sqrt, largestAlpha))
                    {
                        if (largestAlpha < 1.0e-6)
                        {
                            // use ACCD
                            std::cout << "EE CCD failed: " << meshEI.first
                                      << " " << meshEI.second << " "
                                      << meshEJ.first << " " << meshEJ.second
                                      << std::endl;
                            std::cout << "EE CCD failed positions: "
                                      << mesh.vertices.row(meshEI.first) << ";"
                                      << mesh.vertices.row(meshEI.second) << ";"
                                      << mesh.vertices.row(meshEJ.first) << ";"
                                      << mesh.vertices.row(meshEJ.second)
                                      << std::endl;
                            std::cout
                                << "EE CCD failed directions: "
                                << searchDir.segment<dim>(meshEI.first * dim)
                                << ";"
                                << searchDir.segment<dim>(meshEI.second * dim)
                                << ";"
                                << searchDir.segment<dim>(meshEJ.first * dim)
                                << ";"
                                << searchDir.segment<dim>(meshEJ.second * dim)
                                << std::endl;
                            std::cout
                                << "EE CCD failed grad: "
                                << gradient.segment<dim>(meshEI.first * dim)
                                << ";"
                                << gradient.segment<dim>(meshEI.second * dim)
                                << ";"
                                << gradient.segment<dim>(meshEJ.first * dim)
                                << ";"
                                << gradient.segment<dim>(meshEJ.second * dim)
                                << std::endl;
                            double cur_t = 0;
                            Eigen::RowVector3d ei0 =
                                mesh.vertices.row(meshEI.first);
                            Eigen::RowVector3d ei1 =
                                mesh.vertices.row(meshEI.second);
                            Eigen::RowVector3d ej0 =
                                mesh.vertices.row(meshEJ.first);
                            Eigen::RowVector3d ej1 =
                                mesh.vertices.row(meshEJ.second);

                            Eigen::RowVector3d dei0 =
                                searchDir.segment<dim>(meshEI.first * dim);
                            Eigen::RowVector3d dei1 =
                                searchDir.segment<dim>(meshEI.second * dim);
                            Eigen::RowVector3d dej0 =
                                searchDir.segment<dim>(meshEJ.first * dim);
                            Eigen::RowVector3d dej1 =
                                searchDir.segment<dim>(meshEJ.second * dim);

                            auto meanDir = (dei0 + dei1 + dej0 + dej1) * 0.25;
                            dei0 -= meanDir;
                            dei1 -= meanDir;
                            dej0 -= meanDir;
                            dej1 -= meanDir;

                            double relD = std::max(dei0.norm(), dei1.norm()) +
                                          std::max(dej0.norm(), dej1.norm());
                            std::cout << "relD: " << relD << std::endl;
                            double dist;
                            IPC::computeEdgeEdgeD(ei0, ei1, ej0, ej1, dist);
                            std::cout << "dist: " << dist << std::endl;
                            double thresh = (1 - slackness) * dist;
                            double dt = thresh / relD;
                            int iter = 0;
                            while (iter++ < maxIter)
                            {
                                cur_t += dt;
                                ei0 += dt * dei0;
                                ei1 += dt * dei1;
                                ej0 += dt * dej0;
                                ej1 += dt * dej1;
                                IPC::computeEdgeEdgeD(ei0, ei1, ej0, ej1, dist);

                                if (dist < thresh)
                                {
                                    if (cur_t > stepSize) cur_t = stepSize;
                                    break;
                                }
                                if (cur_t > stepSize)
                                {
                                    cur_t = stepSize;
                                    break;
                                }
                                dt = (1 - slackness) * dist / relD;
                            }
                            largestAlpha = cur_t;
                            std::cout << "ee accd largest alpha: " << cur_t
                                      << std::endl;
                        }

                        if (largestAlpha < largestAlphasEE[eI])
                            largestAlphasEE[eI] = largestAlpha;
                    }
                }
            }
        });
    stepSize = std::min(stepSize, largestAlphasEE.minCoeff());
}

void Optimizer::setPCD(PointsObject pcd) { this->pcd = pcd; }

void Optimizer::setCoarseMesh(MeshObject mesh) { this->coarseMesh = mesh; }
}  // namespace cp