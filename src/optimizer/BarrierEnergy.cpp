#include <clothpose/IPC/BarrierFunctions.hpp>
#include <clothpose/IPC/Constraints.hpp>
#include <clothpose/optimizer/Energy.hpp>
#include <clothpose/optimizer/Optimizer.hpp>

namespace cp
{
BarrierEnergy::BarrierEnergy(double dHat, double weight)
    : dHat{dHat}, weight{weight}
{
}

void BarrierEnergy::computeEnergy(Optimizer& optimizer, double& energy)
{
    const auto& deformedMesh = *optimizer.meshObject;
    const auto& MMActiveSet = optimizer.MMActiveSet;
    const auto& paraEEMMCVIDSet = optimizer.paraEEMMCVIDSet;
    const auto& paraEEeIeJSet = optimizer.paraEEeIeJSet;
    Eigen::VectorXd constraintVals, bVals;

    int startCI = 0;
    IPC::evaluateConstraints(deformedMesh, MMActiveSet, constraintVals);
    bVals.conservativeResize(constraintVals.size());
    // TODO: parallelize
    for (int cI = startCI; cI < constraintVals.size(); ++cI)
    {
        if (constraintVals[cI] > 0.0)
        {
            IPC::compute_b(constraintVals[cI], dHat, bVals[cI]);
            int duplication = MMActiveSet[cI - startCI][3];
            if (duplication < -1)
            {
                // PP or PE, handle duplication
                bVals[cI] *= -duplication;
            }
        }
    }

    startCI = constraintVals.size();
    IPC::evaluateConstraints(deformedMesh, paraEEMMCVIDSet, constraintVals);
    bVals.conservativeResize(constraintVals.size());
    for (int cI = startCI; cI < constraintVals.size(); ++cI)
    {
        if (constraintVals[cI] > 0.0)
        {
            const auto& MMCVIDI = paraEEMMCVIDSet[cI - startCI];
            double eps_x, e;
            if (MMCVIDI[3] >= 0)
            {
                // EE
                IPC::compute_eps_x(deformedMesh, MMCVIDI[0], MMCVIDI[1],
                                   MMCVIDI[2], MMCVIDI[3], eps_x);
                IPC::compute_e(deformedMesh.vertices.row(MMCVIDI[0]),
                               deformedMesh.vertices.row(MMCVIDI[1]),
                               deformedMesh.vertices.row(MMCVIDI[2]),
                               deformedMesh.vertices.row(MMCVIDI[3]), eps_x, e);
            }
            else
            {
                // PP or PE
                const std::pair<int, int>& eIeJ = paraEEeIeJSet[cI - startCI];
                const std::pair<int, int>& eI = deformedMesh.edges[eIeJ.first];
                const std::pair<int, int>& eJ = deformedMesh.edges[eIeJ.second];
                IPC::compute_eps_x(deformedMesh, eI.first, eI.second, eJ.first,
                                   eJ.second, eps_x);
                IPC::compute_e(deformedMesh.vertices.row(eI.first),
                               deformedMesh.vertices.row(eI.second),
                               deformedMesh.vertices.row(eJ.first),
                               deformedMesh.vertices.row(eJ.second), eps_x, e);
            }
            IPC::compute_b(constraintVals[cI], dHat, bVals[cI]);
            bVals[cI] *= e;
        }
    }
    energy += weight * bVals.sum();
}

static void leftMultiplyConstraintJacobianT(
    const TriMesh& mesh, const std::vector<IPC::MMCVID>& activeSet,
    const Eigen::VectorXd& input, Eigen::VectorXd& output_incremental,
    double coef)
{
    // TODO: parallelize
    int constraintI = 0;
    for (const auto& MMCVIDI : activeSet)
    {
        if (MMCVIDI[0] >= 0)
        {
            // edge-edge
            Eigen::Matrix<double, 3 * 4, 1> g;
            IPC::g_EE(mesh.vertices.row(MMCVIDI[0]),
                      mesh.vertices.row(MMCVIDI[1]),
                      mesh.vertices.row(MMCVIDI[2]),
                      mesh.vertices.row(MMCVIDI[3]), g);
            g *= coef * input[constraintI];

            output_incremental.segment<3>(MMCVIDI[0] * 3) +=
                g.template segment<3>(0);
            output_incremental.segment<3>(MMCVIDI[1] * 3) +=
                g.template segment<3>(3);
            output_incremental.segment<3>(MMCVIDI[2] * 3) +=
                g.template segment<3>(3 * 2);
            output_incremental.segment<3>(MMCVIDI[3] * 3) +=
                g.template segment<3>(3 * 3);
        }
        else
        {
            // point-triangle and degenerate edge-edge
            assert(MMCVIDI[1] >= 0);

            int v0I = -MMCVIDI[0] - 1;
            if (MMCVIDI[2] < 0)
            {
                // PP
                Eigen::Matrix<double, 3 * 2, 1> g;
                IPC::g_PP(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          g);
                g *= coef * -MMCVIDI[3] * input[constraintI];

                output_incremental.segment<3>(v0I * 3) +=
                    g.template segment<3>(0);
                output_incremental.segment<3>(MMCVIDI[1] * 3) +=
                    g.template segment<3>(3);
            }
            else if (MMCVIDI[3] < 0)
            {
                // PE
                Eigen::Matrix<double, 3 * 3, 1> g;
                IPC::g_PE(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          mesh.vertices.row(MMCVIDI[2]), g);
                g *= coef * -MMCVIDI[3] * input[constraintI];

                output_incremental.segment<3>(v0I * 3) +=
                    g.template segment<3>(0);
                output_incremental.segment<3>(MMCVIDI[1] * 3) +=
                    g.template segment<3>(3);
                output_incremental.segment<3>(MMCVIDI[2] * 3) +=
                    g.template segment<3>(3 * 2);
            }
            else
            {
                // PT
                Eigen::Matrix<double, 3 * 4, 1> g;
                IPC::g_PT(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          mesh.vertices.row(MMCVIDI[2]),
                          mesh.vertices.row(MMCVIDI[3]), g);
                g *= coef * input[constraintI];

                output_incremental.segment<3>(v0I * 3) +=
                    g.template segment<3>(0);
                output_incremental.segment<3>(MMCVIDI[1] * 3) +=
                    g.template segment<3>(3);
                output_incremental.segment<3>(MMCVIDI[2] * 3) +=
                    g.template segment<3>(3 * 2);
                output_incremental.segment<3>(MMCVIDI[3] * 3) +=
                    g.template segment<3>(3 * 3);
            }
        }

        ++constraintI;
    }
}

static void augmentParaEEGradient(
    const TriMesh& deformedMesh,
    const std::vector<IPC::MMCVID>& paraEEMMCVIDSet,
    const std::vector<std::pair<int, int>>& paraEEeIeJSet,
    Eigen::VectorXd& grad_inc, double dHat, double coef)
{
    constexpr int dim = 3;

    Eigen::VectorXd e_db_div_dd;
    IPC::evaluateConstraints(deformedMesh, paraEEMMCVIDSet, e_db_div_dd);
    for (int cI = 0; cI < e_db_div_dd.size(); ++cI)
    {
        double b;
        IPC::compute_b(e_db_div_dd[cI], dHat, b);
        IPC::compute_g_b(e_db_div_dd[cI], dHat, e_db_div_dd[cI]);

        const auto& MMCVIDI = paraEEMMCVIDSet[cI];
        double eps_x, e;
        double coef_b = coef * b;
        if (MMCVIDI[3] >= 0)
        {
            // EE
            IPC::compute_eps_x(deformedMesh, MMCVIDI[0], MMCVIDI[1], MMCVIDI[2],
                               MMCVIDI[3], eps_x);
            IPC::compute_e(deformedMesh.vertices.row(MMCVIDI[0]),
                           deformedMesh.vertices.row(MMCVIDI[1]),
                           deformedMesh.vertices.row(MMCVIDI[2]),
                           deformedMesh.vertices.row(MMCVIDI[3]), eps_x, e);

            Eigen::Matrix<double, 12, 1> e_g;
            IPC::compute_e_g(deformedMesh.vertices.row(MMCVIDI[0]),
                             deformedMesh.vertices.row(MMCVIDI[1]),
                             deformedMesh.vertices.row(MMCVIDI[2]),
                             deformedMesh.vertices.row(MMCVIDI[3]), eps_x, e_g);
            grad_inc.template segment<dim>(MMCVIDI[0] * dim) +=
                coef_b * e_g.template segment<dim>(0);
            grad_inc.template segment<dim>(MMCVIDI[1] * dim) +=
                coef_b * e_g.template segment<dim>(dim);
            grad_inc.template segment<dim>(MMCVIDI[2] * dim) +=
                coef_b * e_g.template segment<dim>(dim * 2);
            grad_inc.template segment<dim>(MMCVIDI[3] * dim) +=
                coef_b * e_g.template segment<dim>(dim * 3);
        }
        else
        {
            // PP or PE
            const std::pair<int, int>& eIeJ = paraEEeIeJSet[cI];
            const std::pair<int, int>& eI = deformedMesh.edges[eIeJ.first];
            const std::pair<int, int>& eJ = deformedMesh.edges[eIeJ.second];
            IPC::compute_eps_x(deformedMesh, eI.first, eI.second, eJ.first,
                               eJ.second, eps_x);
            IPC::compute_e(deformedMesh.vertices.row(eI.first),
                           deformedMesh.vertices.row(eI.second),
                           deformedMesh.vertices.row(eJ.first),
                           deformedMesh.vertices.row(eJ.second), eps_x, e);

            Eigen::Matrix<double, 12, 1> e_g;
            IPC::compute_e_g(deformedMesh.vertices.row(eI.first),
                             deformedMesh.vertices.row(eI.second),
                             deformedMesh.vertices.row(eJ.first),
                             deformedMesh.vertices.row(eJ.second), eps_x, e_g);
            grad_inc.template segment<dim>(eI.first * dim) +=
                coef_b * e_g.template segment<dim>(0);
            grad_inc.template segment<dim>(eI.second * dim) +=
                coef_b * e_g.template segment<dim>(dim);
            grad_inc.template segment<dim>(eJ.first * dim) +=
                coef_b * e_g.template segment<dim>(dim * 2);
            grad_inc.template segment<dim>(eJ.second * dim) +=
                coef_b * e_g.template segment<dim>(dim * 3);
        }
        e_db_div_dd[cI] *= e;
    }
    leftMultiplyConstraintJacobianT(deformedMesh, paraEEMMCVIDSet, e_db_div_dd,
                                    grad_inc, coef);
}

void BarrierEnergy::computeGradient(Optimizer& optimizer,
                                    Eigen::VectorXd& gradient)
{
    if (weight == 0) return;
    const auto& deformedMesh = *optimizer.meshObject;
    const auto& MMActiveSet = optimizer.MMActiveSet;
    const auto& paraEEMMCVIDSet = optimizer.paraEEMMCVIDSet;
    const auto& paraEEeIeJSet = optimizer.paraEEeIeJSet;
    Eigen::VectorXd constraintVal;

    int startCI = 0;
    // MMActiveSet should be computed previously
    evaluateConstraints(deformedMesh, MMActiveSet, constraintVal);
    for (int cI = startCI; cI < constraintVal.size(); ++cI)
        IPC::compute_g_b(constraintVal[cI], dHat, constraintVal[cI]);
    assert(MMActiveSet.size() == constraintVal.size());
    leftMultiplyConstraintJacobianT(
        deformedMesh, MMActiveSet,
        constraintVal.segment(startCI, MMActiveSet.size()), gradient, weight);
    augmentParaEEGradient(deformedMesh, paraEEMMCVIDSet, paraEEeIeJSet,
                          gradient, dHat, weight);
}

static void augmentIPHessian(
    const TriMesh& mesh, const std::vector<IPC::MMCVID>& activeSet,
    IPC::LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* mtr_incremental,
    double dHat, double coef)
{
    constexpr int dim = 3;
    std::vector<Eigen::Matrix<double, 4 * dim, 4 * dim>> IPHessian(
        activeSet.size());
    std::vector<Eigen::Matrix<int, 4, 1>> rowIStart(activeSet.size());
    for (int cI = 0; cI < activeSet.size(); ++cI)
    {
        const auto& MMCVIDI = activeSet[cI];
        if (MMCVIDI[0] >= 0)
        {
            // edge-edge
            double d;
            IPC::d_EE(mesh.vertices.row(MMCVIDI[0]),
                      mesh.vertices.row(MMCVIDI[1]),
                      mesh.vertices.row(MMCVIDI[2]),
                      mesh.vertices.row(MMCVIDI[3]), d);
            Eigen::Matrix<double, dim * 4, 1> g;
            IPC::g_EE(mesh.vertices.row(MMCVIDI[0]),
                      mesh.vertices.row(MMCVIDI[1]),
                      mesh.vertices.row(MMCVIDI[2]),
                      mesh.vertices.row(MMCVIDI[3]), g);
            Eigen::Matrix<double, dim * 4, dim * 4> H;
            IPC::H_EE(mesh.vertices.row(MMCVIDI[0]),
                      mesh.vertices.row(MMCVIDI[1]),
                      mesh.vertices.row(MMCVIDI[2]),
                      mesh.vertices.row(MMCVIDI[3]), H);

            double g_b, H_b;
            IPC::compute_g_b(d, dHat, g_b);
            IPC::compute_H_b(d, dHat, H_b);

            IPHessian[cI] =
                ((coef * H_b) * g) * g.transpose() + (coef * g_b) * H;
            IPC::IglUtils::makePD(IPHessian[cI]);

            rowIStart[cI][0] = (MMCVIDI[0] * dim);
            rowIStart[cI][1] = (MMCVIDI[1] * dim);
            rowIStart[cI][2] = (MMCVIDI[2] * dim);
            rowIStart[cI][3] = (MMCVIDI[3] * dim);
        }
        else
        {
            // point-triangle and degenerate edge-edge
            assert(MMCVIDI[1] >= 0);

            int v0I = -MMCVIDI[0] - 1;
            if (MMCVIDI[2] < 0)
            {
                // PP
                double d;
                IPC::d_PP(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          d);
                Eigen::Matrix<double, dim * 2, 1> g;
                IPC::g_PP(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          g);
                Eigen::Matrix<double, dim * 2, dim * 2> H;
                IPC::H_PP(H);

                double g_b, H_b;
                IPC::compute_g_b(d, dHat, g_b);
                IPC::compute_H_b(d, dHat, H_b);

                double coef_dup = coef * -MMCVIDI[3];
                Eigen::Matrix<double, dim * 2, dim * 2> HessianBlock =
                    ((coef_dup * H_b) * g) * g.transpose() +
                    (coef_dup * g_b) * H;
                IPC::IglUtils::makePD(HessianBlock);
                IPHessian[cI].template block<dim * 2, dim * 2>(0, 0) =
                    HessianBlock;

                // TODO: add isAnchor for mesh vertices
                rowIStart[cI][0] = (v0I * dim);
                rowIStart[cI][1] = (MMCVIDI[1] * dim);
                rowIStart[cI][2] = -1;
                rowIStart[cI][3] = -1;
            }
            else if (MMCVIDI[3] < 0)
            {
                // PE
                double d;
                IPC::d_PE(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          mesh.vertices.row(MMCVIDI[2]), d);
                Eigen::Matrix<double, dim * 3, 1> g;
                IPC::g_PE(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          mesh.vertices.row(MMCVIDI[2]), g);
                Eigen::Matrix<double, dim * 3, dim * 3> H;
                IPC::H_PE(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          mesh.vertices.row(MMCVIDI[2]), H);

                double g_b, H_b;
                IPC::compute_g_b(d, dHat, g_b);
                IPC::compute_H_b(d, dHat, H_b);

                double coef_dup = coef * -MMCVIDI[3];
                Eigen::Matrix<double, dim * 3, dim * 3> HessianBlock =
                    ((coef_dup * H_b) * g) * g.transpose() +
                    (coef_dup * g_b) * H;
                IPC::IglUtils::makePD(HessianBlock);
                IPHessian[cI].block(0, 0, dim * 3, dim * 3) = HessianBlock;

                rowIStart[cI][0] = (v0I * dim);
                rowIStart[cI][1] = (MMCVIDI[1] * dim);
                rowIStart[cI][2] = (MMCVIDI[2] * dim);
                rowIStart[cI][3] = -1;
            }
            else
            {
                // PT
                double d;
                IPC::d_PT(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          mesh.vertices.row(MMCVIDI[2]),
                          mesh.vertices.row(MMCVIDI[3]), d);
                Eigen::Matrix<double, dim * 4, 1> g;
                IPC::g_PT(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          mesh.vertices.row(MMCVIDI[2]),
                          mesh.vertices.row(MMCVIDI[3]), g);
                Eigen::Matrix<double, dim * 4, dim * 4> H;
                IPC::H_PT(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          mesh.vertices.row(MMCVIDI[2]),
                          mesh.vertices.row(MMCVIDI[3]), H);

                double g_b, H_b;
                IPC::compute_g_b(d, dHat, g_b);
                IPC::compute_H_b(d, dHat, H_b);

                IPHessian[cI] =
                    ((coef * H_b) * g) * g.transpose() + (coef * g_b) * H;
                IPC::IglUtils::makePD(IPHessian[cI]);

                rowIStart[cI][0] = (v0I * dim);
                rowIStart[cI][1] = (MMCVIDI[1] * dim);
                rowIStart[cI][2] = (MMCVIDI[2] * dim);
                rowIStart[cI][3] = (MMCVIDI[3] * dim);
            }
        }
    }
    // TODO: parallelize
    for (int cI = 0; cI < activeSet.size(); ++cI)
    {
        for (int i = 0; i < rowIStart[cI].size(); ++i)
        {
            int rowIStartI = rowIStart[cI][i];
            if (rowIStartI >= 0)
            {
                for (int j = 0; j < rowIStart[cI].size(); ++j)
                {
                    int colIStartI = rowIStart[cI][j];
                    if (colIStartI >= 0)
                    {
                        mtr_incremental->addCoeff(
                            rowIStartI, colIStartI,
                            IPHessian[cI](i * dim, j * dim));
                        mtr_incremental->addCoeff(
                            rowIStartI, colIStartI + 1,
                            IPHessian[cI](i * dim, j * dim + 1));
                        mtr_incremental->addCoeff(
                            rowIStartI + 1, colIStartI,
                            IPHessian[cI](i * dim + 1, j * dim));
                        mtr_incremental->addCoeff(
                            rowIStartI + 1, colIStartI + 1,
                            IPHessian[cI](i * dim + 1, j * dim + 1));

                        mtr_incremental->addCoeff(
                            rowIStartI, colIStartI + 2,
                            IPHessian[cI](i * dim, j * dim + 2));
                        mtr_incremental->addCoeff(
                            rowIStartI + 1, colIStartI + 2,
                            IPHessian[cI](i * dim + 1, j * dim + 2));

                        mtr_incremental->addCoeff(
                            rowIStartI + 2, colIStartI,
                            IPHessian[cI](i * dim + 2, j * dim));
                        mtr_incremental->addCoeff(
                            rowIStartI + 2, colIStartI + 1,
                            IPHessian[cI](i * dim + 2, j * dim + 1));
                        mtr_incremental->addCoeff(
                            rowIStartI + 2, colIStartI + 2,
                            IPHessian[cI](i * dim + 2, j * dim + 2));
                    }
                }
            }
        }
    }
}

static void augmentParaEEHessian(
    const TriMesh& mesh, const std::vector<IPC::MMCVID>& paraEEMMCVIDSet,
    const std::vector<std::pair<int, int>>& paraEEeIeJSet,
    IPC::LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>* H_inc, double dHat,
    double coef)
{
    constexpr int dim = 3;
    std::vector<Eigen::Matrix<double, 4 * dim, 4 * dim>> PEEHessian(
        paraEEMMCVIDSet.size());
    std::vector<Eigen::Matrix<int, 4, 1>> rowIStart(paraEEMMCVIDSet.size());

    for (int cI = 0; cI < paraEEMMCVIDSet.size(); ++cI)
    {
        double b, g_b, H_b;

        double eps_x, e;
        Eigen::Matrix<double, 12, 1> e_g;
        Eigen::Matrix<double, 12, 12> e_H;

        const auto& MMCVIDI = paraEEMMCVIDSet[cI];
        double d;
        IPC::evaluateConstraint(mesh, MMCVIDI, d);
        Eigen::Matrix<double, 12, 1> grad_d;
        Eigen::Matrix<double, 12, 12> H_d;

        if (MMCVIDI[3] >= 0)
        {
            // EE
            IPC::compute_b(d, dHat, b);
            IPC::compute_g_b(d, dHat, g_b);
            IPC::compute_H_b(d, dHat, H_b);

            IPC::compute_eps_x(mesh, MMCVIDI[0], MMCVIDI[1], MMCVIDI[2],
                               MMCVIDI[3], eps_x);
            IPC::compute_e(mesh.vertices.row(MMCVIDI[0]),
                           mesh.vertices.row(MMCVIDI[1]),
                           mesh.vertices.row(MMCVIDI[2]),
                           mesh.vertices.row(MMCVIDI[3]), eps_x, e);
            IPC::compute_e_g(mesh.vertices.row(MMCVIDI[0]),
                             mesh.vertices.row(MMCVIDI[1]),
                             mesh.vertices.row(MMCVIDI[2]),
                             mesh.vertices.row(MMCVIDI[3]), eps_x, e_g);
            IPC::compute_e_H(mesh.vertices.row(MMCVIDI[0]),
                             mesh.vertices.row(MMCVIDI[1]),
                             mesh.vertices.row(MMCVIDI[2]),
                             mesh.vertices.row(MMCVIDI[3]), eps_x, e_H);

            IPC::g_EE(mesh.vertices.row(MMCVIDI[0]),
                      mesh.vertices.row(MMCVIDI[1]),
                      mesh.vertices.row(MMCVIDI[2]),
                      mesh.vertices.row(MMCVIDI[3]), grad_d);
            IPC::H_EE(mesh.vertices.row(MMCVIDI[0]),
                      mesh.vertices.row(MMCVIDI[1]),
                      mesh.vertices.row(MMCVIDI[2]),
                      mesh.vertices.row(MMCVIDI[3]), H_d);

            rowIStart[cI][0] = (MMCVIDI[0] * dim);
            rowIStart[cI][1] = (MMCVIDI[1] * dim);
            rowIStart[cI][2] = (MMCVIDI[2] * dim);
            rowIStart[cI][3] = (MMCVIDI[3] * dim);
        }
        else
        {
            // PP or PE
            IPC::compute_b(d, dHat, b);
            IPC::compute_g_b(d, dHat, g_b);
            IPC::compute_H_b(d, dHat, H_b);

            const std::pair<int, int>& eIeJ = paraEEeIeJSet[cI];
            const std::pair<int, int>& eI = mesh.edges[eIeJ.first];
            const std::pair<int, int>& eJ = mesh.edges[eIeJ.second];
            IPC::compute_eps_x(mesh, eI.first, eI.second, eJ.first, eJ.second,
                               eps_x);
            IPC::compute_e(mesh.vertices.row(eI.first),
                           mesh.vertices.row(eI.second),
                           mesh.vertices.row(eJ.first),
                           mesh.vertices.row(eJ.second), eps_x, e);
            IPC::compute_e_g(mesh.vertices.row(eI.first),
                             mesh.vertices.row(eI.second),
                             mesh.vertices.row(eJ.first),
                             mesh.vertices.row(eJ.second), eps_x, e_g);
            IPC::compute_e_H(mesh.vertices.row(eI.first),
                             mesh.vertices.row(eI.second),
                             mesh.vertices.row(eJ.first),
                             mesh.vertices.row(eJ.second), eps_x, e_H);

            rowIStart[cI][0] = (eI.first * dim);
            rowIStart[cI][1] = (eI.second * dim);
            rowIStart[cI][2] = (eJ.first * dim);
            rowIStart[cI][3] = (eJ.second * dim);

            int v0I = -MMCVIDI[0] - 1;
            if (MMCVIDI[2] >= 0)
            {
                // PE
                Eigen::Matrix<double, 9, 1> grad_d_PE;
                IPC::g_PE(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          mesh.vertices.row(MMCVIDI[2]), grad_d_PE);
                Eigen::Matrix<double, 9, 9> H_d_PE;
                IPC::H_PE(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          mesh.vertices.row(MMCVIDI[2]), H_d_PE);

                // fill in
                int ind[4] = {eI.first, eI.second, eJ.first, eJ.second};
                int indMap[3];
                for (int i = 0; i < 4; ++i)
                {
                    if (v0I == ind[i])
                    {
                        indMap[0] = i;
                    }
                    else if (MMCVIDI[1] == ind[i])
                    {
                        indMap[1] = i;
                    }
                    else if (MMCVIDI[2] == ind[i])
                    {
                        indMap[2] = i;
                    }
                }

                grad_d.setZero();
                H_d.setZero();
                for (int i = 0; i < 3; ++i)
                {
                    grad_d.template segment<dim>(indMap[i] * dim) =
                        grad_d_PE.template segment<dim>(i * dim);
                    for (int j = 0; j < 3; ++j)
                    {
                        H_d.template block<dim, dim>(indMap[i] * dim,
                                                     indMap[j] * dim) =
                            H_d_PE.template block<dim, dim>(i * dim, j * dim);
                    }
                }
            }
            else
            {
                // PP
                Eigen::Matrix<double, 6, 1> grad_d_PP;
                IPC::g_PP(mesh.vertices.row(v0I), mesh.vertices.row(MMCVIDI[1]),
                          grad_d_PP);
                Eigen::Matrix<double, 6, 6> H_d_PP;
                IPC::H_PP(H_d_PP);

                int ind[4] = {eI.first, eI.second, eJ.first, eJ.second};
                int indMap[2];
                for (int i = 0; i < 4; ++i)
                {
                    if (v0I == ind[i])
                    {
                        indMap[0] = i;
                    }
                    else if (MMCVIDI[1] == ind[i])
                    {
                        indMap[1] = i;
                    }
                }

                grad_d.setZero();
                H_d.setZero();
                for (int i = 0; i < 2; ++i)
                {
                    grad_d.template segment<dim>(indMap[i] * dim) =
                        grad_d_PP.template segment<dim>(i * dim);
                    for (int j = 0; j < 2; ++j)
                    {
                        H_d.template block<dim, dim>(indMap[i] * dim,
                                                     indMap[j] * dim) =
                            H_d_PP.template block<dim, dim>(i * dim, j * dim);
                    }
                }
            }
        }

        Eigen::Matrix<double, 12, 12> kappa_gradb_gradeT;
        kappa_gradb_gradeT = ((coef * g_b) * grad_d) * e_g.transpose();

        PEEHessian[cI] = kappa_gradb_gradeT + kappa_gradb_gradeT.transpose() +
                         (coef * b) * e_H +
                         ((coef * e * H_b) * grad_d) * grad_d.transpose() +
                         (coef * e * g_b) * H_d;
        IPC::IglUtils::makePD(PEEHessian[cI]);
    }

    // TODO: parallelize
    for (int cI = 0; cI < paraEEMMCVIDSet.size(); ++cI)
    {
        for (int i = 0; i < rowIStart[cI].size(); ++i)
        {
            int rowIStartI = rowIStart[cI][i];
            if (rowIStartI >= 0)
            {
                for (int j = 0; j < rowIStart[cI].size(); ++j)
                {
                    int colIStartI = rowIStart[cI][j];
                    if (colIStartI >= 0)
                    {
                        H_inc->addCoeff(rowIStartI, colIStartI,
                                        PEEHessian[cI](i * dim, j * dim));
                        H_inc->addCoeff(rowIStartI, colIStartI + 1,
                                        PEEHessian[cI](i * dim, j * dim + 1));
                        H_inc->addCoeff(rowIStartI + 1, colIStartI,
                                        PEEHessian[cI](i * dim + 1, j * dim));
                        H_inc->addCoeff(
                            rowIStartI + 1, colIStartI + 1,
                            PEEHessian[cI](i * dim + 1, j * dim + 1));
                        H_inc->addCoeff(rowIStartI, colIStartI + 2,
                                        PEEHessian[cI](i * dim, j * dim + 2));
                        H_inc->addCoeff(
                            rowIStartI + 1, colIStartI + 2,
                            PEEHessian[cI](i * dim + 1, j * dim + 2));

                        H_inc->addCoeff(rowIStartI + 2, colIStartI,
                                        PEEHessian[cI](i * dim + 2, j * dim));
                        H_inc->addCoeff(
                            rowIStartI + 2, colIStartI + 1,
                            PEEHessian[cI](i * dim + 2, j * dim + 1));
                        H_inc->addCoeff(
                            rowIStartI + 2, colIStartI + 2,
                            PEEHessian[cI](i * dim + 2, j * dim + 2));
                    }
                }
            }
        }
    }
}

void BarrierEnergy::computeHessian(Optimizer& optimizer, LinearSolver& hessian)
{
    if (weight == 0) return;
    const auto& deformedMesh = *optimizer.meshObject;
    const auto& MMActiveSet = optimizer.MMActiveSet;
    const auto& paraEEMMCVIDSet = optimizer.paraEEMMCVIDSet;
    const auto& paraEEeIeJSet = optimizer.paraEEeIeJSet;
    augmentIPHessian(deformedMesh, MMActiveSet, hessian.ptr, dHat, weight);
    augmentParaEEHessian(deformedMesh, paraEEMMCVIDSet, paraEEeIeJSet,
                         hessian.ptr, dHat, weight);
}
}  // namespace cp