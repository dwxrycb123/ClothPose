#pragma once

#ifndef Constraint_hpp
#define Constraint_hpp

#include <Eigen/Eigen>
#include <clothpose/meshio/TriMesh.hpp>

#include "BarrierFunctions.hpp"
#include "CTCD/CTCD.h"
#include "SpatialHash.hpp"
#include "Utils/IglUtils.hpp"
#include "Utils/MeshCollisionUtils.hpp"
namespace IPC
{
// from SelfCollisionHandler.cpp from IPC
inline void computeConstraintSet(
    const cp::TriMesh& mesh, const SpatialHash& sh, double dHat,
    std::vector<MMCVID>& constraintSet, std::vector<MMCVID>& paraEEMMCVIDSet,
    std::vector<std::pair<int, int>>& paraEEeIeJSet, bool getPTEE,
    std::vector<std::pair<int, int>>& cs_PTEE)
{
    double sqrtDHat = std::sqrt(dHat);

    std::vector<std::vector<MMCVID>> constraintSetPT(mesh.vertices.rows());
    std::vector<std::vector<int>> cs_PT;
    if (getPTEE)
    {
        cs_PT.resize(mesh.vertices.rows());
    }

    tbb::parallel_for(
        0, (int)mesh.vertices.rows(), 1,
        [&](int vI)
        {
            std::unordered_set<int> triInds;
            sh.queryPointForTriangles(vI, mesh.vertices.row(vI), sqrtDHat,
                                      triInds);
            for (const auto& sfI : triInds)
            {
                const Eigen::RowVector3i& sfVInd = mesh.triangles.row(sfI);
                if (!(vI == sfVInd[0] || vI == sfVInd[1] || vI == sfVInd[2]))
                {
                    int dtype = dType_PT(mesh.vertices.row(vI),
                                         mesh.vertices.row(sfVInd[0]),
                                         mesh.vertices.row(sfVInd[1]),
                                         mesh.vertices.row(sfVInd[2]));
                    double d;
                    switch (dtype)
                    {
                        case 0:
                        {
                            d_PP(mesh.vertices.row(vI),
                                 mesh.vertices.row(sfVInd[0]), d);
                            if (d < dHat)
                            {
                                constraintSetPT[vI].emplace_back(
                                    -vI - 1, sfVInd[0], -1, -1);
                            }
                            break;
                        }

                        case 1:
                        {
                            d_PP(mesh.vertices.row(vI),
                                 mesh.vertices.row(sfVInd[1]), d);
                            if (d < dHat)
                            {
                                constraintSetPT[vI].emplace_back(
                                    -vI - 1, sfVInd[1], -1, -1);
                            }
                            break;
                        }

                        case 2:
                        {
                            d_PP(mesh.vertices.row(vI),
                                 mesh.vertices.row(sfVInd[2]), d);
                            if (d < dHat)
                            {
                                constraintSetPT[vI].emplace_back(
                                    -vI - 1, sfVInd[2], -1, -1);
                            }
                            break;
                        }

                        case 3:
                        {
                            d_PE(mesh.vertices.row(vI),
                                 mesh.vertices.row(sfVInd[0]),
                                 mesh.vertices.row(sfVInd[1]), d);
                            if (d < dHat)
                            {
                                constraintSetPT[vI].emplace_back(
                                    -vI - 1, sfVInd[0], sfVInd[1], -1);
                            }
                            break;
                        }

                        case 4:
                        {
                            d_PE(mesh.vertices.row(vI),
                                 mesh.vertices.row(sfVInd[1]),
                                 mesh.vertices.row(sfVInd[2]), d);
                            if (d < dHat)
                            {
                                constraintSetPT[vI].emplace_back(
                                    -vI - 1, sfVInd[1], sfVInd[2], -1);
                            }
                            break;
                        }

                        case 5:
                        {
                            d_PE(mesh.vertices.row(vI),
                                 mesh.vertices.row(sfVInd[2]),
                                 mesh.vertices.row(sfVInd[0]), d);
                            if (d < dHat)
                            {
                                constraintSetPT[vI].emplace_back(
                                    -vI - 1, sfVInd[2], sfVInd[0], -1);
                            }
                            break;
                        }

                        case 6:
                        {
                            d_PT(mesh.vertices.row(vI),
                                 mesh.vertices.row(sfVInd[0]),
                                 mesh.vertices.row(sfVInd[1]),
                                 mesh.vertices.row(sfVInd[2]), d);
                            if (d < dHat)
                            {
                                constraintSetPT[vI].emplace_back(
                                    -vI - 1, sfVInd[0], sfVInd[1], sfVInd[2]);
                            }
                            break;
                        }

                        default:
                            break;
                    }

                    if (getPTEE && d < dHat)
                    {
                        cs_PT[vI].emplace_back(sfI);
                    }
                }
            }
        });
    // edge-edge
    std::vector<std::vector<MMCVID>> constraintSetEE(mesh.edges.size());
    std::vector<std::vector<int>> cs_EE;
    if (getPTEE)
    {
        cs_EE.resize(mesh.edges.size());
    }
    tbb::parallel_for(
        0, (int)mesh.edges.size(), 1,
        [&](int eI)
        {
            const auto& meshEI = mesh.edges[eI];

            std::vector<int> edgeInds;
            sh.queryEdgeForEdgesWithBBoxCheck(
                mesh, eI, mesh.vertices.row(meshEI.first),
                mesh.vertices.row(meshEI.second), sqrtDHat, edgeInds, eI);
            for (const auto& eJ : edgeInds)
            {
                const auto& meshEJ = mesh.edges[eJ];
                if (!(meshEI.first == meshEJ.first ||
                      meshEI.first == meshEJ.second ||
                      meshEI.second == meshEJ.first ||
                      meshEI.second == meshEJ.second || eI > eJ))
                {
                    int dtype = dType_EE(mesh.vertices.row(meshEI.first),
                                         mesh.vertices.row(meshEI.second),
                                         mesh.vertices.row(meshEJ.first),
                                         mesh.vertices.row(meshEJ.second));
                    double EECrossSqNorm, eps_x;
                    computeEECrossSqNorm(mesh.vertices.row(meshEI.first),
                                         mesh.vertices.row(meshEI.second),
                                         mesh.vertices.row(meshEJ.first),
                                         mesh.vertices.row(meshEJ.second),
                                         EECrossSqNorm);
                    compute_eps_x(mesh, meshEI.first, meshEI.second,
                                  meshEJ.first, meshEJ.second, eps_x);
                    int add_e = (EECrossSqNorm < eps_x) ? -eJ - 2 : -1;
                    // == -1: regular,
                    // <= -2 && >= -mesh.SFEdges.size()-1: nearly parallel PP or
                    // PE,
                    // <= -mesh.SFEdges.size()-2: nearly parallel EE
                    double d;
                    switch (dtype)
                    {
                        case 0:
                        {
                            d_PP(mesh.vertices.row(meshEI.first),
                                 mesh.vertices.row(meshEJ.first), d);
                            if (d < dHat)
                            {
                                constraintSetEE[eI].emplace_back(
                                    -meshEI.first - 1, meshEJ.first, -1, add_e);
                            }
                            break;
                        }

                        case 1:
                        {
                            d_PP(mesh.vertices.row(meshEI.first),
                                 mesh.vertices.row(meshEJ.second), d);
                            if (d < dHat)
                            {
                                constraintSetEE[eI].emplace_back(
                                    -meshEI.first - 1, meshEJ.second, -1,
                                    add_e);
                            }
                            break;
                        }

                        case 2:
                        {
                            d_PE(mesh.vertices.row(meshEI.first),
                                 mesh.vertices.row(meshEJ.first),
                                 mesh.vertices.row(meshEJ.second), d);
                            if (d < dHat)
                            {
                                constraintSetEE[eI].emplace_back(
                                    -meshEI.first - 1, meshEJ.first,
                                    meshEJ.second, add_e);
                            }
                            break;
                        }

                        case 3:
                        {
                            d_PP(mesh.vertices.row(meshEI.second),
                                 mesh.vertices.row(meshEJ.first), d);
                            if (d < dHat)
                            {
                                constraintSetEE[eI].emplace_back(
                                    -meshEI.second - 1, meshEJ.first, -1,
                                    add_e);
                            }
                            break;
                        }

                        case 4:
                        {
                            d_PP(mesh.vertices.row(meshEI.second),
                                 mesh.vertices.row(meshEJ.second), d);
                            if (d < dHat)
                            {
                                constraintSetEE[eI].emplace_back(
                                    -meshEI.second - 1, meshEJ.second, -1,
                                    add_e);
                            }
                            break;
                        }

                        case 5:
                        {
                            d_PE(mesh.vertices.row(meshEI.second),
                                 mesh.vertices.row(meshEJ.first),
                                 mesh.vertices.row(meshEJ.second), d);
                            if (d < dHat)
                            {
                                constraintSetEE[eI].emplace_back(
                                    -meshEI.second - 1, meshEJ.first,
                                    meshEJ.second, add_e);
                            }
                            break;
                        }

                        case 6:
                        {
                            d_PE(mesh.vertices.row(meshEJ.first),
                                 mesh.vertices.row(meshEI.first),
                                 mesh.vertices.row(meshEI.second), d);
                            if (d < dHat)
                            {
                                constraintSetEE[eI].emplace_back(
                                    -meshEJ.first - 1, meshEI.first,
                                    meshEI.second, add_e);
                            }
                            break;
                        }

                        case 7:
                        {
                            d_PE(mesh.vertices.row(meshEJ.second),
                                 mesh.vertices.row(meshEI.first),
                                 mesh.vertices.row(meshEI.second), d);
                            if (d < dHat)
                            {
                                constraintSetEE[eI].emplace_back(
                                    -meshEJ.second - 1, meshEI.first,
                                    meshEI.second, add_e);
                            }
                            break;
                        }

                        case 8:
                        {
                            d_EE(mesh.vertices.row(meshEI.first),
                                 mesh.vertices.row(meshEI.second),
                                 mesh.vertices.row(meshEJ.first),
                                 mesh.vertices.row(meshEJ.second), d);
                            if (d < dHat)
                            {
                                if (add_e <= -2)
                                {
                                    constraintSetEE[eI].emplace_back(
                                        meshEI.first, meshEI.second,
                                        meshEJ.first,
                                        -meshEJ.second - mesh.edges.size() - 2);
                                }
                                else
                                {
                                    constraintSetEE[eI].emplace_back(
                                        meshEI.first, meshEI.second,
                                        meshEJ.first, meshEJ.second);
                                }
                            }
                            break;
                        }

                        default:
                            break;
                    }

                    if (getPTEE && d < dHat)
                    {
                        cs_EE[eI].emplace_back(eJ);
                    }
                }
            }
        });
    if (getPTEE)
    {
        cs_PTEE.resize(0);
        cs_PTEE.reserve(cs_PT.size() + cs_EE.size());
        for (int svI = 0; svI < cs_PT.size(); ++svI)
        {
            for (const auto& sfI : cs_PT[svI])
            {
                cs_PTEE.emplace_back(-svI - 1, sfI);
            }
        }
        for (int eI = 0; eI < cs_EE.size(); ++eI)
        {
            for (const auto& eJ : cs_EE[eI])
            {
                cs_PTEE.emplace_back(eI, eJ);
            }
        }
    }

    constraintSet.resize(0);
    constraintSet.reserve(constraintSetPT.size() + constraintSetEE.size());
    std::map<MMCVID, int> constraintCounter;
    for (const auto& csI : constraintSetPT)
    {
        for (const auto& cI : csI)
        {
            if (cI[3] < 0)
            {
                // PP or PE
                ++constraintCounter[cI];
            }
            else
            {
                constraintSet.emplace_back(cI);
            }
        }
    }
    paraEEMMCVIDSet.resize(0);
    paraEEeIeJSet.resize(0);
    int eI = 0;
    for (const auto& csI : constraintSetEE)
    {
        for (const auto& cI : csI)
        {
            if (cI[3] >= 0)
            {
                // regular EE
                constraintSet.emplace_back(cI);
            }
            else if (cI[3] == -1)
            {
                // regular PP or PE
                ++constraintCounter[cI];
            }
            else if (cI[3] >= -int(mesh.edges.size()) - 1)
            {
                // nearly parallel PP or PE
                paraEEMMCVIDSet.emplace_back(cI[0], cI[1], cI[2], -1);
                paraEEeIeJSet.emplace_back(eI, -cI[3] - 2);
            }
            else
            {
                // nearly parallel EE
                paraEEMMCVIDSet.emplace_back(cI[0], cI[1], cI[2],
                                             -cI[3] - mesh.edges.size() - 2);
                paraEEeIeJSet.emplace_back(-1, -1);
            }
        }
        ++eI;
    }

    constraintSet.reserve(constraintSet.size() + constraintCounter.size());
    for (const auto& ccI : constraintCounter)
    {
        constraintSet.emplace_back(
            MMCVID(ccI.first[0], ccI.first[1], ccI.first[2], -ccI.second));
    }
}

inline void evaluateConstraint(const cp::TriMesh& mesh, const MMCVID& MMCVIDI,
                               double& val, double coef = 1.0)
{
    if (MMCVIDI[0] >= 0)
    {
        // edge-edge
        d_EE(mesh.vertices.row(MMCVIDI[0]), mesh.vertices.row(MMCVIDI[1]),
             mesh.vertices.row(MMCVIDI[2]), mesh.vertices.row(MMCVIDI[3]), val);
    }
    else
    {
        // point-triangle and degenerate edge-edge
        assert(MMCVIDI[1] >= 0);
        if (MMCVIDI[2] < 0)
        {
            // PP
            d_PP(mesh.vertices.row(-MMCVIDI[0] - 1),
                 mesh.vertices.row(MMCVIDI[1]), val);
        }
        else if (MMCVIDI[3] < 0)
        {
            // PE
            d_PE(mesh.vertices.row(-MMCVIDI[0] - 1),
                 mesh.vertices.row(MMCVIDI[1]), mesh.vertices.row(MMCVIDI[2]),
                 val);
        }
        else
        {
            // PT
            d_PT(mesh.vertices.row(-MMCVIDI[0] - 1),
                 mesh.vertices.row(MMCVIDI[1]), mesh.vertices.row(MMCVIDI[2]),
                 mesh.vertices.row(MMCVIDI[3]), val);
        }
    }
}

inline void evaluateConstraints(const cp::TriMesh& mesh,
                                const std::vector<MMCVID>& activeSet,
                                Eigen::VectorXd& val, double coef = 1.0)
{
    const int constraintStartInd = val.size();  // 0
    val.conservativeResize(constraintStartInd +
                           activeSet.size());  // activeSet.size()
    for (int cI = 0; cI < activeSet.size(); ++cI)
    {
        evaluateConstraint(mesh, activeSet[cI], val[constraintStartInd + cI],
                           coef);
    }
}

inline void augmentConnectivity(const cp::TriMesh& mesh,
                                const std::vector<MMCVID>& activeSet,
                                std::vector<std::set<int>>& vNeighbor)
{
    // TODO: parallelize?

    for (const auto& MMCVIDI : activeSet)
    {
        if (MMCVIDI[0] >= 0)
        {
            // edge-edge
            vNeighbor[MMCVIDI[0]].insert(MMCVIDI[2]);
            vNeighbor[MMCVIDI[2]].insert(MMCVIDI[0]);
            vNeighbor[MMCVIDI[0]].insert(MMCVIDI[3]);
            vNeighbor[MMCVIDI[3]].insert(MMCVIDI[0]);
            vNeighbor[MMCVIDI[1]].insert(MMCVIDI[2]);
            vNeighbor[MMCVIDI[2]].insert(MMCVIDI[1]);
            vNeighbor[MMCVIDI[1]].insert(MMCVIDI[3]);
            vNeighbor[MMCVIDI[3]].insert(MMCVIDI[1]);
        }
        else
        {
            // point-triangle and degenerate edge-edge
            assert(MMCVIDI[1] >= 0);

            int v0I = -MMCVIDI[0] - 1;
            if (MMCVIDI[2] < 0)
            {
                // PP
                vNeighbor[v0I].insert(MMCVIDI[1]);
                vNeighbor[MMCVIDI[1]].insert(v0I);
            }
            else if (MMCVIDI[3] < 0)
            {
                // PE
                vNeighbor[v0I].insert(MMCVIDI[1]);
                vNeighbor[MMCVIDI[1]].insert(v0I);
                vNeighbor[v0I].insert(MMCVIDI[2]);
                vNeighbor[MMCVIDI[2]].insert(v0I);
            }
            else
            {
                // PT
                vNeighbor[v0I].insert(MMCVIDI[1]);
                vNeighbor[MMCVIDI[1]].insert(v0I);
                vNeighbor[v0I].insert(MMCVIDI[2]);
                vNeighbor[MMCVIDI[2]].insert(v0I);
                vNeighbor[v0I].insert(MMCVIDI[3]);
                vNeighbor[MMCVIDI[3]].insert(v0I);
            }
        }
    }
}

inline void augmentConnectivity(
    const cp::TriMesh& mesh, const std::vector<MMCVID>& paraEEMMCVIDSet,
    const std::vector<std::pair<int, int>>& paraEEeIeJSet,
    std::vector<std::set<int>>& vNeighbor)
{
    // TODO: parallelize?
    for (const auto& eIeJ : paraEEeIeJSet)
    {
        if (eIeJ.first < 0 || eIeJ.second < 0)
        {
            continue;
        }
        // PE or PP
        const std::pair<int, int>& eI = mesh.edges[eIeJ.first];
        const std::pair<int, int>& eJ = mesh.edges[eIeJ.second];
        vNeighbor[eI.first].insert(eJ.first);
        vNeighbor[eJ.first].insert(eI.first);
        vNeighbor[eI.first].insert(eJ.second);
        vNeighbor[eJ.second].insert(eI.first);
        vNeighbor[eI.second].insert(eJ.first);
        vNeighbor[eJ.first].insert(eI.second);
        vNeighbor[eI.second].insert(eJ.second);
        vNeighbor[eJ.second].insert(eI.second);
    }
    for (const auto& MMCVIDI : paraEEMMCVIDSet)
    {
        if (MMCVIDI[0] < 0)
        {
            continue;
        }
        // EE
        vNeighbor[MMCVIDI[0]].insert(MMCVIDI[2]);
        vNeighbor[MMCVIDI[2]].insert(MMCVIDI[0]);
        vNeighbor[MMCVIDI[0]].insert(MMCVIDI[3]);
        vNeighbor[MMCVIDI[3]].insert(MMCVIDI[0]);
        vNeighbor[MMCVIDI[1]].insert(MMCVIDI[2]);
        vNeighbor[MMCVIDI[2]].insert(MMCVIDI[1]);
        vNeighbor[MMCVIDI[1]].insert(MMCVIDI[3]);
        vNeighbor[MMCVIDI[3]].insert(MMCVIDI[1]);
    }
}
}  // namespace IPC
#endif