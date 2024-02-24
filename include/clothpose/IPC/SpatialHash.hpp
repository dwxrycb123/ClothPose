//
//  SpatialHash.hpp
//  IPC
//
//  Created by Minchen Li on 6/26/19.
//

#ifndef SpatialHash_hpp
#define SpatialHash_hpp

#include <tbb/tbb.h>

#include <cassert>
#include <clothpose/meshio/TriMesh.hpp>
#include <clothpose/meta.hpp>
#include <unordered_map>
#include <unordered_set>
namespace IPC
{
class SpatialHash
{
public:  // data
    Eigen::Matrix<double, 1, 3> leftBottomCorner, rightTopCorner;
    double one_div_voxelSize;
    Eigen::Array<int, 1, 3> voxelCount;
    int voxelCount0x1;

    int surfEdgeStartInd, surfTriStartInd;

    std::unordered_map<int, std::vector<int>> voxel;
    std::vector<std::vector<int>> pointAndEdgeOccupancy;
    std::unordered_set<std::tuple<int, int>, cp::TupleHash<int, int>>
        ignoredPTs, ignoredEEs;

public:  // constructor
    SpatialHash(void) {}
    SpatialHash(const cp::TriMesh& mesh, double voxelSize)
    {
        build(mesh, voxelSize);
    }
    SpatialHash(const cp::TriMesh& mesh, const Eigen::VectorXd& searchDir,
                double curMaxStepSize, double voxelSize)
    {
        build(mesh, searchDir, curMaxStepSize, voxelSize);
    }

public:  // API
    void addIgnoredPTs(int pointIndex, int triangleIndex)
    {
        ignoredPTs.insert(std::make_tuple(pointIndex, triangleIndex));
    }
    void addIgnoredEEs(int firstEdgeIndex, int secondEdgeIndex)
    {
        ignoredEEs.insert(std::make_tuple(firstEdgeIndex, secondEdgeIndex));
    }
    void clearIgnoredPairs()
    {
        ignoredEEs.clear();
        ignoredPTs.clear();
    }
    void build(const cp::TriMesh& mesh, double voxelSize)
    {
        const Eigen::MatrixXd& V = mesh.vertices;
        leftBottomCorner = V.colwise().minCoeff();
        rightTopCorner = V.colwise().maxCoeff();
        one_div_voxelSize = 1.0 / voxelSize;
        Eigen::Array<double, 1, 3> range = rightTopCorner - leftBottomCorner;
        voxelCount = (range * one_div_voxelSize).ceil().template cast<int>();
        if (voxelCount.minCoeff() <= 0)
        {
            // cast overflow due to huge search direction
            one_div_voxelSize = 1.0 / (range.maxCoeff() * 1.01);
            voxelCount.setOnes();
        }
        voxelCount0x1 = voxelCount[0] * voxelCount[1];

        surfEdgeStartInd = V.rows();
        surfTriStartInd = surfEdgeStartInd + mesh.edges.size();

        std::vector<Eigen::Array<int, 1, 3>> svVoxelAxisIndex(V.rows());
        tbb::parallel_for(
            0, (int)V.rows(), 1, [&](int vI)
            { locateVoxelAxisIndex(V.row(vI), svVoxelAxisIndex[vI]); });

        voxel.clear();
#if 0 
// no parallel 
		for (int vI = 0; vI < V.rows(); ++vI) {
			voxel[locateVoxelIndex(V.row(vI))].emplace_back(vI);
		}
#else
        // parallel using tbb
        std::vector<std::pair<int, int>> voxel_tmp;
        for (int vI = 0; vI < V.rows(); ++vI)
        {
            voxel_tmp.emplace_back(locateVoxelIndex(V.row(vI)), vI);
        }
#endif
        // timer_mt.start(16);
        std::vector<std::vector<int>> voxelLoc_e(
            mesh.edges.size());  //(mesh.SFEdges.size());
        tbb::parallel_for(
            0, (int)mesh.edges.size(), 1,
            [&](int seCount)
            {
                const auto& seI = mesh.edges[seCount];
                // const auto& seI = mesh.SFEdges[seCount];

                const Eigen::Array<int, 1, 3>& voxelAxisIndex_first =
                    svVoxelAxisIndex[seI.first];
                const Eigen::Array<int, 1, 3>& voxelAxisIndex_second =
                    svVoxelAxisIndex[seI.second];
                Eigen::Array<int, 1, 3> mins =
                    voxelAxisIndex_first.min(voxelAxisIndex_second);
                Eigen::Array<int, 1, 3> maxs =
                    voxelAxisIndex_first.max(voxelAxisIndex_second);
                for (int iz = mins[2]; iz <= maxs[2]; ++iz)
                {
                    int zOffset = iz * voxelCount0x1;
                    for (int iy = mins[1]; iy <= maxs[1]; ++iy)
                    {
                        int yzOffset = iy * voxelCount[0] + zOffset;
                        for (int ix = mins[0]; ix <= maxs[0]; ++ix)
                        {
                            voxelLoc_e[seCount].emplace_back(ix + yzOffset);
                        }
                    }
                }
            });

        std::vector<std::vector<int>> voxelLoc_sf(mesh.triangles.rows());
        tbb::parallel_for(
            0, (int)mesh.triangles.rows(), 1,
            [&](int sfI)
            {
                const Eigen::Array<int, 1, 3>& voxelAxisIndex0 =
                    svVoxelAxisIndex[mesh.triangles(sfI, 0)];
                const Eigen::Array<int, 1, 3>& voxelAxisIndex1 =
                    svVoxelAxisIndex[mesh.triangles(sfI, 1)];
                const Eigen::Array<int, 1, 3>& voxelAxisIndex2 =
                    svVoxelAxisIndex[mesh.triangles(sfI, 2)];
                Eigen::Array<int, 1, 3> mins =
                    voxelAxisIndex0.min(voxelAxisIndex1).min(voxelAxisIndex2);
                Eigen::Array<int, 1, 3> maxs =
                    voxelAxisIndex0.max(voxelAxisIndex1).max(voxelAxisIndex2);
                for (int iz = mins[2]; iz <= maxs[2]; ++iz)
                {
                    int zOffset = iz * voxelCount0x1;
                    for (int iy = mins[1]; iy <= maxs[1]; ++iy)
                    {
                        int yzOffset = iy * voxelCount[0] + zOffset;
                        for (int ix = mins[0]; ix <= maxs[0]; ++ix)
                        {
                            voxelLoc_sf[sfI].emplace_back(ix + yzOffset);
                        }
                    }
                }
            });
#if 1
        // parallel using tbb
        for (int seCount = 0; seCount < voxelLoc_e.size(); ++seCount)
        {
            for (const auto& voxelI : voxelLoc_e[seCount])
            {
                voxel_tmp.emplace_back(voxelI, seCount + surfEdgeStartInd);
            }
        }

        for (int sfI = 0; sfI < voxelLoc_sf.size(); ++sfI)
        {
            for (const auto& voxelI : voxelLoc_sf[sfI])
            {
                voxel_tmp.emplace_back(voxelI, sfI + surfTriStartInd);
            }
        }
        tbb::parallel_sort(
            voxel_tmp.begin(), voxel_tmp.end(),
            [](const std::pair<int, int>& f, const std::pair<int, int>& s)
            { return f.first < s.first; });

        std::vector<std::pair<int, std::vector<int>>> voxel_tmp_merged;
        voxel_tmp_merged.reserve(voxel_tmp.size());
        int current_voxel = -1;
        for (const auto& v : voxel_tmp)
        {
            if (current_voxel != v.first)
            {
                assert(current_voxel < v.first);
                voxel_tmp_merged.emplace_back();
                voxel_tmp_merged.back().first = v.first;
                current_voxel = v.first;
            }

            voxel_tmp_merged.back().second.push_back(v.second);
        }
        assert(voxel_tmp_merged.size() <= voxel_tmp.size());

        voxel.insert(voxel_tmp_merged.begin(), voxel_tmp_merged.end());
#else
        // no parallel
        for (int seCount = 0; seCount < voxelLoc_e.size(); ++seCount)
        {
            for (const auto& voxelI : voxelLoc_e[seCount])
            {
                voxel[voxelI].emplace_back(seCount + surfEdgeStartInd);
            }
        }
        for (int sfI = 0; sfI < voxelLoc_sf.size(); ++sfI)
        {
            for (const auto& voxelI : voxelLoc_sf[sfI])
            {
                voxel[voxelI].emplace_back(sfI + surfTriStartInd);
            }
        }
#endif
        // timer_mt.stop();
    }
    void queryPointForTriangles(int vI, const Eigen::Matrix<double, 1, 3>& pos,
                                double radius,
                                std::unordered_set<int>& triInds) const
    {
        Eigen::Array<int, 1, 3> mins, maxs;
        locateVoxelAxisIndex(pos.array() - radius, mins);
        locateVoxelAxisIndex(pos.array() + radius, maxs);
        mins = mins.max(Eigen::Array<int, 1, 3>::Zero());
        maxs = maxs.min(voxelCount - 1);

        triInds.clear();
        for (int iz = mins[2]; iz <= maxs[2]; ++iz)
        {
            int zOffset = iz * voxelCount0x1;
            for (int iy = mins[1]; iy <= maxs[1]; ++iy)
            {
                int yzOffset = iy * voxelCount[0] + zOffset;
                for (int ix = mins[0]; ix <= maxs[0]; ++ix)
                {
                    const auto voxelI = voxel.find(ix + yzOffset);
                    if (voxelI != voxel.end())
                    {
                        for (const auto& indI : voxelI->second)
                        {
                            if (indI >= surfTriStartInd)
                            {
                                if (ignoredPTs.find(
                                        {vI, indI - surfTriStartInd}) ==
                                    ignoredPTs.end())
                                    triInds.insert(indI - surfTriStartInd);
                            }
                        }
                    }
                }
            }
        }
    }
    void queryEdgeForEdges(int eI, const Eigen::Matrix<double, 1, 3>& vBegin,
                           const Eigen::Matrix<double, 1, 3>& vEnd,
                           double radius, std::vector<int>& edgeInds,
                           int eIq = -1) const
    {
        // timer_mt.start(19);
        Eigen::Matrix<double, 1, 3> leftBottom =
            vBegin.array().min(vEnd.array()) - radius;
        Eigen::Matrix<double, 1, 3> rightTop =
            vBegin.array().max(vEnd.array()) + radius;
        Eigen::Array<int, 1, 3> mins, maxs;
        locateVoxelAxisIndex(leftBottom, mins);
        locateVoxelAxisIndex(rightTop, maxs);
        mins = mins.max(Eigen::Array<int, 1, 3>::Zero());
        maxs = maxs.min(voxelCount - 1);

        // timer_mt.start(20);
        edgeInds.resize(0);
        for (int iz = mins[2]; iz <= maxs[2]; ++iz)
        {
            int zOffset = iz * voxelCount0x1;
            for (int iy = mins[1]; iy <= maxs[1]; ++iy)
            {
                int yzOffset = iy * voxelCount[0] + zOffset;
                for (int ix = mins[0]; ix <= maxs[0]; ++ix)
                {
                    const auto voxelI = voxel.find(ix + yzOffset);
                    if (voxelI != voxel.end())
                    {
                        for (const auto& indI : voxelI->second)
                        {
                            if (indI >= surfEdgeStartInd &&
                                indI < surfTriStartInd &&
                                indI - surfEdgeStartInd > eIq)
                            {
                                if (ignoredEEs.find(
                                        {eI, indI - surfEdgeStartInd}) ==
                                    ignoredEEs.end())
                                    edgeInds.emplace_back(indI -
                                                          surfEdgeStartInd);
                            }
                        }
                    }
                }
            }
        }
        std::sort(edgeInds.begin(), edgeInds.end());
        edgeInds.erase(std::unique(edgeInds.begin(), edgeInds.end()),
                       edgeInds.end());
    }
    void queryEdgeForEdgesWithBBoxCheck(
        const cp::TriMesh& mesh, int eI,
        const Eigen::Matrix<double, 1, 3>& vBegin,
        const Eigen::Matrix<double, 1, 3>& vEnd, double radius,
        std::vector<int>& edgeInds, int eIq = -1) const
    {
        Eigen::Matrix<double, 1, 3> leftBottom =
            vBegin.array().min(vEnd.array()) - radius;
        Eigen::Matrix<double, 1, 3> rightTop =
            vBegin.array().max(vEnd.array()) + radius;
        Eigen::Array<int, 1, 3> mins, maxs;
        locateVoxelAxisIndex(leftBottom, mins);
        locateVoxelAxisIndex(rightTop, maxs);
        mins = mins.max(Eigen::Array<int, 1, 3>::Zero());
        maxs = maxs.min(voxelCount - 1);

        edgeInds.resize(0);
        for (int iz = mins[2]; iz <= maxs[2]; ++iz)
        {
            int zOffset = iz * voxelCount0x1;
            for (int iy = mins[1]; iy <= maxs[1]; ++iy)
            {
                int yzOffset = iy * voxelCount[0] + zOffset;
                for (int ix = mins[0]; ix <= maxs[0]; ++ix)
                {
                    // timer_mt.start(21);
                    const auto voxelI = voxel.find(ix + yzOffset);
                    if (voxelI != voxel.end())
                    {
                        for (const auto& indI : voxelI->second)
                        {
                            if (indI >= surfEdgeStartInd &&
                                indI < surfTriStartInd &&
                                indI - surfEdgeStartInd > eIq)
                            {
                                int seJ = indI - surfEdgeStartInd;
                                const Eigen::Matrix<double, 1, 3>& eJ_v0 =
                                    mesh.vertices.row(mesh.edges[seJ].first);
                                const Eigen::Matrix<double, 1, 3>& eJ_v1 =
                                    mesh.vertices.row(mesh.edges[seJ].second);
                                Eigen::Array<double, 1, 3> bboxEJTopRight =
                                    eJ_v0.array().max(eJ_v1.array());
                                Eigen::Array<double, 1, 3> bboxEJBottomLeft =
                                    eJ_v0.array().min(eJ_v1.array());
                                if (!((bboxEJBottomLeft - rightTop.array() >
                                       0.0)
                                          .any() ||
                                      (leftBottom.array() - bboxEJTopRight >
                                       0.0)
                                          .any()))
                                {
                                    if (ignoredEEs.find(
                                            {eI, indI - surfEdgeStartInd}) ==
                                        ignoredEEs.end())
                                        edgeInds.emplace_back(indI -
                                                              surfEdgeStartInd);
                                }
                            }
                        }
                    }
                }
            }
        }
        std::sort(edgeInds.begin(), edgeInds.end());
        edgeInds.erase(std::unique(edgeInds.begin(), edgeInds.end()),
                       edgeInds.end());
    }
    void build(const cp::TriMesh& mesh, const Eigen::VectorXd& searchDir,
               double& curMaxStepSize, double voxelSize)
    {
        double pSize = 0;

        for (int vI = 0; vI < mesh.vertices.rows(); ++vI)
        {
            pSize += std::abs(searchDir[vI * 3]);
            pSize += std::abs(searchDir[vI * 3 + 1]);
            pSize += std::abs(searchDir[vI * 3 + 2]);
        }
        pSize /= mesh.vertices.rows() * 3;

        const double spanSize = curMaxStepSize * pSize / voxelSize;
        if (spanSize > 1)
        {
            curMaxStepSize /= spanSize;
            // curMaxStepSize reduced for CCD spatial hash efficiency
        }

        const Eigen::MatrixXd& V = mesh.vertices;
        Eigen::MatrixXd Vt(mesh.vertices.rows(), 3);
        for (int vI = 0; vI < mesh.vertices.rows(); ++vI)
        {
            Vt.row(vI) = V.row(vI) +
                         curMaxStepSize *
                             searchDir.template segment<3>(vI * 3).transpose();
        }

        leftBottomCorner =
            V.colwise().minCoeff().array().min(Vt.colwise().minCoeff().array());
        rightTopCorner =
            V.colwise().maxCoeff().array().max(Vt.colwise().maxCoeff().array());
        one_div_voxelSize = 1.0 / voxelSize;
        Eigen::Array<double, 1, 3> range = rightTopCorner - leftBottomCorner;
        voxelCount = (range * one_div_voxelSize).ceil().template cast<int>();
        if (voxelCount.minCoeff() <= 0)
        {
            // cast overflow due to huge search direction
            one_div_voxelSize = 1.0 / (range.maxCoeff() * 1.01);
            voxelCount.setOnes();
        }
        voxelCount0x1 = voxelCount[0] * voxelCount[1];

        surfEdgeStartInd = mesh.vertices.rows();
        surfTriStartInd = surfEdgeStartInd + mesh.edges.size();

        // precompute svVAI
        std::vector<Eigen::Array<int, 1, 3>> svMinVAI(mesh.vertices.rows());
        std::vector<Eigen::Array<int, 1, 3>> svMaxVAI(mesh.vertices.rows());
        tbb::parallel_for(0, (int)mesh.vertices.rows(), 1,
                          [&](int vI)
                          {
                              Eigen::Array<int, 1, 3> v0VAI, vtVAI;
                              locateVoxelAxisIndex(V.row(vI), v0VAI);
                              locateVoxelAxisIndex(Vt.row(vI), vtVAI);
                              svMinVAI[vI] = v0VAI.min(vtVAI);
                              svMaxVAI[vI] = v0VAI.max(vtVAI);
                          });
        voxel.clear();  // TODO: parallel insert
        pointAndEdgeOccupancy.resize(0);
        pointAndEdgeOccupancy.resize(surfTriStartInd);

        // for (int svI = 0; svI < mesh.SVI.size(); ++svI)
        tbb::parallel_for(
            0, (int)mesh.vertices.rows(), 1,
            [&](int vI)
            {
                const Eigen::Array<int, 1, 3>& mins = svMinVAI[vI];
                const Eigen::Array<int, 1, 3>& maxs = svMaxVAI[vI];
                pointAndEdgeOccupancy[vI].reserve((maxs - mins + 1).prod());
                for (int iz = mins[2]; iz <= maxs[2]; ++iz)
                {
                    int zOffset = iz * voxelCount0x1;
                    for (int iy = mins[1]; iy <= maxs[1]; ++iy)
                    {
                        int yzOffset = iy * voxelCount[0] + zOffset;
                        for (int ix = mins[0]; ix <= maxs[0]; ++ix)
                        {
                            pointAndEdgeOccupancy[vI].emplace_back(ix +
                                                                   yzOffset);
                        }
                    }
                }
            });
        tbb::parallel_for(
            0, (int)mesh.edges.size(), 1,
            [&](int seCount)
            {
                int seIInd = seCount + surfEdgeStartInd;
                const auto& seI = mesh.edges[seCount];

                Eigen::Array<int, 1, 3> mins =
                    svMinVAI[seI.first].min(svMinVAI[seI.second]);
                Eigen::Array<int, 1, 3> maxs =
                    svMaxVAI[seI.first].max(svMaxVAI[seI.second]);
                pointAndEdgeOccupancy[seIInd].reserve((maxs - mins + 1).prod());
                for (int iz = mins[2]; iz <= maxs[2]; ++iz)
                {
                    int zOffset = iz * voxelCount0x1;
                    for (int iy = mins[1]; iy <= maxs[1]; ++iy)
                    {
                        int yzOffset = iy * voxelCount[0] + zOffset;
                        for (int ix = mins[0]; ix <= maxs[0]; ++ix)
                        {
                            pointAndEdgeOccupancy[seIInd].emplace_back(
                                ix + yzOffset);
                        }
                    }
                }
            });
        std::vector<std::vector<int>> voxelLoc_sf(mesh.triangles.rows());
        tbb::parallel_for(
            0, (int)mesh.triangles.rows(), 1,
            [&](int sfI)
            {
                Eigen::Array<int, 1, 3> mins =
                    svMinVAI[mesh.triangles(sfI, 0)]
                        .min(svMinVAI[mesh.triangles(sfI, 1)])
                        .min(svMinVAI[mesh.triangles(sfI, 2)]);
                Eigen::Array<int, 1, 3> maxs =
                    svMaxVAI[mesh.triangles(sfI, 0)]
                        .max(svMaxVAI[mesh.triangles(sfI, 1)])
                        .max(svMaxVAI[mesh.triangles(sfI, 2)]);
                for (int iz = mins[2]; iz <= maxs[2]; ++iz)
                {
                    int zOffset = iz * voxelCount0x1;
                    for (int iy = mins[1]; iy <= maxs[1]; ++iy)
                    {
                        int yzOffset = iy * voxelCount[0] + zOffset;
                        for (int ix = mins[0]; ix <= maxs[0]; ++ix)
                        {
                            voxelLoc_sf[sfI].emplace_back(ix + yzOffset);
                        }
                    }
                }
            });

        for (int i = 0; i < pointAndEdgeOccupancy.size(); ++i)
        {
            for (const auto& voxelI : pointAndEdgeOccupancy[i])
            {
                voxel[voxelI].emplace_back(i);
            }
        }
        for (int sfI = 0; sfI < voxelLoc_sf.size(); ++sfI)
        {
            for (const auto& voxelI : voxelLoc_sf[sfI])
            {
                voxel[voxelI].emplace_back(sfI + surfTriStartInd);
            }
        }
    }
    void queryPointForTriangles(int svI,
                                std::unordered_set<int>& sTriInds) const
    {
        sTriInds.clear();
        for (const auto& voxelInd : pointAndEdgeOccupancy[svI])
        {
            const auto& voxelI = voxel.find(voxelInd);
            assert(voxelI != voxel.end());
            for (const auto& indI : voxelI->second)
            {
                if (indI >= surfTriStartInd)
                {
                    if (ignoredPTs.find({svI, indI - surfTriStartInd}) ==
                        ignoredPTs.end())
                        sTriInds.insert(indI - surfTriStartInd);
                }
            }
        }
    }

    // will only put edges with larger than seI index into sEdgeInds
    void queryEdgeForEdges(int seI, std::unordered_set<int>& sEdgeInds) const
    {
        sEdgeInds.clear();
        for (const auto& voxelInd :
             pointAndEdgeOccupancy[seI + surfEdgeStartInd])
        {
            const auto& voxelI = voxel.find(voxelInd);
            assert(voxelI != voxel.end());
            for (const auto& indI : voxelI->second)
            {
                if (indI >= surfEdgeStartInd && indI < surfTriStartInd &&
                    indI - surfEdgeStartInd > seI)
                {
                    if (ignoredEEs.find({seI, indI - surfEdgeStartInd}) ==
                        ignoredEEs.end())
                        sEdgeInds.insert(indI - surfEdgeStartInd);
                }
            }
        }
    }

    void queryEdgeForEdgesWithBBoxCheck(
        const cp::TriMesh& mesh, const Eigen::VectorXd& searchDir,
        double curMaxStepSize, int seI,
        std::unordered_set<int>& sEdgeInds) const
    {
        const Eigen::Matrix<double, 1, 3>& eI_v0 =
            mesh.vertices.row(mesh.edges[seI].first);
        const Eigen::Matrix<double, 1, 3>& eI_v1 =
            mesh.vertices.row(mesh.edges[seI].second);
        Eigen::Matrix<double, 1, 3> eI_v0t =
            eI_v0 + curMaxStepSize *
                        searchDir.template segment<3>(mesh.edges[seI].first * 3)
                            .transpose();
        Eigen::Matrix<double, 1, 3> eI_v1t =
            eI_v1 +
            curMaxStepSize *
                searchDir.template segment<3>(mesh.edges[seI].second * 3)
                    .transpose();
        Eigen::Array<double, 1, 3> bboxEITopRight = eI_v0.array()
                                                        .max(eI_v0t.array())
                                                        .max(eI_v1.array())
                                                        .max(eI_v1t.array());
        Eigen::Array<double, 1, 3> bboxEIBottomLeft = eI_v0.array()
                                                          .min(eI_v0t.array())
                                                          .min(eI_v1.array())
                                                          .min(eI_v1t.array());
        sEdgeInds.clear();
        for (const auto& voxelInd :
             pointAndEdgeOccupancy[seI + surfEdgeStartInd])
        {
            const auto& voxelI = voxel.find(voxelInd);
            assert(voxelI != voxel.end());
            for (const auto& indI : voxelI->second)
            {
                if (indI >= surfEdgeStartInd && indI < surfTriStartInd &&
                    indI - surfEdgeStartInd > seI)
                {
                    int seJ = indI - surfEdgeStartInd;
                    const Eigen::Matrix<double, 1, 3>& eJ_v0 =
                        mesh.vertices.row(mesh.edges[seJ].first);
                    const Eigen::Matrix<double, 1, 3>& eJ_v1 =
                        mesh.vertices.row(mesh.edges[seJ].second);
                    Eigen::Matrix<double, 1, 3> eJ_v0t =
                        eJ_v0 +
                        curMaxStepSize *
                            searchDir
                                .template segment<3>(mesh.edges[seJ].first * 3)
                                .transpose();
                    Eigen::Matrix<double, 1, 3> eJ_v1t =
                        eJ_v1 +
                        curMaxStepSize *
                            searchDir
                                .template segment<3>(mesh.edges[seJ].second * 3)
                                .transpose();
                    Eigen::Array<double, 1, 3> bboxEJTopRight =
                        eJ_v0.array()
                            .max(eJ_v0t.array())
                            .max(eJ_v1.array())
                            .max(eJ_v1t.array());
                    Eigen::Array<double, 1, 3> bboxEJBottomLeft =
                        eJ_v0.array()
                            .min(eJ_v0t.array())
                            .min(eJ_v1.array())
                            .min(eJ_v1t.array());
                    if (!((bboxEJBottomLeft - bboxEITopRight > 0.0).any() ||
                          (bboxEIBottomLeft - bboxEJTopRight > 0.0).any()))
                    {
                        if (ignoredEEs.find({seI, indI - surfEdgeStartInd}) ==
                            ignoredEEs.end())
                            sEdgeInds.insert(indI - surfEdgeStartInd);
                    }
                }
            }
        }
    }

public:  // helper functions
    int locateVoxelIndex(const Eigen::Matrix<double, 1, 3>& pos) const
    {
        Eigen::Array<int, 1, 3> voxelAxisIndex;
        locateVoxelAxisIndex(pos, voxelAxisIndex);
        return voxelAxisIndex2VoxelIndex(voxelAxisIndex.data());
    }
    void locateVoxelAxisIndex(const Eigen::Matrix<double, 1, 3>& pos,
                              Eigen::Array<int, 1, 3>& voxelAxisIndex) const
    {
        voxelAxisIndex = ((pos - leftBottomCorner) * one_div_voxelSize)
                             .array()
                             .floor()
                             .template cast<int>();
    }
    int voxelAxisIndex2VoxelIndex(const int voxelAxisIndex[3]) const
    {
        return voxelAxisIndex2VoxelIndex(voxelAxisIndex[0], voxelAxisIndex[1],
                                         voxelAxisIndex[2]);
    }
    int voxelAxisIndex2VoxelIndex(int ix, int iy, int iz) const
    {
        assert(ix >= 0 && iy >= 0 && iz >= 0 && ix < voxelCount[0] &&
               iy < voxelCount[1] && iz < voxelCount[2]);
        return ix + iy * voxelCount[0] + iz * voxelCount0x1;
    }
};  // class SpatialHash
}  // namespace IPC
#endif  // SpatialHash_hpp