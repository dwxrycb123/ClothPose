#include <igl/writeOBJ.h>

#include <clothpose/meshio/TriMesh.hpp>
#include <cmath>
#include <iostream>
#include <set>
#include <utility>
namespace cp
{
TriMesh::TriMesh(const Eigen::MatrixXd &vertices,
                 const Eigen::MatrixXi &triangles, double cot_weight_thresh)
    : vertices{vertices},
      triangles{triangles},
      _cot_weight_thresh{cot_weight_thresh}
{
    init(cot_weight_thresh);
}

TriMesh::TriMesh(Eigen::MatrixXd &&vertices, Eigen::MatrixXi &&triangles,
                 double cot_weight_thresh)
    : vertices{vertices},
      triangles{triangles},
      _cot_weight_thresh{cot_weight_thresh}
{
    init(cot_weight_thresh);
}

TriMesh TriMesh::clone() { return *this; }

void TriMesh::init(double cot_weight_thresh)
{
    std::set<std::pair<int, int>> edgeSet, nonBoundaryEdgeSet;

    // build the edges
    for (std::size_t idx = 0; idx < triangles.rows(); idx++)
    {
        auto tri = triangles.row(idx);
        int t0 = tri[0];
        int t1 = tri[1];
        int t2 = tri[2];

        auto e01 = std::make_pair(t0, t1);
        auto e10 = std::make_pair(t1, t0);
        auto e12 = std::make_pair(t1, t2);
        auto e21 = std::make_pair(t2, t1);
        auto e20 = std::make_pair(t2, t0);
        auto e02 = std::make_pair(t0, t2);

        if (edgeSet.find(e01) == edgeSet.end() &&
            edgeSet.find(e10) == edgeSet.end())
            edgeSet.insert(e01);
        else
            nonBoundaryEdgeSet.insert(e01);
        if (edgeSet.find(e12) == edgeSet.end() &&
            edgeSet.find(e21) == edgeSet.end())
            edgeSet.insert(e12);
        else
            nonBoundaryEdgeSet.insert(e12);
        if (edgeSet.find(e20) == edgeSet.end() &&
            edgeSet.find(e02) == edgeSet.end())
            edgeSet.insert(e20);
        else
            nonBoundaryEdgeSet.insert(e20);
    }

    // build the neighbours & edges
    edges.clear();
    neighbours.resize(nVerts(), {});
    for (auto &e : edgeSet)
    {
        neighbours[e.first].push_back(e.second);
        neighbours[e.second].push_back(e.first);
        edges.push_back(e);
        if (nonBoundaryEdgeSet.find(e) == nonBoundaryEdgeSet.end())
        {
            boundaryEdges.insert(e);
            boundaryVertices.insert(e.first);
            boundaryVertices.insert(e.second);
        }
    }

    // these functions are from IPC
    computeVNeighbor();
    computeAvgEdgeLen();

    // build the weights
    weights = Eigen::MatrixXd::Zero(vertices.size(), vertices.size());
    for (std::size_t triInd = 0; triInd < triangles.rows(); triInd++)
    {
        auto tri = triangles.row(triInd);
        int t0 = tri[0];
        int t1 = tri[1];
        int t2 = tri[2];
        auto v0 = vertices.row(t0);
        auto v1 = vertices.row(t1);
        auto v2 = vertices.row(t2);

        auto e01 = v1 - v0;
        auto e12 = v2 - v1;
        auto e20 = v0 - v2;

        double cos0 = -e20.dot(e01) / (e20.norm() * e01.norm());
        double cos1 = -e01.dot(e12) / (e01.norm() * e12.norm());
        double cos2 = -e12.dot(e20) / (e12.norm() * e20.norm());

        double ang0 = acos(cos0);
        double ang1 = acos(cos1);
        double ang2 = acos(cos2);

        double w0 = cos0 / sin(ang0);
        double w1 = cos1 / sin(ang1);
        double w2 = cos2 / sin(ang2);

        weights(t0, t1) += w2;
        weights(t1, t0) += w2;
        weights(t1, t2) += w0;
        weights(t2, t1) += w0;
        weights(t2, t0) += w1;
        weights(t0, t2) += w1;
    }

    // traverse all vertices and their neighbours to thresh the weights
    for (std::size_t i = 0; i < vertices.rows(); i++)
        for (auto j : neighbours[i])
            if (weights(i, j) < cot_weight_thresh)
                weights(i, j) = cot_weight_thresh;
}

void TriMesh::writeObj(std::string path) const
{
    igl::writeOBJ(path, vertices, triangles);
}

std::size_t TriMesh::nVerts() const { return vertices.rows(); }

std::size_t TriMesh::nTris() const { return triangles.rows(); }

double TriMesh::matSpaceBBoxSize2() const
{
    Eigen::Array<double, 1, 3> bottomLeft, topRight;
    bottomLeft = std::numeric_limits<double>::infinity();
    topRight = -std::numeric_limits<double>::infinity();
    bottomLeft = bottomLeft.min(vertices.colwise().minCoeff().array());
    topRight = topRight.max(vertices.colwise().maxCoeff().array());

    if ((bottomLeft - topRight > 0.0).any())
    {
        return 0.0;
    }
    else
    {
        return (topRight - bottomLeft).matrix().squaredNorm();
    }
}

void TriMesh::computeVNeighbor()
{
    vNeighbor.resize(0);
    vNeighbor.resize(vertices.rows());
    for (int sfI = 0; sfI < triangles.rows(); ++sfI)
    {
        const Eigen::Matrix<int, 1, 3> &sfVInd = triangles.row(sfI);
        for (int vI = 0; vI < 3; vI++)
        {
            for (int vJ = vI + 1; vJ < 3; vJ++)
            {
                vNeighbor[sfVInd[vI]].insert(sfVInd[vJ]);
                vNeighbor[sfVInd[vJ]].insert(sfVInd[vI]);
            }
        }
    }
}

void TriMesh::setAnchorPointsAs(const std::vector<int> &anchorIndices,
                                const TriMesh &deformed)
{
    for (int vI : anchorIndices)
    {
        vertices.row(vI) = deformed.vertices.row(vI);
    }
    computeAvgEdgeLen();
}

void TriMesh::setAnchorPointsAs(
    const std::vector<int> &anchorIndices,
    const std::vector<Eigen::Vector3d> &anchorPositions)
{
    for (uint idx = 0; idx < anchorIndices.size(); idx++)
    {
        vertices.row(anchorIndices[idx]) = anchorPositions[idx];
    }
    computeAvgEdgeLen();
}

void TriMesh::computeAvgEdgeLen()
{
    _avgEdgeLen = 0;
    for (auto e : edges)
        _avgEdgeLen += (vertices.row(e.second) - vertices.row(e.first)).norm();
    _avgEdgeLen /= double(edges.size());
}

double TriMesh::avgEdgeLen() const { return _avgEdgeLen; }

double TriMesh::cot_weight_thresh() const { return _cot_weight_thresh; }

std::ostream &operator<<(std::ostream &os, const TriMesh &mesh)
{
    os << "A TriMesh with " << mesh.vertices.rows() << " vertices, "
       << mesh.triangles.rows() << " triangles. ";
    return os;
}
}  // namespace cp