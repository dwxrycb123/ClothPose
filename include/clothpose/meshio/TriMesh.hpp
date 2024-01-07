#pragma once
#include <Eigen/Eigen>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cp
{
class TriMesh
{
public:
    Eigen::MatrixXi triangles;
    Eigen::MatrixXd vertices;
    Eigen::MatrixXd weights;

    std::vector<std::vector<int>> neighbours;
    std::vector<std::set<int>> vNeighbor;
    std::vector<std::pair<int, int>> edges;
    std::set<std::pair<int, int>> boundaryEdges;
    std::set<int> boundaryVertices;

    TriMesh() = default;
    TriMesh(const TriMesh &) = default;
    TriMesh &operator=(const TriMesh &) = default;
    TriMesh(const Eigen::MatrixXd &verts, const Eigen::MatrixXi &triangles,
            double cot_weight_thresh = 1e-3);
    TriMesh(Eigen::MatrixXd &&verts, Eigen::MatrixXi &&triangles,
            double cot_weight_thresh = 1e-3);

    TriMesh clone();
    void writeObj(std::string path) const;
    std::size_t nVerts() const;
    std::size_t nTris() const;
    double avgEdgeLen() const;
    double cot_weight_thresh() const;
    double matSpaceBBoxSize2() const;

    void setAnchorPointsAs(const std::vector<int> &anchorIndices,
                           const TriMesh &deformed);
    void setAnchorPointsAs(const std::vector<int> &anchorIndices,
                           const std::vector<Eigen::Vector3d> &anchorPositions);

    friend std::ostream &operator<<(std::ostream &os, const TriMesh &mesh);

private:
    double _avgEdgeLen;
    double _cot_weight_thresh;
    void computeAvgEdgeLen();
    void computeVNeighbor();
    void init(double cot_weight_thresh = 1e-3);
};

using MeshObject = std::shared_ptr<TriMesh>;
using Points = Eigen::MatrixXd;
using PointsObject = std::shared_ptr<Points>;
}  // namespace cp