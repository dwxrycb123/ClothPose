#pragma once
#include <Eigen/Core>
#include <clothpose/meshio/TriMesh.hpp>
#include <clothpose/meta.hpp>
#include <memory>
#include <string>
#include <type_traits>

namespace cp
{
MeshObject readObj(std::string path);
MeshObject readPly(std::string path, wrapt<MeshObject>);
PointsObject readPly(std::string path, wrapt<PointsObject>);
void showMesh(MeshObject mesh);
void showPoints(PointsObject points, Eigen::RowVector3d color = {0., 0., 0.});
}  // namespace cp