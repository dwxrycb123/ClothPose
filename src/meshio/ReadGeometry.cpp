#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>

#include <clothpose/meshio/ReadGeometry.hpp>
#include <iostream>
#include <memory>
namespace cp
{
MeshObject readObj(std::string path)
{
    Eigen::MatrixXd verts;
    Eigen::MatrixXi tris;
    igl::readOBJ(path, verts, tris);
    return std::make_shared<TriMesh>(std::move(verts), std::move(tris));
}

MeshObject readPly(std::string path, cp::wrapt<MeshObject>)
{
    Eigen::MatrixXd verts;
    Eigen::MatrixXi tris;
    igl::readPLY(path, verts, tris);
    return std::make_shared<TriMesh>(std::move(verts), std::move(tris));
}

PointsObject readPly(std::string path, cp::wrapt<PointsObject>)
{
    Eigen::MatrixXd verts;
    Eigen::MatrixXi tris;
    igl::readPLY(path, verts, tris);
    return std::make_shared<Points>(verts);
}

void showMesh(MeshObject mesh)
{
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(mesh->vertices, mesh->triangles);
    viewer.launch();
}

void showPoints(PointsObject points, Eigen::RowVector3d color)
{
    igl::opengl::glfw::Viewer viewer;
    viewer.data().add_points(*points, color);
    viewer.launch();
}
}  // namespace cp