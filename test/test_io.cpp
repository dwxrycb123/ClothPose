#include <clothpose/meshio/ReadGeometry.hpp>
#include <iostream>
#include <igl/opengl/glfw/Viewer.h>

int main()
{
    auto restMesh = cp::readObj("./resources/init/rest_repaired.obj");
    cp::showMesh(restMesh);

    auto restPoints = cp::readPly("./resources/pcd/156.ply", cp::wrapt<cp::PointsObject>{});
    cp::showPoints(restPoints);

    igl::opengl::glfw::Viewer viewer; 
    viewer.data().set_mesh(restMesh->vertices, restMesh->triangles); 
    viewer.data().set_points(*restPoints, Eigen::RowVector3d::Zero()); 
    viewer.launch(); 

    return 0;
}