#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/rapidjson.h>

#include <clothpose/meshio/ReadGeometry.hpp>
#include <clothpose/optimizer/Optimizer.hpp>
#include <filesystem>
#include <string>

static std::vector<std::size_t> readJsonInds(std::string filePath)
{
    std::vector<std::size_t> results;
    rapidjson::Document doc;
    FILE* fp =
        fopen(filePath.c_str(), "r");  // non-Windows use "r", windows use "rb"
    char readBuffer[65536];
    rapidjson::FileReadStream ifs(fp, readBuffer, sizeof(readBuffer));
    doc.ParseStream(ifs);
    std::fclose(fp);
    assert(doc.IsArray());
    for (int i = 0; i < doc.Size(); i++)
        results.push_back(static_cast<std::size_t>(doc[i].GetInt()));
    return results;
}

int main()
{
    auto restMesh = cp::readObj("./resources/init/rest_repaired.obj");
    auto coarseMesh = cp::readObj("./resources/init/ClothMesh_156.obj");
    auto currentMesh = std::make_shared<cp::TriMesh>(*restMesh);
    auto initPCD = cp::readPly("./resources/init/156.ply", cp::wrapt<cp::PointsObject>{});
    auto anchorInds = readJsonInds("./resources/init/rest_verts_mapping.json");
    std::vector<cp::Energy> energies{cp::L2Energy{anchorInds, 0.25}};
    cp::Optimizer optimizer{*restMesh, currentMesh, 3e-3,     1e4,
                        1.0,       1.,          energies, 2e-2};
    optimizer.setCoarseMesh(coarseMesh);
    optimizer.solve(128);
    std::vector<cp::Energy> postEnergies{cp::MeshToPointsCDEnergy{2.0},
                                     cp::PointsToMeshCDEnergy{2.0}};
    cp::Optimizer postOptimizer{*restMesh, currentMesh, 3e-3,         1e5,
                            10.0,      1.,          postEnergies, 2e-2};
    postOptimizer.setPCD(initPCD);
    postOptimizer.solve(128);
    std::filesystem::create_directories("./output");
    currentMesh->writeObj("./output/fit_init.obj");

    return 0;
}