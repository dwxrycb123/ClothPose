#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <clothpose/meshio/ReadGeometry.hpp>
#include <clothpose/meshio/TriMesh.hpp>
#include <clothpose/optimizer/Energy.hpp>
#include <clothpose/optimizer/Optimizer.hpp>

namespace py = pybind11;

// references: https://github.com/pybind/pybind11/issues/4108
template <typename T, typename... Args>
struct variant_prepender;

template <typename... Args0, typename... Args1>
struct variant_prepender<std::variant<Args0...>, Args1...>
{
    using type = std::variant<Args1..., Args0...>;
};

template <typename Variant>
struct monostated : public variant_prepender<Variant, std::monostate>
{
};

template <class... Ts>
struct overloaded : Ts...
{
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// Implementation of variant_cast_no_monostate()
// inspired by https://stackoverflow.com/a/47204507
template <class... Args>
struct variant_cast_no_monostate_impl
{
    std::variant<Args...> v;

    template <class... ToArgs>
    operator std::variant<ToArgs...>() const
    {
        return std::visit(
            overloaded{
                [](const std::monostate&) -> std::variant<ToArgs...>
                {
                    throw std::runtime_error(
                        "variant_cast_no_monostate received a value of type "
                        "std::monostate, cannot cast it");
                },
                [](auto&& arg) -> std::variant<ToArgs...> { return arg; },
            },
            v);
    }
};

template <class... Args>
auto variant_cast_no_monostate(const std::variant<Args...>& v)
    -> variant_cast_no_monostate_impl<Args...>
{
    return {v};
}

PYBIND11_MODULE(py_clothpose, m)
{
    m.doc() = "Python binding for ClothPose Optimizer Library.";

    m.def("show_mesh", &cp::showMesh, "a function that shows a Mesh object",
          py::arg("mesh"));
    m.def(
        "show_points", [](cp::PointsObject points) { cp::showPoints(points); },
        "a function that shows a Mesh object", py::arg("points"));
    py::class_<cp::Points, std::shared_ptr<cp::Points>>(m, "Points")
        .def(
            py::init<>(
                [](py::array_t<double> points)
                {
                    auto pointsBuffer = points.request();
                    if (pointsBuffer.ndim != 2)
                        throw std::runtime_error(
                            "Number of point clouds points dimension must be "
                            "2!");
                    if (pointsBuffer.shape[1] != 3)
                        throw std::runtime_error(
                            "The 2nd dimenstion of point clouds points must be "
                            "3!");
                    auto pointsBufferPtr =
                        static_cast<double*>(pointsBuffer.ptr);
                    auto nPoints = pointsBuffer.shape[0];
                    Eigen::MatrixXd pcd(nPoints, 3);
                    for (std::size_t pi = 0; pi < nPoints; pi++)
                    {
                        pcd.row(pi) << pointsBufferPtr[pi * 3],
                            pointsBufferPtr[pi * 3 + 1],
                            pointsBufferPtr[pi * 3 + 2];
                    }
                    return std::make_shared<cp::Points>(std::move(pcd));
                }),
            py::arg("points"));
    py::class_<cp::TriMesh, std::shared_ptr<cp::TriMesh>>(m, "Mesh")
        .def(py::init<>())
        .def(
            py::init<>(
                [](py::array_t<double> vertsArr, py::array_t<int> trisArr)
                {
                    auto vertsBuffer = vertsArr.request();
                    auto trisBuffer = trisArr.request();
                    if (vertsBuffer.ndim != 2 || trisBuffer.ndim != 2)
                        throw std::runtime_error(
                            "Number of mesh verts and tris dimension must be "
                            "2!");
                    if (vertsBuffer.shape[1] != 3 || trisBuffer.shape[1] != 3)
                        throw std::runtime_error(
                            "The 2nd dimenstion of mesh verts and tris must be "
                            "3!");
                    auto vertsBufferPtr = static_cast<double*>(vertsBuffer.ptr);
                    auto trisBufferPtr = static_cast<int*>(trisBuffer.ptr);
                    auto nVerts = vertsBuffer.shape[0];
                    auto nTris = trisBuffer.shape[0];
                    Eigen::MatrixXd verts(nVerts, 3);
                    Eigen::MatrixXi tris(nTris, 3);
                    for (std::size_t vi = 0; vi < nVerts; vi++)
                    {
                        verts.row(vi) << vertsBufferPtr[vi * 3],
                            vertsBufferPtr[vi * 3 + 1],
                            vertsBufferPtr[vi * 3 + 2];
                    }
                    for (std::size_t ti = 0; ti < nTris; ti++)
                    {
                        tris.row(ti) << trisBufferPtr[ti * 3],
                            trisBufferPtr[ti * 3 + 1],
                            trisBufferPtr[ti * 3 + 2];
                    }
                    return cp::TriMesh{std::move(verts), std::move(tris)};
                }),
            py::arg("vertices"), py::arg("triangles"))
        .def("clone", &cp::TriMesh::clone, "clone this Mesh object")
        .def_property_readonly(
            "vertices",
            [](cp::TriMesh& self)
            {
                std::vector<std::size_t> stride{sizeof(double) * 3,
                                                sizeof(double)},
                    shape{self.nVerts(), 3};
                auto data = new double[self.nVerts() * 3];
                for (std::size_t vi = 0; vi != self.nVerts(); vi++)
                    for (std::size_t d = 0; d != 3; d++)
                        data[vi * 3 + d] = self.vertices(vi, d);
                py::capsule capsule{data, [](void* data)
                                    { delete[] static_cast<double*>(data); }};
                return py::array_t<double>{shape, stride, data, capsule};
            },
            "Mesh vertices")
        .def_property_readonly(
            "triangles",
            [](cp::TriMesh& self)
            {
                std::vector<std::size_t> stride{sizeof(int) * 3, sizeof(int)},
                    shape{self.nTris(), 3};
                auto data = new int[self.nTris() * 3];
                for (std::size_t ti = 0; ti != self.nTris(); ti++)
                    for (std::size_t d = 0; d != 3; d++)
                        data[ti * 3 + d] = self.triangles(ti, d);
                py::capsule capsule{
                    data, [](void* data) { delete[] static_cast<int*>(data); }};
                return py::array_t<int>{shape, stride, data, capsule};
            },
            "Mesh triangles")
        .def("write_obj", &cp::TriMesh::writeObj,
             "write this Mesh object in obj format", py::arg("path"));

    py::class_<cp::ARAPEnergy>(m, "ARAPEnergy")
        .def(py::init<cp::TriMesh, double, double>(), py::arg("rest_mesh"),
             py::arg("weight"), py::arg("boundary_weight"));
    py::class_<cp::L2Energy>(m, "L2Energy")
        .def(py::init<>(
                 [](py::array_t<int> indsArr, double weight)
                 {
                     auto indsBuffer = indsArr.request();
                     if (indsBuffer.ndim != 1)
                         throw std::runtime_error(
                             "Number of dimensions must be one!");
                     auto indsBufferPtr = static_cast<int*>(indsBuffer.ptr);
                     std::vector<std::size_t> inds(indsArr.size());
                     for (std::size_t i = 0; i != inds.size(); i++)
                         inds[i] = static_cast<std::size_t>(indsBufferPtr[i]);
                     return cp::L2Energy{inds, weight};
                 }),
             py::arg("anchor_inds"), py::arg("weight"));
    py::class_<cp::BarrierEnergy>(m, "BarrierEnergy")
        .def(py::init<double, double>(), py::arg("dHat"), py::arg("weight"));
    py::class_<cp::MeshToPointsCDEnergy>(m, "Mesh2PointEnergy")
        .def(py::init<double>(), py::arg("weight"));
    py::class_<cp::PointsToMeshCDEnergy>(m, "Point2MeshEnergy")
        .def(py::init<double>(), py::arg("weight"));

    py::class_<cp::Optimizer>(m, "Optimizer")
        .def(py::init<>(
                 [](const cp::TriMesh& restMesh, cp::MeshObject meshObject,
                    double relDHat = 3e-3, double barrierWeight = 1e5,
                    double ARAPWeight = 10.0, double ARAPBoundaryWeight = 1.0,
                    std::vector<monostated<cp::Energy>::type> monoEnergyTerms =
                        {},
                    double relNewtonDirRes = 1e-2)
                 {
                     std::vector<cp::Energy> energyTerms;
                     for (auto e : monoEnergyTerms)
                         energyTerms.push_back(variant_cast_no_monostate(e));

                     return cp::Optimizer{restMesh,    meshObject,
                                          relDHat,     barrierWeight,
                                          ARAPWeight,  ARAPBoundaryWeight,
                                          energyTerms, relNewtonDirRes};
                 }),
             py::arg("rest_mesh"), py::arg("current_mesh"), py::arg("rel_dHat"),
             py::arg("barrier_weight"), py::arg("ARAP_weight"),
             py::arg("ARAP_boundary_weight"), py::arg("energies"),
             py::arg("rel_newton_res"))
        .def("set_coarse_mesh", &cp::Optimizer::setCoarseMesh, py::arg("mesh"))
        .def("set_pcd", &cp::Optimizer::setPCD, py::arg("pcd"))
        .def("solve", &cp::Optimizer::solve, py::arg("max_iters"),
             py::arg("est_rot_per_iters"))
        .def("setup_ignored_collision_pairs",
             &cp::Optimizer::setupIgnoredCollisionPairs)
        .def("clear_ignored_collision_pairs",
             &cp::Optimizer::clearIgnoredCollisionPairs);
}
