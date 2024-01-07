#include <clothpose/IPC/LinSysSolver/EigenLibSolver.hpp>
#include <clothpose/IPC/LinSysSolver/LinSysSolver.hpp>
namespace IPC
{
template <typename vectorTypeI, typename vectorTypeS>
LinSysSolver<vectorTypeI, vectorTypeS>*
LinSysSolver<vectorTypeI, vectorTypeS>::create()
{
    return new EigenLibSolver<vectorTypeI, vectorTypeS>();
}

template class LinSysSolver<Eigen::VectorXi, Eigen::VectorXd>;
}