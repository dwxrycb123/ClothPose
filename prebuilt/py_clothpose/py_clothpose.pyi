"""
Python binding for ClothPose Optimizer Library.
"""
from __future__ import annotations
import numpy
import typing
__all__ = ['ARAPEnergy', 'BarrierEnergy', 'L2Energy', 'Mesh',
           'Mesh2PointEnergy', 'Optimizer', 'Point2MeshEnergy', 'Points',
           'show_mesh', 'show_points']


class ARAPEnergy:
    def __init__(self, rest_mesh: Mesh, weight: float, boundary_weight: float) -> None:
        ...


class BarrierEnergy:
    def __init__(self, dHat: float, weight: float) -> None:
        ...


class L2Energy:
    def __init__(
            self, anchor_inds: numpy.ndarray[numpy.int32],
            weight: float) -> None:
        ...


class Mesh:
    @typing.overload
    def __init__(self) -> None:
        ...

    @typing.overload
    def __init__(
            self, vertices: numpy.ndarray[numpy.float64],
            triangles: numpy.ndarray[numpy.int32]) -> None:
        ...

    def clone(self) -> Mesh:
        """
        clone this Mesh object
        """
    @property
    def triangles(self) -> numpy.ndarray[numpy.int32]:
        """
        Mesh triangles
        """
    @property
    def vertices(self) -> numpy.ndarray[numpy.float64]:
        """
        Mesh vertices
        """

    def write_obj(self, path: str) -> None:
        """
        write this Mesh object in obj format
        """


class Mesh2PointEnergy:
    def __init__(self, weight: float) -> None:
        ...


class Optimizer:
    def __init__(
            self, rest_mesh: Mesh, current_mesh: Mesh, rel_dHat: float,
            barrier_weight: float, ARAP_weight: float,
            ARAP_boundary_weight: float,
            energies:
            list
            [ARAPEnergy | L2Energy | Mesh2PointEnergy | Point2MeshEnergy |
             BarrierEnergy],
            rel_newton_res: float) -> None:
        ...

    def set_coarse_mesh(self, mesh: Mesh) -> None:
        ...

    def set_pcd(self, pcd: Points) -> None:
        ...

    def solve(self, max_iters: int, est_rot_per_iters: int) -> None:
        ...


class Point2MeshEnergy:
    def __init__(self, weight: float) -> None:
        ...


class Points:
    def __init__(self, points: numpy.ndarray[numpy.float64]) -> None:
        ...


def show_mesh(mesh: Mesh) -> None:
    """
    a function that shows a Mesh object
    """


def show_points(points: numpy.ndarray[numpy.float64]) -> None:
    """
    a function that shows a Mesh object
    """
