# jax_spatial.py
"""Outline of scipy.spatial-like API implemented in JAX (stubs only)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from jax.numpy import ndarray as Array

Number = Union[float, int]
ArrayLike = Array  # for clarity in signatures; adapt if you accept other types


# --- Errors / Warnings -------------------------------------------------------
class QhullError(RuntimeError):
    """Raised when Qhull-like routines encounter an error/degeneracy."""
    pass


# --- Simple types ------------------------------------------------------------
@dataclass
class Rectangle:
    """Hyperrectangle represented by mins and maxes.

    Attributes:
        maxes: shape (n_dim,)
        mins: shape (n_dim,)
    """
    maxes: Array
    mins: Array

    def volume(self) -> Number:
        """Return hyperrectangle volume."""
        raise NotImplementedError


# --- KD-tree and variants ----------------------------------------------------
class KDTree:
    """KD-tree for nearest-neighbor lookup using JAX arrays."""

    def __init__(self, data: ArrayLike, leafsize: int = 10, compact_nodes: bool = True):
        """
        Parameters
        ----------
        data : (n_points, n_dims) array
        leafsize : int
            Maximum number of points in leaf nodes.
        compact_nodes : bool
            Whether to compact node storage.
        """
        raise NotImplementedError

    @property
    def data(self) -> Array:
        """Original data array (n_points, n_dims)."""
        raise NotImplementedError

    def query(self, x: ArrayLike, k: int = 1, return_distance: bool = True) -> Union[Tuple[Array, Array], Array]:
        """Query k nearest neighbors for point(s) x.

        Returns either distances and indices or only indices depending on return_distance.
        """
        raise NotImplementedError

    def query_ball_point(self, x: ArrayLike, r: float) -> List[List[int]]:
        """Return list of neighbors within radius r for each point in x."""
        raise NotImplementedError

    def query_pairs(self, r: float) -> Set[Tuple[int, int]]:
        """Return set of pairs of points within distance r of each other."""
        raise NotImplementedError

    def count_neighbors(self, other: "KDTree", r: float) -> int:
        """Count number of pairs within distance r between this tree and another."""
        raise NotImplementedError


class cKDTree(KDTree):
    """C-optimized KDTree API compatibility stub (implemented in JAX)."""

    def __init__(self, data: ArrayLike, leafsize: int = 10, compact_nodes: bool = True):
        super().__init__(data, leafsize=leafsize, compact_nodes=compact_nodes)
        raise NotImplementedError


# --- Triangulation / Hulls / Voronoi ---------------------------------------
class Delaunay:
    """Delaunay triangulation in N dimensions."""

    def __init__(self, points: ArrayLike, furthest_site: bool = False, qhull_options: Optional[str] = None):
        """
        Parameters:
            points: (n_points, n_dims) array
        """
        raise NotImplementedError

    @property
    def points(self) -> Array:
        raise NotImplementedError

    @property
    def simplices(self) -> Array:
        """Indices of points forming simplices (m, ndim+1)."""
        raise NotImplementedError

    @property
    def neighbors(self) -> Array:
        """Neighboring simplex indices (-1 for no neighbor)."""
        raise NotImplementedError

    def find_simplex(self, x: ArrayLike) -> Array:
        """Return index of simplex containing each point in x, or -1."""
        raise NotImplementedError

    def barycentric_coordinates(self, x: ArrayLike) -> Array:
        """Return barycentric coordinates of x with respect to containing simplices."""
        raise NotImplementedError


class ConvexHull:
    """Convex hull in N dimensions."""

    def __init__(self, points: ArrayLike, incremental: bool = False, qhull_options: Optional[str] = None):
        raise NotImplementedError

    @property
    def points(self) -> Array:
        raise NotImplementedError

    @property
    def simplices(self) -> Array:
        """Indices of points forming hull facets (m, ndim)."""
        raise NotImplementedError

    @property
    def equations(self) -> Array:
        """Plane equations for facets (m, ndim+1): normal and offset."""
        raise NotImplementedError


class Voronoi:
    """Voronoi tessellation in N dimensions."""

    def __init__(self, points: ArrayLike, furthest_site: bool = False, qhull_options: Optional[str] = None):
        raise NotImplementedError

    @property
    def points(self) -> Array:
        raise NotImplementedError

    @property
    def vertices(self) -> Array:
        raise NotImplementedError

    @property
    def ridge_vertices(self) -> Array:
        """Vertices indices forming each ridge."""
        raise NotImplementedError

    @property
    def ridge_points(self) -> Array:
        """Pair of input point indices for each ridge."""
        raise NotImplementedError


class SphericalVoronoi:
    """Voronoi diagram on the surface of a sphere."""

    def __init__(self, points: ArrayLike, radius: Optional[float] = None, center: Optional[ArrayLike] = None):
        raise NotImplementedError

    @property
    def points(self) -> Array:
        raise NotImplementedError

    @property
    def regions(self) -> List[List[int]]:
        """Indices of vertices for each Voronoi region on the sphere."""
        raise NotImplementedError


class HalfspaceIntersection:
    """Intersection of halfspaces in N dimensions."""

    def __init__(self, halfspaces: ArrayLike, interior_point: ArrayLike):
        """
        halfspaces: (m, n+1) array of plane equations [a, b, c, ..., d] representing a.x + d <= 0
        interior_point: a point strictly inside the intersection
        """
        raise NotImplementedError

    @property
    def vertices(self) -> Array:
        raise NotImplementedError


# --- Plot helpers (2-D) -----------------------------------------------------
def delaunay_plot_2d(tri: Delaunay, ax: Any = None) -> Any:
    """Plot Delaunay triangulation in 2D. Returns the matplotlib Axes used."""
    raise NotImplementedError


def convex_hull_plot_2d(hull: ConvexHull, ax: Any = None) -> Any:
    """Plot convex hull in 2D."""
    raise NotImplementedError


def voronoi_plot_2d(vor: Voronoi, ax: Any = None) -> Any:
    """Plot Voronoi diagram in 2D."""
    raise NotImplementedError


# --- Utility functions ------------------------------------------------------
def tsearch(tri: Delaunay, xi: ArrayLike) -> Array:
    """Find indices of simplices that contain the points xi.

    Returns:
        simplex_indices: (m,) array of simplex index or -1 if not found
    """
    raise NotImplementedError


def distance_matrix(x: ArrayLike, y: ArrayLike, p: Number = 2.0, threshold: Optional[float] = None) -> Array:
    """Compute distance matrix between rows of x and rows of y.

    Parameters
    ----------
    x : (n, d)
    y : (m, d)
    p : float or int
        Minkowski p-norm.
    threshold : Optional[float]
        If provided, optionally cut off or accelerate computation with a threshold.
    """
    raise NotImplementedError


def minkowski_distance(x: ArrayLike, y: ArrayLike, p: Number = 2.0) -> Array:
    """Compute L**p distance between two arrays of points row-wise.

    Returns shape (n,).
    """
    raise NotImplementedError


def minkowski_distance_p(x: ArrayLike, y: ArrayLike, p: Number = 2.0) -> Array:
    """Compute p-th power of L**p distance (i.e., sum |x-y|**p) row-wise."""
    raise NotImplementedError


def procrustes(data1: ArrayLike, data2: ArrayLike) -> Tuple[Array, Array, dict]:
    """Procrustes analysis: align data2 to data1 by scaling/rotation/translation.

    Returns:
        mtx1, mtx2, disparity_info
    """
    raise NotImplementedError


def geometric_slerp(start: ArrayLike, end: ArrayLike, t: ArrayLike, tol: float = 1e-8) -> Array:
    """Spherical linear interpolation between start and end points on sphere.

    Parameters
    ----------
    start, end : (..., d) arrays (unit vectors)
    t : scalar or array of interpolation parameters in [0, 1]
    """
    raise NotImplementedError


# --- Additional helpers often present in spatial.distance -------------------
def cdist(XA: ArrayLike, XB: ArrayLike, metric: str = "euclidean", **kwargs) -> Array:
    """Pairwise distances between two collections of observations."""
    raise NotImplementedError


def pdist(X: ArrayLike, metric: str = "euclidean", **kwargs) -> Array:
    """Pairwise distances within a single collection (condensed form)."""
    raise NotImplementedError


# --- End of file -------------------------------------------------------------
