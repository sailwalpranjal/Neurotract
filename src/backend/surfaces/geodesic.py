"""
Geodesic Distance Computation on Cortical Surfaces

Implements the Heat Method for fast geodesic distance computation on triangular meshes.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging
from scipy import sparse
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


class GeodesicDistance:
    """
    Compute geodesic distances on triangular surface meshes using the Heat Method
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Initialize geodesic calculator

        Args:
            vertices: Vertex coordinates (n_vertices, 3)
            faces: Triangle indices (n_faces, 3)
        """
        self.vertices = vertices
        self.faces = faces
        self.n_vertices = len(vertices)
        self.n_faces = len(faces)

        # Precompute geometric quantities
        self._compute_laplacian()
        self._compute_face_areas()

        logger.info(f"Initialized geodesic calculator: {self.n_vertices} vertices, "
                   f"{self.n_faces} faces")

    def _compute_laplacian(self):
        """Compute cotangent Laplacian matrix"""
        logger.debug("Computing cotangent Laplacian...")

        # Initialize sparse matrix
        I = []
        J = []
        V = []

        # Compute cotangent weights
        for face_idx, face in enumerate(self.faces):
            v0, v1, v2 = face

            # Get vertex positions
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            p2 = self.vertices[v2]

            # Edge vectors
            e0 = p1 - p2  # Opposite to v0
            e1 = p2 - p0  # Opposite to v1
            e2 = p0 - p1  # Opposite to v2

            # Cotangent weights
            cot0 = self._cotangent(e1, e2)
            cot1 = self._cotangent(e2, e0)
            cot2 = self._cotangent(e0, e1)

            # Add entries (symmetric)
            # Edge v1-v2
            I.extend([v1, v2])
            J.extend([v2, v1])
            V.extend([cot0, cot0])

            # Edge v2-v0
            I.extend([v2, v0])
            J.extend([v0, v2])
            V.extend([cot1, cot1])

            # Edge v0-v1
            I.extend([v0, v1])
            J.extend([v1, v0])
            V.extend([cot2, cot2])

        # Build sparse matrix
        L = sparse.coo_matrix((V, (I, J)), shape=(self.n_vertices, self.n_vertices))
        L = L.tocsr()

        # Diagonal entries (negative sum of off-diagonal)
        L.setdiag(-np.array(L.sum(axis=1)).flatten())

        self.laplacian = L
        logger.debug("Laplacian computed")

    def _cotangent(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute cotangent of angle between vectors u and v"""
        cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-10)
        sin_angle = np.linalg.norm(np.cross(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-10)
        cot = cos_angle / (sin_angle + 1e-10)
        return cot

    def _compute_face_areas(self):
        """Compute area of each face"""
        areas = np.zeros(self.n_faces)

        for i, face in enumerate(self.faces):
            v0, v1, v2 = self.vertices[face]
            e1 = v1 - v0
            e2 = v2 - v0
            area = 0.5 * np.linalg.norm(np.cross(e1, e2))
            areas[i] = area

        self.face_areas = areas
        self.total_area = np.sum(areas)

    def compute_distance_from_source(
        self,
        source_vertex: int,
        time_step: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute geodesic distance from a source vertex using Heat Method

        Args:
            source_vertex: Index of source vertex
            time_step: Heat diffusion time (auto-determined if None)

        Returns:
            Distance to all vertices (n_vertices,)
        """
        # Auto-determine time step
        if time_step is None:
            # Use mean edge length squared
            mean_edge_length = self._compute_mean_edge_length()
            time_step = mean_edge_length ** 2
            logger.debug(f"Auto time step: {time_step:.6f}")

        # Step 1: Diffuse heat from source
        u = self._diffuse_heat(source_vertex, time_step)

        # Step 2: Compute gradient of heat
        X = self._compute_gradient(u)

        # Step 3: Normalize gradient
        X_normalized = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)

        # Step 4: Solve Poisson equation for distance
        phi = self._solve_poisson(X_normalized)

        # Normalize so source has distance 0
        phi = phi - phi[source_vertex]

        return np.abs(phi)

    def _diffuse_heat(self, source_idx: int, time_step: float) -> np.ndarray:
        """
        Diffuse heat from source vertex

        Solves: (I - t*L) u = delta_source
        """
        # Build system matrix
        I_matrix = sparse.identity(self.n_vertices)
        A = I_matrix - time_step * self.laplacian

        # Build right-hand side (delta function at source)
        b = np.zeros(self.n_vertices)
        b[source_idx] = 1.0

        # Solve linear system
        u = spsolve(A, b)

        return u

    def _compute_gradient(self, u: np.ndarray) -> np.ndarray:
        """
        Compute gradient of scalar field on faces

        Args:
            u: Scalar field on vertices (n_vertices,)

        Returns:
            Gradient vectors on faces (n_faces, 3)
        """
        X = np.zeros((self.n_faces, 3))

        for i, face in enumerate(self.faces):
            v0, v1, v2 = face

            # Vertex positions
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            p2 = self.vertices[v2]

            # Edge vectors
            e1 = p1 - p0
            e2 = p2 - p0

            # Face normal
            n = np.cross(e1, e2)
            area = 0.5 * np.linalg.norm(n)
            n = n / (np.linalg.norm(n) + 1e-10)

            # Gradient formula
            grad = (u[v0] * np.cross(n, p2 - p1) +
                   u[v1] * np.cross(n, p0 - p2) +
                   u[v2] * np.cross(n, p1 - p0)) / (2 * area + 1e-10)

            X[i] = grad

        return X

    def _solve_poisson(self, X: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation: L*phi = div(X)

        Args:
            X: Vector field on faces (n_faces, 3)

        Returns:
            Distance field (n_vertices,)
        """
        # Compute divergence
        div = self._compute_divergence(X)

        # Solve L * phi = div
        # Pin one vertex to remove kernel
        L_pinned = self.laplacian.tolil()
        L_pinned[0, :] = 0
        L_pinned[0, 0] = 1
        L_pinned = L_pinned.tocsr()

        div_pinned = div.copy()
        div_pinned[0] = 0

        phi = spsolve(L_pinned, div_pinned)

        return phi

    def _compute_divergence(self, X: np.ndarray) -> np.ndarray:
        """
        Compute divergence of vector field on faces

        Args:
            X: Vector field on faces (n_faces, 3)

        Returns:
            Divergence on vertices (n_vertices,)
        """
        div = np.zeros(self.n_vertices)

        for i, face in enumerate(self.faces):
            v0, v1, v2 = face

            # Vertex positions
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            p2 = self.vertices[v2]

            # Edge vectors
            e0 = p1 - p2
            e1 = p2 - p0
            e2 = p0 - p1

            # Face normal
            n = np.cross(e1, e2)
            n = n / (np.linalg.norm(n) + 1e-10)

            # Divergence contributions
            X_face = X[i]

            div[v0] += np.dot(X_face, np.cross(n, e0))
            div[v1] += np.dot(X_face, np.cross(n, e1))
            div[v2] += np.dot(X_face, np.cross(n, e2))

        div /= 2.0

        return div

    def _compute_mean_edge_length(self) -> float:
        """Compute mean edge length of mesh"""
        edge_lengths = []

        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            edge_lengths.append(np.linalg.norm(v1 - v0))
            edge_lengths.append(np.linalg.norm(v2 - v1))
            edge_lengths.append(np.linalg.norm(v0 - v2))

        return np.mean(edge_lengths)

    def compute_distance_matrix(
        self,
        source_vertices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Compute geodesic distance matrix for multiple sources

        Args:
            source_vertices: List of source vertex indices (all vertices if None)

        Returns:
            Distance matrix (n_sources, n_vertices)
        """
        if source_vertices is None:
            source_vertices = list(range(self.n_vertices))

        n_sources = len(source_vertices)
        distances = np.zeros((n_sources, self.n_vertices))

        logger.info(f"Computing geodesic distances for {n_sources} sources...")

        for i, source in enumerate(source_vertices):
            distances[i] = self.compute_distance_from_source(source)

            if (i + 1) % max(1, n_sources // 10) == 0:
                logger.debug(f"Processed {i+1}/{n_sources} sources")

        logger.info("Geodesic distance computation complete")

        return distances


def compute_curvature(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and Gaussian curvature at each vertex

    Args:
        vertices: Vertex coordinates (n_vertices, 3)
        faces: Triangle indices (n_faces, 3)

    Returns:
        Tuple of (mean_curvature, gaussian_curvature)
    """
    logger.info("Computing surface curvature...")

    n_vertices = len(vertices)
    mean_curv = np.zeros(n_vertices)
    gauss_curv = np.zeros(n_vertices)

    # Simplified curvature estimation
    # Production would use more robust methods (e.g., osculating circles)

    for i in range(n_vertices):
        # Find neighboring vertices
        neighbors = []
        for face in faces:
            if i in face:
                neighbors.extend([v for v in face if v != i])
        neighbors = list(set(neighbors))

        if len(neighbors) < 3:
            continue

        # Compute local normal
        neighbor_verts = vertices[neighbors]
        center = vertices[i]
        vectors = neighbor_verts - center

        # Estimate normal using PCA
        cov = vectors.T @ vectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # Smallest eigenvalue

        # Estimate mean curvature (simplified)
        projections = vectors @ normal
        mean_curv[i] = np.mean(np.abs(projections))

        # Gaussian curvature (angle deficit method)
        angle_sum = 0
        for j in range(len(neighbors)):
            v1 = vectors[j]
            v2 = vectors[(j + 1) % len(neighbors)]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angle_sum += angle

        gauss_curv[i] = (2 * np.pi - angle_sum) / len(neighbors)

    logger.info("Curvature computation complete")

    return mean_curv, gauss_curv
