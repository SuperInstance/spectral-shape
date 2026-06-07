//! # embedding — Spectral embedding of graph vertices into low-dimensional space
//!
//! Uses the first k eigenvectors of the graph Laplacian to embed vertices
//! into ℝᵏ, following the Laplacian Eigenmaps framework of Belkin & Niyogi (2003).

use serde::{Deserialize, Serialize};

use crate::laplacian::{LaplacianMatrix, NormalizedLaplacian, qr_eigen};

/// Spectral embedding of graph vertices into ℝᵏ.
///
/// The embedding maps each vertex `v` to the point
/// `(φ₁(v), φ₂(v), …, φₖ(v))` where `φᵢ` are the eigenvectors
/// of the graph Laplacian corresponding to the smallest non-zero eigenvalues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralEmbedding {
    /// Embedding dimension k
    pub dim: usize,
    /// Embedding vectors for each vertex: Vec of length n, each a Vec<f64> of length k
    pub points: Vec<Vec<f64>>,
    /// Eigenvalues used (smallest k non-zero)
    pub eigenvalues: Vec<f64>,
    /// Number of vertices
    pub n: usize,
}

impl SpectralEmbedding {
    /// Compute spectral embedding from an adjacency matrix.
    ///
    /// Uses the combinatorial Laplacian and takes the first `dim` eigenvectors
    /// corresponding to the smallest non-zero eigenvalues.
    pub fn from_adjacency(adjacency: &[Vec<f64>], dim: usize) -> Self {
        let lap = LaplacianMatrix::from_adjacency(adjacency);
        Self::from_laplacian(&lap, dim)
    }

    /// Compute spectral embedding from a combinatorial Laplacian.
    pub fn from_laplacian(laplacian: &LaplacianMatrix, dim: usize) -> Self {
        let (evals, evecs) = qr_eigen(&laplacian.matrix, 1000, 1e-10);
        Self::from_eigen(&evals, &evecs, dim)
    }

    /// Compute spectral embedding using the normalized Laplacian.
    pub fn from_normalized(adjacency: &[Vec<f64>], dim: usize) -> Self {
        let norm_lap = NormalizedLaplacian::from_adjacency(adjacency);
        let (evals, evecs) = qr_eigen(&norm_lap.matrix, 1000, 1e-10);
        Self::from_eigen(&evals, &evecs, dim)
    }

    /// Build embedding from precomputed eigenvalues and eigenvectors.
    pub fn from_eigen(eigenvalues: &[f64], eigenvectors: &[Vec<f64>], dim: usize) -> Self {
        let n = eigenvectors.first().map_or(0, |v| v.len());
        // Skip the trivial eigenvector (eigenvalue ≈ 0), take next `dim`
        let start_idx = if eigenvalues.first().is_some_and(|&e| e.abs() < 1e-8) {
            1
        } else {
            0
        };
        let actual_dim = dim.min(eigenvalues.len().saturating_sub(start_idx));
        let used_evals: Vec<f64> = (start_idx..start_idx + actual_dim)
            .map(|i| eigenvalues[i])
            .collect();
        // Each vertex i gets a point from eigenvectors at that index
        let points: Vec<Vec<f64>> = (0..n)
            .map(|vertex| {
                (0..actual_dim)
                    .map(|d| eigenvectors[start_idx + d][vertex])
                    .collect()
            })
            .collect();
        Self {
            dim: actual_dim,
            points,
            eigenvalues: used_evals,
            n,
        }
    }
}

/// Map graph vertices into ℝᵏ using spectral embedding.
///
/// Convenience wrapper around [`SpectralEmbedding`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapGraphPoints {
    /// The spectral embedding
    pub embedding: SpectralEmbedding,
}

impl MapGraphPoints {
    /// Create a new vertex mapping from an adjacency matrix.
    pub fn new(adjacency: &[Vec<f64>], dim: usize) -> Self {
        Self {
            embedding: SpectralEmbedding::from_adjacency(adjacency, dim),
        }
    }

    /// Get the embedding of a specific vertex.
    pub fn point(&self, vertex: usize) -> &[f64] {
        &self.embedding.points[vertex]
    }
}

/// Euclidean distance between vertices in spectral embedding space.
///
/// This distance captures graph structure better than raw (geodesic)
/// distance because the spectral embedding respects the manifold
/// geometry of the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingDistance;

impl EmbeddingDistance {
    /// Compute Euclidean distance between two points in embedding space.
    pub fn euclidean(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// Compute squared Euclidean distance (avoids sqrt for comparisons).
    pub fn euclidean_squared(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Compute cosine distance between two points.
    pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-15 || norm_b < 1e-15 {
            return 1.0;
        }
        1.0 - dot / (norm_a * norm_b)
    }
}

/// k-nearest neighbors in spectral embedding space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNearestNeighbors {
    /// For each vertex, its k nearest neighbors as (index, distance) pairs.
    pub neighbors: Vec<Vec<(usize, f64)>>,
}

impl KNearestNeighbors {
    /// Compute k-NN from a spectral embedding.
    pub fn from_embedding(embedding: &SpectralEmbedding, k: usize) -> Self {
        let n = embedding.n;
        let actual_k = k.min(n.saturating_sub(1));
        let mut neighbors = Vec::with_capacity(n);
        for i in 0..n {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    (
                        j,
                        EmbeddingDistance::euclidean(&embedding.points[i], &embedding.points[j]),
                    )
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(actual_k);
            neighbors.push(dists);
        }
        Self { neighbors }
    }

    /// Get the k nearest neighbors of vertex i.
    pub fn neighbors_of(&self, vertex: usize) -> &[(usize, f64)] {
        &self.neighbors[vertex]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_adj() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn test_spectral_embedding_dim() {
        let adj = triangle_adj();
        let emb = SpectralEmbedding::from_adjacency(&adj, 2);
        assert_eq!(emb.dim, 2);
        assert_eq!(emb.points.len(), 3);
        for p in &emb.points {
            assert_eq!(p.len(), 2);
        }
    }

    #[test]
    fn test_spectral_embedding_normalized() {
        let adj = triangle_adj();
        let emb = SpectralEmbedding::from_normalized(&adj, 2);
        assert_eq!(emb.dim, 2);
    }

    #[test]
    fn test_map_graph_points() {
        let adj = triangle_adj();
        let mgp = MapGraphPoints::new(&adj, 2);
        assert_eq!(mgp.embedding.n, 3);
        assert_eq!(mgp.point(0).len(), 2);
    }

    #[test]
    fn test_embedding_distance_self() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((EmbeddingDistance::euclidean(&a, &a)).abs() < 1e-10);
    }

    #[test]
    fn test_embedding_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![[3.0_f64, 4.0_f64][0], 4.0];
        assert!((EmbeddingDistance::euclidean(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance_same() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((EmbeddingDistance::cosine(&a, &a)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((EmbeddingDistance::cosine(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_knn() {
        let adj = vec![
            vec![0.0, 1.0, 0.5, 0.0],
            vec![1.0, 0.0, 0.5, 0.0],
            vec![0.5, 0.5, 0.0, 1.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let emb = SpectralEmbedding::from_adjacency(&adj, 2);
        let knn = KNearestNeighbors::from_embedding(&emb, 2);
        assert_eq!(knn.neighbors.len(), 4);
        for n in &knn.neighbors {
            assert!(n.len() <= 2);
        }
    }

    #[test]
    fn test_knn_triangle() {
        let adj = triangle_adj();
        let emb = SpectralEmbedding::from_adjacency(&adj, 1);
        let knn = KNearestNeighbors::from_embedding(&emb, 2);
        assert_eq!(knn.neighbors_of(0).len(), 2);
    }
}
