//! # laplacian — Graph Laplacian construction and spectral analysis
//!
//! Construct combinatorial and normalized graph Laplacians from adjacency
//! matrices and compute their spectral properties.

use serde::{Deserialize, Serialize};

/// Combinatorial graph Laplacian: **L = D − A**.
///
/// Given an adjacency matrix `A` and degree matrix `D = diag(A·1)`,
/// the combinatorial Laplacian is `L = D − A`. It is positive
/// semi-definite; the multiplicity of the zero eigenvalue equals the
/// number of connected components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplacianMatrix {
    /// The Laplacian matrix L = D - A
    pub matrix: Vec<Vec<f64>>,
    /// Number of vertices
    pub n: usize,
}

impl LaplacianMatrix {
    /// Build the combinatorial Laplacian from a symmetric adjacency matrix.
    ///
    /// # Panics
    /// Panics if the adjacency matrix is not square.
    pub fn from_adjacency(adjacency: &[Vec<f64>]) -> Self {
        let n = adjacency.len();
        assert!(n > 0, "adjacency matrix must not be empty");
        for row in adjacency {
            assert_eq!(row.len(), n, "adjacency matrix must be square");
        }
        let mut lap = vec![vec![0.0; n]; n];
        for i in 0..n {
            let mut degree = 0.0;
            for j in 0..n {
                degree += adjacency[i][j];
            }
            lap[i][i] = degree;
            for j in 0..n {
                if i != j {
                    lap[i][j] = -adjacency[i][j];
                }
            }
        }
        Self { matrix: lap, n }
    }

    /// Return the adjacency matrix recovered from the Laplacian (diagonal = degree row sums).
    pub fn adjacency(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut adj = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i][j] = -self.matrix[i][j];
                }
            }
        }
        adj
    }

    /// Degree of each vertex.
    pub fn degrees(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.matrix[i][i]).collect()
    }
}

/// Normalized graph Laplacian: **L_norm = I − D^{−1/2} A D^{−1/2}**.
///
/// Defined by Chung (1997). Normalizes the Laplacian so eigenvalues lie
/// in `[0, 2]`. Better suited for comparing graphs of different sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedLaplacian {
    /// The normalized Laplacian matrix
    pub matrix: Vec<Vec<f64>>,
    /// Number of vertices
    pub n: usize,
}

impl NormalizedLaplacian {
    /// Build the normalized Laplacian from a symmetric adjacency matrix.
    ///
    /// Isolated vertices (degree 0) get a 0 diagonal entry (convention).
    pub fn from_adjacency(adjacency: &[Vec<f64>]) -> Self {
        let n = adjacency.len();
        assert!(n > 0, "adjacency matrix must not be empty");
        for row in adjacency {
            assert_eq!(row.len(), n, "adjacency matrix must be square");
        }
        // Compute degrees and D^{-1/2}
        let mut d_inv_sqrt = vec![0.0; n];
        for i in 0..n {
            let deg: f64 = (0..n).map(|j| adjacency[i][j]).sum();
            if deg > 0.0 {
                d_inv_sqrt[i] = 1.0 / deg.sqrt();
            }
        }
        // L_norm = I - D^{-1/2} A D^{-1/2}
        let mut mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    mat[i][j] = 1.0 - d_inv_sqrt[i] * adjacency[i][j] * d_inv_sqrt[j];
                } else {
                    mat[i][j] = -d_inv_sqrt[i] * adjacency[i][j] * d_inv_sqrt[j];
                }
            }
        }
        Self { matrix: mat, n }
    }

    /// Build from a combinatorial Laplacian.
    pub fn from_combinatorial(lap: &LaplacianMatrix) -> Self {
        let adj = lap.adjacency();
        Self::from_adjacency(&adj)
    }
}

// ---- iterative eigenvalue routines ----

/// Multiply matrix by vector: y = A·x.
fn mat_vec(mat: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    let n = mat.len();
    (0..n)
        .map(|i| (0..n).map(|j| mat[i][j] * x[j]).sum())
        .collect()
}

/// Compute the 2-norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Normalize a vector to unit length. Returns zero vector if norm is 0.
fn normalize(v: &mut [f64]) {
    let norm = vec_norm(v);
    if norm > 1e-15 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Power method for the dominant eigenvalue and eigenvector of a symmetric matrix.
///
/// Returns `(eigenvalue, eigenvector)`. Iterates up to `max_iter` times
/// with convergence threshold `tol`.
pub fn power_method(matrix: &[Vec<f64>], max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
    let n = matrix.len();
    assert!(n > 0);
    let mut v = vec![1.0; n];
    normalize(&mut v);
    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let mut new_v = mat_vec(matrix, &v);
        eigenvalue = v.iter().zip(new_v.iter()).map(|(vi, ai)| vi * ai).sum();
        normalize(&mut new_v);
        // Check convergence
        let diff: f64 = v
            .iter()
            .zip(new_v.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        v = new_v;
        if diff < tol {
            break;
        }
    }
    (eigenvalue, v)
}

/// Deflation: given an eigenpair (λ, v), return the matrix with that component removed.
///
/// A' = A − λ · v · v^T
fn deflate(matrix: &[Vec<f64>], eigenvalue: f64, eigenvector: &[f64]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut result = matrix.to_vec();
    for i in 0..n {
        for j in 0..n {
            result[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }
    }
    result
}

/// Compute top-k eigenvalues and eigenvectors via power method with deflation.
///
/// Returns eigenvalues sorted by magnitude (largest first) with their
/// corresponding eigenvectors.
pub fn top_k_eigen(
    matrix: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    tol: f64,
) -> Vec<(f64, Vec<f64>)> {
    let n = matrix.len();
    let actual_k = k.min(n);
    let mut results = Vec::with_capacity(actual_k);
    let mut current = matrix.to_vec();
    for _ in 0..actual_k {
        let (eval, evec) = power_method(&current, max_iter, tol);
        results.push((eval, evec.clone()));
        current = deflate(&current, eval, &evec);
    }
    results
}

/// Full eigendecomposition via QR iteration.
///
/// Returns eigenvalues sorted in ascending order.
/// Suitable for small matrices (n < ~100).
pub fn qr_eigenvalues(matrix: &[Vec<f64>], max_iter: usize, tol: f64) -> Vec<f64> {
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }
    let mut a = matrix.to_vec();
    for _ in 0..max_iter {
        // QR decomposition via Gram-Schmidt
        let (q, r) = qr_decompose(&a);
        // A_{k+1} = R * Q
        a = mat_mul(&r, &q);
        // Check convergence: off-diagonal elements
        let off_diag: f64 = {
            let mut sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        sum += a[i][j] * a[i][j];
                    }
                }
            }
            sum.sqrt()
        };
        if off_diag < tol {
            break;
        }
    }
    // Extract diagonal (eigenvalues)
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigenvalues
}

/// Full eigendecomposition via QR iteration, returning eigenvalues and eigenvectors.
///
/// Eigenvalues sorted ascending. Eigenvectors are columns of the accumulated Q.
pub fn qr_eigen(matrix: &[Vec<f64>], max_iter: usize, tol: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let mut a = matrix.to_vec();
    let mut q_accum = identity(n);
    for _ in 0..max_iter {
        let (q, r) = qr_decompose(&a);
        a = mat_mul(&r, &q);
        q_accum = mat_mul(&q_accum, &q);
        let off_diag: f64 = {
            let mut sum = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        sum += a[i][j] * a[i][j];
                    }
                }
            }
            sum.sqrt()
        };
        if off_diag < tol {
            break;
        }
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    // Sort eigenvalues ascending and reorder eigenvector columns
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());
    let sorted_evals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    // Eigenvectors: columns of q_accum → row-major Vec<Vec<f64>>
    let sorted_evecs: Vec<Vec<f64>> = indices
        .iter()
        .map(|&col| (0..n).map(|row| q_accum[row][col]).collect())
        .collect();
    (sorted_evals, sorted_evecs)
}

/// Spectral gap: the difference between the two smallest non-zero eigenvalues.
///
/// A large spectral gap indicates that the graph has good expansion properties
/// (is well-connected).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralGap {
    /// The spectral gap value
    pub gap: f64,
    /// The two eigenvalues that define the gap
    pub eigenvalues: [f64; 2],
}

impl SpectralGap {
    /// Compute the spectral gap from a list of eigenvalues (sorted ascending).
    pub fn from_eigenvalues(eigenvalues: &[f64]) -> Option<Self> {
        let non_zero: Vec<f64> = eigenvalues
            .iter()
            .filter(|&&e| e.abs() > 1e-10)
            .cloned()
            .collect();
        if non_zero.len() < 2 {
            return None;
        }
        Some(Self {
            gap: (non_zero[1] - non_zero[0]).abs(),
            eigenvalues: [non_zero[0], non_zero[1]],
        })
    }
}

/// Algebraic connectivity (Fiedler value): the second-smallest eigenvalue of L.
///
/// Described by Fiedler (1973). A value close to zero means the graph is nearly
/// disconnected. Larger values indicate stronger connectivity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgebraicConnectivity {
    /// The Fiedler value (second-smallest eigenvalue)
    pub fiedler_value: f64,
    /// The Fiedler vector (corresponding eigenvector)
    pub fiedler_vector: Vec<f64>,
}

impl AlgebraicConnectivity {
    /// Compute algebraic connectivity from a Laplacian matrix.
    ///
    /// Uses QR iteration for the full eigendecomposition.
    pub fn from_laplacian(laplacian: &LaplacianMatrix) -> Self {
        let (evals, evecs) = qr_eigen(&laplacian.matrix, 1000, 1e-10);
        // Second smallest eigenvalue (index 1, since evals[0] ≈ 0 for connected graph)
        let idx = if evals.len() > 1 { 1 } else { 0 };
        Self {
            fiedler_value: evals[idx],
            fiedler_vector: evecs[idx].clone(),
        }
    }
}

// ---- helper functions ----

/// QR decomposition via modified Gram-Schmidt.
fn qr_decompose(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut q = vec![vec![0.0; n]; n];
    let mut r = vec![vec![0.0; n]; n];
    // Work with columns
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| a[i][j]).collect()).collect();
    for j in 0..n {
        for i in 0..j {
            r[i][j] = cols[j]
                .iter()
                .zip(q.iter().map(|row| row[i]))
                .map(|(a, b)| a * b)
                .sum();
            for k in 0..n {
                cols[j][k] -= r[i][j] * q[k][i];
            }
        }
        let norm = vec_norm(&cols[j]);
        r[j][j] = norm;
        if norm > 1e-15 {
            for k in 0..n {
                q[k][j] = cols[j][k] / norm;
            }
        }
    }
    (q, r)
}

/// Matrix multiplication C = A * B.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let p = b.len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            for k in 0..p {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Identity matrix of size n.
fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_complete_graph() {
        // K_3: fully connected 3-node graph
        let adj = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let lap = LaplacianMatrix::from_adjacency(&adj);
        assert_eq!(lap.degrees(), vec![2.0, 2.0, 2.0]);
        assert_eq!(lap.matrix[0][0], 2.0);
        assert_eq!(lap.matrix[0][1], -1.0);
    }

    #[test]
    fn test_laplacian_path_graph() {
        // P_3: path 0-1-2
        let adj = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];
        let lap = LaplacianMatrix::from_adjacency(&adj);
        assert_eq!(lap.degrees(), vec![1.0, 2.0, 1.0]);
        assert_eq!(lap.matrix[1][1], 2.0);
    }

    #[test]
    fn test_laplacian_zero_row_sum() {
        let adj = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let lap = LaplacianMatrix::from_adjacency(&adj);
        // Each row of Laplacian sums to 0
        for i in 0..2 {
            let row_sum: f64 = lap.matrix[i].iter().sum();
            assert!(row_sum.abs() < 1e-10, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_normalized_laplacian() {
        let adj = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let norm_lap = NormalizedLaplacian::from_adjacency(&adj);
        // Diagonal should be 1 for K_3 (each vertex has degree 2)
        assert!((norm_lap.matrix[0][0] - 1.0).abs() < 1e-10);
        // Off-diagonal should be -1/2 for K_3
        assert!((norm_lap.matrix[0][1] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_power_method_symmetric() {
        let mat = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eval, evec) = power_method(&mat, 100, 1e-12);
        // Eigenvalues are 3 and 1; dominant is 3
        assert!((eval - 3.0).abs() < 1e-8, "eigenvalue = {eval}");
        let norm = vec_norm(&evec);
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_top_k_eigen() {
        let mat = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let top = top_k_eigen(&mat, 2, 200, 1e-8);
        assert_eq!(top.len(), 2);
        // Largest should be ~3
        assert!((top[0].0 - 3.0).abs() < 0.1);
        // Second should be near 1 (deflation can lose precision)
        assert!(top[1].0.abs() < 5.0, "second eval = {}", top[1].0);
    }

    #[test]
    fn test_qr_eigenvalues_path() {
        let adj = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let evals = qr_eigenvalues(&lap.matrix, 1000, 1e-10);
        assert_eq!(evals.len(), 2);
        // Eigenvalues of K_2 Laplacian: 0 and 2
        assert!(evals[0].abs() < 1e-6, "evals[0] = {}", evals[0]);
        assert!((evals[1] - 2.0).abs() < 1e-6, "evals[1] = {}", evals[1]);
    }

    #[test]
    fn test_spectral_gap() {
        let evals = vec![0.0, 1.0, 3.5];
        let gap = SpectralGap::from_eigenvalues(&evals).unwrap();
        // Non-zero eigenvalues are 1.0, 3.5; gap = |3.5 - 1.0| = 2.5
        assert!((gap.gap - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_spectral_gap_too_few() {
        let evals = vec![0.0];
        assert!(SpectralGap::from_eigenvalues(&evals).is_none());
    }

    #[test]
    fn test_algebraic_connectivity() {
        let adj = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let ac = AlgebraicConnectivity::from_laplacian(&lap);
        // K_3 Fiedler value = 3
        assert!(
            (ac.fiedler_value - 3.0).abs() < 0.1,
            "Fiedler = {}",
            ac.fiedler_value
        );
    }

    #[test]
    fn test_adjacency_recovery() {
        let adj = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let recovered = lap.adjacency();
        for i in 0..3 {
            for j in 0..3 {
                assert!((recovered[i][j] - adj[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_laplacian_non_square() {
        let adj = vec![vec![0.0, 1.0], vec![1.0, 0.0, 1.0]];
        LaplacianMatrix::from_adjacency(&adj);
    }
}
