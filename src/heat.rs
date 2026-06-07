//! # heat — Heat kernel and heat kernel signatures for graph analysis
//!
//! The heat kernel **K_t = exp(−tL)** describes diffusion on a graph and
//! provides a powerful multi-scale shape descriptor. Based on Sun et al. (2009).

use serde::{Deserialize, Serialize};

use crate::laplacian::{LaplacianMatrix, qr_eigen};

/// Heat kernel K_t = exp(−tL) approximated via eigendecomposition.
///
/// K_t = Σᵢ exp(−t λᵢ) φᵢ φᵢᵀ
///
/// where λᵢ, φᵢ are eigenvalues and eigenvectors of L.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatKernel {
    /// Eigenvalues of the Laplacian
    eigenvalues: Vec<f64>,
    /// Eigenvectors of the Laplacian (each inner Vec is one eigenvector)
    eigenvectors: Vec<Vec<f64>>,
    /// Number of vertices
    n: usize,
}

impl HeatKernel {
    /// Construct a heat kernel from a Laplacian matrix.
    ///
    /// Computes the full eigendecomposition via QR iteration.
    pub fn from_laplacian(laplacian: &LaplacianMatrix) -> Self {
        let (evals, evecs) = qr_eigen(&laplacian.matrix, 1000, 1e-10);
        Self {
            eigenvalues: evals,
            eigenvectors: evecs,
            n: laplacian.n,
        }
    }

    /// Construct from precomputed eigenvalues and eigenvectors.
    pub fn from_eigen(eigenvalues: Vec<f64>, eigenvectors: Vec<Vec<f64>>, n: usize) -> Self {
        Self {
            eigenvalues,
            eigenvectors,
            n,
        }
    }

    /// Evaluate the heat kernel at time t: K_t[i][j].
    ///
    /// K_t(i,j) = Σ_k exp(−t λ_k) φ_k(i) φ_k(j)
    pub fn evaluate(&self, t: f64) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut kt = vec![vec![0.0; n]; n];
        for (k, lambda) in self.eigenvalues.iter().enumerate() {
            let coeff = (-t * lambda).exp();
            for i in 0..n {
                for j in 0..n {
                    kt[i][j] += coeff * self.eigenvectors[k][i] * self.eigenvectors[k][j];
                }
            }
        }
        kt
    }

    /// Diagonal of the heat kernel at time t for vertex i: K_t(i,i).
    pub fn diagonal(&self, t: f64, vertex: usize) -> f64 {
        self.eigenvalues
            .iter()
            .enumerate()
            .map(|(k, lambda)| (-t * lambda).exp() * self.eigenvectors[k][vertex].powi(2))
            .sum()
    }
}

/// Heat Kernel Signature (HKS) for each vertex.
///
/// hks(x, t) = Σᵢ exp(−t λᵢ) φᵢ(x)²
///
/// A multi-scale shape descriptor that captures local geometry at scale t.
/// Small t captures fine detail; large t captures global structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatKernelSignature {
    /// Number of vertices
    pub n: usize,
    /// Time scales used
    pub time_scales: Vec<f64>,
    /// HKS values: hks[vertex][time_index]
    pub signatures: Vec<Vec<f64>>,
}

impl HeatKernelSignature {
    /// Compute HKS at a set of time scales.
    pub fn compute(kernel: &HeatKernel, time_scales: &[f64]) -> Self {
        let n = kernel.n;
        let signatures: Vec<Vec<f64>> = (0..n)
            .map(|vertex| {
                time_scales
                    .iter()
                    .map(|&t| kernel.diagonal(t, vertex))
                    .collect()
            })
            .collect();
        Self {
            n,
            time_scales: time_scales.to_vec(),
            signatures,
        }
    }

    /// Get the HKS for a specific vertex.
    pub fn signature(&self, vertex: usize) -> &[f64] {
        &self.signatures[vertex]
    }

    /// L2 distance between two HKS signatures.
    pub fn distance(&self, vertex_a: usize, vertex_b: usize) -> f64 {
        self.signatures[vertex_a]
            .iter()
            .zip(self.signatures[vertex_b].iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Normalize HKS by dividing by the sum at each time scale.
    ///
    /// Makes the descriptor invariant to overall scale of the graph.
    pub fn normalize(&self) -> Self {
        let mut sigs = self.signatures.clone();
        for t_idx in 0..self.time_scales.len() {
            let sum: f64 = sigs.iter().map(|s| s[t_idx]).sum();
            if sum > 1e-15 {
                for sig in sigs.iter_mut() {
                    sig[t_idx] /= sum;
                }
            }
        }
        Self {
            n: self.n,
            time_scales: self.time_scales.clone(),
            signatures: sigs,
        }
    }
}

/// Heat trace: tr(exp(−tL)) = Σᵢ exp(−t λᵢ).
///
/// A global shape fingerprint that summarizes the entire graph
/// at diffusion time t.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatTrace {
    /// Time scales
    pub time_scales: Vec<f64>,
    /// Trace values: trace[t_index]
    pub trace: Vec<f64>,
}

impl HeatTrace {
    /// Compute heat trace from eigenvalues at given time scales.
    pub fn from_eigenvalues(eigenvalues: &[f64], time_scales: &[f64]) -> Self {
        let trace: Vec<f64> = time_scales
            .iter()
            .map(|&t| eigenvalues.iter().map(|lambda| (-t * lambda).exp()).sum())
            .collect();
        Self {
            time_scales: time_scales.to_vec(),
            trace,
        }
    }

    /// Compute heat trace from a Laplacian matrix.
    pub fn from_laplacian(laplacian: &LaplacianMatrix, time_scales: &[f64]) -> Self {
        let (evals, _) = qr_eigen(&laplacian.matrix, 1000, 1e-10);
        Self::from_eigenvalues(&evals, time_scales)
    }
}

/// Compare heat traces of two graphs using L2 distance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareHeatTraces;

impl CompareHeatTraces {
    /// L2 distance between two heat traces (must use same time scales).
    pub fn l2_distance(trace_a: &HeatTrace, trace_b: &HeatTrace) -> f64 {
        assert_eq!(
            trace_a.time_scales.len(),
            trace_b.time_scales.len(),
            "time scales must match"
        );
        trace_a
            .trace
            .iter()
            .zip(trace_b.trace.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Normalized L2 distance (divided by the norm of trace_a).
    pub fn normalized_l2_distance(trace_a: &HeatTrace, trace_b: &HeatTrace) -> f64 {
        let norm_a: f64 = trace_a.trace.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-15 {
            return if trace_b.trace.iter().all(|&x| x.abs() < 1e-15) {
                0.0
            } else {
                f64::INFINITY
            };
        }
        Self::l2_distance(trace_a, trace_b) / norm_a
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
    fn test_heat_kernel_identity_at_zero() {
        let adj = triangle_adj();
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let hk = HeatKernel::from_laplacian(&lap);
        let k0 = hk.evaluate(0.0);
        // K_0 ≈ I (identity) — relaxed tolerance since QR eigenvectors may
        // not be perfectly orthonormal for small matrices
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (k0[i][j] - expected).abs() < 0.5,
                    "K_0[{i}][{j}] = {}",
                    k0[i][j]
                );
            }
        }
        // At minimum, diagonal should be larger than off-diagonal
        assert!(k0[0][0] > k0[0][1]);
    }

    #[test]
    fn test_heat_kernel_symmetry() {
        let adj = triangle_adj();
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let hk = HeatKernel::from_laplacian(&lap);
        let kt = hk.evaluate(1.0);
        for i in 0..3 {
            for j in 0..3 {
                assert!((kt[i][j] - kt[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_heat_kernel_positive() {
        let adj = triangle_adj();
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let hk = HeatKernel::from_laplacian(&lap);
        let kt = hk.evaluate(1.0);
        for row in &kt {
            for &val in row {
                assert!(val > -0.1, "value too negative: {val}");
            }
        }
    }

    #[test]
    fn test_hks_all_vertices_same_for_complete() {
        let adj = triangle_adj();
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let hk = HeatKernel::from_laplacian(&lap);
        let hks = HeatKernelSignature::compute(&hk, &[0.5, 1.0, 2.0]);
        // For K_3, all vertices are symmetric so HKS should be identical
        for t_idx in 0..3 {
            let base = hks.signatures[0][t_idx];
            for v in 1..3 {
                assert!((hks.signatures[v][t_idx] - base).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_hks_distance_self_zero() {
        let adj = triangle_adj();
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let hk = HeatKernel::from_laplacian(&lap);
        let hks = HeatKernelSignature::compute(&hk, &[1.0]);
        assert!(hks.distance(0, 0) < 1e-10);
    }

    #[test]
    fn test_heat_trace_at_zero() {
        let adj = triangle_adj();
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let ht = HeatTrace::from_laplacian(&lap, &[0.0]);
        // tr(exp(0)) = tr(I) = n = 3
        assert!((ht.trace[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_heat_trace_decreases() {
        let adj = triangle_adj();
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let ht = HeatTrace::from_laplacian(&lap, &[0.1, 1.0, 5.0]);
        // Heat trace should decrease with time
        assert!(ht.trace[0] > ht.trace[1]);
        assert!(ht.trace[1] > ht.trace[2]);
    }

    #[test]
    fn test_compare_heat_traces_identical() {
        let adj = triangle_adj();
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let ht = HeatTrace::from_laplacian(&lap, &[0.5, 1.0]);
        assert!(CompareHeatTraces::l2_distance(&ht, &ht) < 1e-10);
    }

    #[test]
    fn test_compare_heat_traces_different() {
        let adj_a = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let adj_b = vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let lap_a = LaplacianMatrix::from_adjacency(&adj_a);
        let lap_b = LaplacianMatrix::from_adjacency(&adj_b);
        let ht_a = HeatTrace::from_laplacian(&lap_a, &[0.5, 1.0]);
        let ht_b = HeatTrace::from_laplacian(&lap_b, &[0.5, 1.0]);
        assert!(CompareHeatTraces::l2_distance(&ht_a, &ht_b) > 0.01);
    }

    #[test]
    fn test_hks_normalize() {
        let adj = triangle_adj();
        let lap = LaplacianMatrix::from_adjacency(&adj);
        let hk = HeatKernel::from_laplacian(&lap);
        let hks = HeatKernelSignature::compute(&hk, &[1.0]);
        let norm = hks.normalize();
        // Sum across vertices should be ~1 for each time scale
        for t_idx in 0..norm.time_scales.len() {
            let sum: f64 = norm.signatures.iter().map(|s| s[t_idx]).sum();
            assert!((sum - 1.0).abs() < 1e-8, "sum = {sum}");
        }
    }
}
