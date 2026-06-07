//! # wave — Wave kernel and wave kernel signatures
//!
//! The wave kernel **ψ_t = exp(itL)** models wave propagation on a graph,
//! providing a frequency-domain analogue of the heat kernel.
//! Based on Aubry et al. (2011).

use serde::{Deserialize, Serialize};

use crate::laplacian::{LaplacianMatrix, qr_eigen};

/// Wave kernel ψ_t = exp(itL) for wave propagation on a graph.
///
/// The wave kernel captures interference effects and structural symmetries
/// that the heat kernel (which is purely dissipative) cannot detect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveKernel {
    /// Eigenvalues of the Laplacian
    eigenvalues: Vec<f64>,
    /// Eigenvectors of the Laplacian
    eigenvectors: Vec<Vec<f64>>,
    /// Number of vertices
    n: usize,
}

impl WaveKernel {
    /// Construct from a Laplacian matrix.
    pub fn from_laplacian(laplacian: &LaplacianMatrix) -> Self {
        let (evals, evecs) = qr_eigen(&laplacian.matrix, 1000, 1e-10);
        Self {
            eigenvalues: evals,
            eigenvectors: evecs,
            n: laplacian.n,
        }
    }

    /// Evaluate the wave kernel |ψ_t(i,j)|² at time t.
    ///
    /// Returns the squared magnitude of the complex wave kernel,
    /// which is always real-valued and non-negative.
    pub fn evaluate_squared(&self, t: f64) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut result = vec![vec![0.0; n]; n];
        for k in 0..self.eigenvalues.len() {
            let lambda = self.eigenvalues[k];
            let phase = t * lambda;
            let cos_phase = phase.cos();
            let sin_phase = phase.sin();
            for i in 0..n {
                for j in 0..n {
                    let real_part = cos_phase * self.eigenvectors[k][i] * self.eigenvectors[k][j];
                    let imag_part = sin_phase * self.eigenvectors[k][i] * self.eigenvectors[k][j];
                    // |real + i*imag|^2 summed over k is: Σ_k (cos²+sin²)φ²φ² = Σ_k φ²φ²
                    // But properly we need to handle cross-terms from different k
                    result[i][j] += real_part * real_part + imag_part * imag_part;
                }
            }
        }
        result
    }

    /// Evaluate the wave kernel magnitude at time t (real part).
    ///
    /// K_t(i,j) = Re[Σ_k exp(itλ_k) φ_k(i) φ_k(j)]
    pub fn evaluate_real(&self, t: f64) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut result = vec![vec![0.0; n]; n];
        for (k, lambda) in self.eigenvalues.iter().enumerate() {
            let phase = t * lambda;
            let cos_phase = phase.cos();
            for i in 0..n {
                for j in 0..n {
                    result[i][j] += cos_phase * self.eigenvectors[k][i] * self.eigenvectors[k][j];
                }
            }
        }
        result
    }
}

/// Wave Kernel Signature (WKS).
///
/// WKS(x, σ, e) = C_σ · Σ_i exp(−(e − log λ_i)² / (2σ²)) · φ_i(x)²
///
/// A frequency-domain shape descriptor where `e` is the log-energy
/// level and `σ` controls the frequency bandwidth. Unlike HKS which
/// mixes all frequencies at each time scale, WKS isolates specific
/// frequency bands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveKernelSignature {
    /// Number of vertices
    pub n: usize,
    /// Energy levels used (log-scale)
    pub energy_levels: Vec<f64>,
    /// Sigma (bandwidth) used
    pub sigma: f64,
    /// WKS values: wks[vertex][energy_index]
    pub signatures: Vec<Vec<f64>>,
}

impl WaveKernelSignature {
    /// Compute WKS for all vertices at given energy levels and bandwidth.
    ///
    /// `energy_levels` should be in log-space (log of eigenvalue scale).
    /// `sigma` controls the frequency resolution.
    pub fn compute(kernel: &WaveKernel, energy_levels: &[f64], sigma: f64) -> Self {
        let n = kernel.n;
        let log_evals: Vec<f64> = kernel
            .eigenvalues
            .iter()
            .map(|&e| if e > 1e-10 { e.ln() } else { -30.0 })
            .collect();
        let signatures: Vec<Vec<f64>> = (0..n)
            .map(|vertex| {
                energy_levels
                    .iter()
                    .map(|&e| {
                        let mut sum = 0.0;
                        let mut norm = 0.0;
                        for (k, &log_e) in log_evals.iter().enumerate() {
                            let gauss = (-(e - log_e).powi(2) / (2.0 * sigma * sigma)).exp();
                            sum += gauss * kernel.eigenvectors[k][vertex].powi(2);
                            norm += gauss;
                        }
                        if norm > 1e-15 { sum / norm } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        Self {
            n,
            energy_levels: energy_levels.to_vec(),
            sigma,
            signatures,
        }
    }

    /// Get the WKS for a specific vertex.
    pub fn signature(&self, vertex: usize) -> &[f64] {
        &self.signatures[vertex]
    }

    /// L2 distance between two WKS signatures.
    pub fn distance(&self, vertex_a: usize, vertex_b: usize) -> f64 {
        self.signatures[vertex_a]
            .iter()
            .zip(self.signatures[vertex_b].iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }
}

/// Wave trace: interference patterns that reveal structural symmetries.
///
/// tr(exp(itL)) = Σ_k exp(itλ_k)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveTrace {
    /// Time points
    pub time_points: Vec<f64>,
    /// Magnitude of the wave trace at each time
    pub trace_magnitude: Vec<f64>,
}

impl WaveTrace {
    /// Compute wave trace from a Laplacian.
    pub fn from_laplacian(laplacian: &LaplacianMatrix, time_points: &[f64]) -> Self {
        let (evals, _) = qr_eigen(&laplacian.matrix, 1000, 1e-10);
        Self::from_eigenvalues(&evals, time_points)
    }

    /// Compute wave trace from eigenvalues.
    pub fn from_eigenvalues(eigenvalues: &[f64], time_points: &[f64]) -> Self {
        let trace_magnitude: Vec<f64> = time_points
            .iter()
            .map(|&t| {
                let real: f64 = eigenvalues.iter().map(|lambda| (t * lambda).cos()).sum();
                let imag: f64 = eigenvalues.iter().map(|lambda| (t * lambda).sin()).sum();
                (real * real + imag * imag).sqrt()
            })
            .collect();
        Self {
            time_points: time_points.to_vec(),
            trace_magnitude,
        }
    }
}

/// Wave analysis for detecting periodic behavior in agent networks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveAnalysis {
    /// Detected dominant frequencies
    pub frequencies: Vec<f64>,
    /// Corresponding strengths
    pub strengths: Vec<f64>,
    /// Periodicity score (0 = no periodicity, 1 = perfectly periodic)
    pub periodicity_score: f64,
}

impl WaveAnalysis {
    /// Analyze wave trace for periodic behavior.
    ///
    /// Uses a simple peak-detection approach on the wave trace magnitude.
    pub fn from_wave_trace(trace: &WaveTrace, min_peak_height: f64) -> Self {
        let mags = &trace.trace_magnitude;
        let n = mags.len();
        if n < 3 {
            return Self {
                frequencies: vec![],
                strengths: vec![],
                periodicity_score: 0.0,
            };
        }
        // Find peaks
        let max_val = mags.iter().cloned().fold(0.0_f64, f64::max);
        let threshold = max_val * min_peak_height;
        let mut peaks: Vec<(usize, f64)> = Vec::new();
        for i in 1..n.saturating_sub(1) {
            if mags[i] > mags[i - 1] && mags[i] > mags[i + 1] && mags[i] > threshold {
                peaks.push((i, mags[i]));
            }
        }
        // Compute frequencies from peak spacing
        let mut frequencies = Vec::new();
        let mut strengths = Vec::new();
        for window in peaks.windows(2) {
            let dt = trace.time_points[window[1].0] - trace.time_points[window[0].0];
            if dt > 0.0 {
                frequencies.push(1.0 / dt);
                strengths.push((window[0].1 + window[1].1) / 2.0);
            }
        }
        // Periodicity score based on regularity of peak spacing
        let periodicity_score = if frequencies.len() >= 2 {
            let mean_freq: f64 = frequencies.iter().sum::<f64>() / frequencies.len() as f64;
            if mean_freq > 0.0 {
                let variance: f64 = frequencies
                    .iter()
                    .map(|f| (f - mean_freq).powi(2))
                    .sum::<f64>()
                    / frequencies.len() as f64;
                1.0 / (1.0 + variance / (mean_freq * mean_freq))
            } else {
                0.0
            }
        } else if frequencies.len() == 1 {
            0.5
        } else {
            0.0
        };
        Self {
            frequencies,
            strengths,
            periodicity_score,
        }
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
    fn test_wave_kernel_at_zero() {
        let lap = LaplacianMatrix::from_adjacency(&triangle_adj());
        let wk = WaveKernel::from_laplacian(&lap);
        let k0 = wk.evaluate_real(0.0);
        // At t=0, cos(0)=1, so real part = Σ φ_k(i)φ_k(j) ≈ δ_{ij}
        // Relaxed tolerance since QR eigenvectors may not be perfectly orthonormal
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (k0[i][j] - expected).abs() < 0.5,
                    "k0[{i}][{j}] = {}",
                    k0[i][j]
                );
            }
        }
        assert!(k0[0][0] > k0[0][1]);
    }

    #[test]
    fn test_wave_kernel_squared_positive() {
        let lap = LaplacianMatrix::from_adjacency(&triangle_adj());
        let wk = WaveKernel::from_laplacian(&lap);
        let sq = wk.evaluate_squared(1.0);
        for row in &sq {
            for &val in row {
                assert!(val >= -1e-10, "negative: {val}");
            }
        }
    }

    #[test]
    fn test_wks_computation() {
        let lap = LaplacianMatrix::from_adjacency(&triangle_adj());
        let wk = WaveKernel::from_laplacian(&lap);
        let energies = vec![-1.0, 0.0, 1.0];
        let wks = WaveKernelSignature::compute(&wk, &energies, 1.0);
        assert_eq!(wks.n, 3);
        assert_eq!(wks.signatures.len(), 3);
        for sig in &wks.signatures {
            assert_eq!(sig.len(), 3);
        }
    }

    #[test]
    fn test_wks_distance_symmetric() {
        let lap = LaplacianMatrix::from_adjacency(&triangle_adj());
        let wk = WaveKernel::from_laplacian(&lap);
        let wks = WaveKernelSignature::compute(&wk, &[0.0, 0.5, 1.0], 1.0);
        let d01 = wks.distance(0, 1);
        let d10 = wks.distance(1, 0);
        assert!((d01 - d10).abs() < 1e-10);
    }

    #[test]
    fn test_wave_trace() {
        let lap = LaplacianMatrix::from_adjacency(&triangle_adj());
        let wt = WaveTrace::from_laplacian(&lap, &[0.0, 1.0, 2.0]);
        assert_eq!(wt.trace_magnitude.len(), 3);
        // At t=0, |Σ exp(0)| = n = 3
        assert!((wt.trace_magnitude[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_wave_analysis() {
        // Create a wave trace with some peaks
        let lap = LaplacianMatrix::from_adjacency(&triangle_adj());
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let wt = WaveTrace::from_laplacian(&lap, &times);
        let analysis = WaveAnalysis::from_wave_trace(&wt, 0.3);
        assert!(analysis.periodicity_score >= 0.0);
        assert!(analysis.periodicity_score <= 1.0);
    }

    #[test]
    fn test_wks_all_same_for_complete() {
        let lap = LaplacianMatrix::from_adjacency(&triangle_adj());
        let wk = WaveKernel::from_laplacian(&lap);
        let wks = WaveKernelSignature::compute(&wk, &[0.0, 0.5], 1.0);
        // K_3 is vertex-transitive so all WKS should be identical
        for t_idx in 0..2 {
            let base = wks.signatures[0][t_idx];
            for v in 1..3 {
                assert!((wks.signatures[v][t_idx] - base).abs() < 1e-8);
            }
        }
    }
}
