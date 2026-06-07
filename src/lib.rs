//! # spectral-shape
//!
//! **Spectral methods for shape analysis of agent behavior distributions.**
//!
//! This crate provides a comprehensive toolkit for analyzing the shape and structure
//! of agent behavior distributions using spectral graph theory. It constructs graph
//! representations of agent interactions, computes spectral descriptors (heat kernel
//! signatures, wave kernel signatures, spectral embeddings), and uses these to classify,
//! cluster, and compare behavioral patterns.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
//! │ Agent        │────▶│ Graph         │────▶│ Laplacian        │
//! │ Interactions │     │ Adjacency A   │     │ L = D - A        │
//! └──────────────┘     └───────────────┘     └───────┬──────────┘
//!                                                     │
//!                                              ┌──────▼──────┐
//!                                              │ Eigenvectors│
//!                                              │ φ₁, φ₂,... │
//!                                              └──────┬──────┘
//!                                    ┌────────────────┼────────────────┐
//!                              ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
//!                              │ Embedding  │   │  Heat      │   │  Wave     │
//!                              │ x→(φ₁,..,φₖ)│  │  K_t=e^{-tL}│  │  ψ=e^{itL}│
//!                              └─────┬──────┘   └─────┬─────┘   └─────┬─────┘
//!                                    │                │                │
//!                              ┌─────▼────────────────▼────────────────▼──────┐
//!                              │              Shape Descriptors               │
//!                              │    HKS + WKS + Spectral Embedding → Feature  │
//!                              └──────────────────────┬───────────────────────┘
//!                                    ┌────────────────┼────────────────┐
//!                              ┌─────▼──────┐  ┌──────▼──────┐  ┌─────▼──────┐
//!                              │ Distance   │  │Classification│ │ Clustering │
//!                              │ Comparison │  │ Stable/...   │ │ k-means    │
//!                              └────────────┘  └──────────────┘  └────────────┘
//! ```
//!
//! ## Theory
//!
//! ### Graph Laplacian
//!
//! Given a graph with adjacency matrix **A** and degree matrix **D = diag(A·1)**:
//!
//! - **Combinatorial Laplacian:** `L = D − A`
//! - **Normalized Laplacian** (Chung 1997): `L_norm = I − D^{−1/2} A D^{−1/2}`
//!
//! The Laplacian is positive semi-definite with eigenvalues `0 = λ₁ ≤ λ₂ ≤ ... ≤ λₙ`.
//! The multiplicity of the zero eigenvalue equals the number of connected components.
//! The second-smallest eigenvalue λ₂ (the *Fiedler value* or *algebraic connectivity*)
//! measures how well-connected the graph is.
//!
//! ### Spectral Embedding (Laplacian Eigenmaps)
//!
//! Following Belkin & Niyogi (2003), we embed graph vertices into ℝᵏ using the first
//! k eigenvectors of the Laplacian corresponding to the smallest non-zero eigenvalues:
//!
//! ```text
//! x_i → (φ₁(x_i), φ₂(x_i), ..., φₖ(x_i))
//! ```
//!
//! This embedding preserves local geometric structure: nearby vertices in the graph
//! remain nearby in the embedding space.
//!
//! ### Heat Kernel
//!
//! The heat kernel describes diffusion on a graph at time *t*:
//!
//! `K_t = e^{−tL} = Σᵢ e^{−tλᵢ} φᵢ φᵢᵀ`
//!
//! The **Heat Kernel Signature** (Sun et al. 2009) at vertex *x* and time *t*:
//!
//! `hks(x, t) = Σᵢ e^{−tλᵢ} φᵢ(x)²`
//!
//! HKS is a multi-scale shape descriptor: small *t* captures fine local detail,
//! large *t* captures global structure. It is invariant to isometric deformations.
//!
//! ### Wave Kernel
//!
//! The wave kernel models quantum-mechanical wave propagation:
//!
//! `ψ_t = e^{itL}`
//!
//! The **Wave Kernel Signature** (Aubry et al. 2011) isolates specific frequency bands:
//!
//! `WKS(x, σ, e) = C_σ · Σᵢ e^{−(e − ln λᵢ)²/(2σ²)} · φᵢ(x)²`
//!
//! where *e* is the log-energy level and *σ* controls bandwidth. Unlike HKS which
//! mixes frequencies at each scale, WKS provides better discriminability by isolating
//! narrow frequency bands.
//!
//! ### Scale-Space Analysis
//!
//! Following Lowe (2004) and Lindeberg (1994), we construct a Gaussian scale-space
//! for 1D behavioral time series:
//!
//! `L(x, σ) = G(x, σ) * f(x)` where `G(x, σ) = (2πσ²)^{−1/2} e^{−x²/(2σ²)}`
//!
//! Features detected at multiple scales (scale-space extrema) indicate stable
//! behavioral patterns that persist across observation granularities.
//!
//! ## Design Decisions
//!
//! ### Why Heat + Wave Dual Descriptors?
//!
//! The heat kernel is dissipative (all energy eventually decays) while the wave kernel
//! is conservative (energy is preserved, creating interference patterns). Together they
//! provide complementary information:
//!
//! | Property     | Heat Kernel         | Wave Kernel            |
//! |-------------|---------------------|------------------------|
//! | Physics     | Diffusion           | Wave propagation       |
//! | Time behavior | Monotone decay    | Oscillatory           |
//! | Sensitivity | Global structure    | Local symmetries      |
//! | Best for    | Connectivity analysis | Symmetry detection  |
//!
//! ### Why Iterative QR (Not Lanczos)?
//!
//! We use QR iteration and power method with deflation rather than Lanczos because:
//!
//! 1. **Simplicity**: QR iteration is straightforward to implement correctly for dense
//!    matrices without worrying about loss of orthogonality in the Krylov subspace.
//! 2. **Full spectrum**: For small agent graphs (n < 100), computing the full spectrum
//!    is fast enough, and QR gives all eigenvalues/eigenvectors simultaneously.
//! 3. **Robustness**: Power method with deflation handles degenerate eigenvalues well.
//! 4. **Transparency**: Each step is easy to debug and understand.
//!
//! For very large graphs (n > 1000), Lanczos or randomized methods would be preferred,
//! but agent behavior graphs are typically small.
//!
//! ### Why Laplacian (Not Adjacency)?
//!
//! The Laplacian has superior spectral properties for shape analysis:
//!
//! - Eigenvalues are non-negative ( PSD)
//! - Zero eigenvalue encodes connectivity
//! - Fiedler value directly measures algebraic connectivity
//! - Relationship to random walks and diffusion processes
//! - Normalized Laplacian eigenvalues bounded in [0, 2]
//!
//! ## Comparison with Alternative Approaches
//!
//! ### Spectral vs GNN
//!
//! | Aspect          | Spectral Methods (this crate) | Graph Neural Networks |
//! |----------------|-------------------------------|-----------------------|
//! | Training       | None required                 | Requires labeled data |
//! | Interpretability | Full mathematical basis     | Black box             |
//! | Small graphs   | Excellent                     | Overkill              |
//! | Scalability    | O(n³) full, O(nk) top-k      | O(n·d²) per layer     |
//! | Stability      | Provable guarantees           | Empirical             |
//!
//! ### Spectral vs t-SNE
//!
//! | Aspect          | Spectral Embedding        | t-SNE                     |
//! |----------------|---------------------------|---------------------------|
//! | Deterministic  | Yes                       | No (random init)          |
//! | Parameters     | k (dimension)             | Perplexity, learning rate |
//! | Theory         | Laplacian eigenmaps       | KL divergence minimization|
//! | Preservation   | Local + global            | Local (clusters)          |
//! | New points     | Project via eigenvectors  | Cannot add new points     |
//!
//! ### Spectral vs UMAP
//!
//! | Aspect          | Spectral Embedding     | UMAP                       |
//! |----------------|------------------------|----------------------------|
//! | Speed          | O(n³) eigendecompose   | O(n^{1.14}) approximate    |
//! | Deterministic  | Yes                    | Approximate                |
//! | Theory         | Spectral graph theory  | Fuzzy topological          |
//! | Out-of-sample  | Direct projection      | Needs transform fit        |
//!
//! ## Performance Characteristics
//!
//! | Operation           | Complexity     | Notes                              |
//! |--------------------|----------------|------------------------------------|
//! | Laplacian build    | O(n²)          | From adjacency matrix              |
//! | QR eigendecompose  | O(n³)          | Full spectrum, small matrices      |
//! | Power method top-k | O(n²·k·iter)  | Approximate, good for large k      |
//! | Heat kernel eval   | O(n²·m)        | m = number of eigenvalues          |
//! | HKS computation    | O(n·m·t)       | t = number of time scales          |
//! | WKS computation    | O(n·m·e)       | e = number of energy levels        |
//! | Spectral embedding | O(n²·k)        | k = embedding dimension            |
//! | k-NN in embedding  | O(n²)          | Brute force, exact                 |
//! | K-means clustering | O(n·d·k·iter) | d = feature dim                    |
//!
//! For agent behavior graphs with n < 100 vertices, all operations complete in
//! milliseconds. The O(n³) eigendecomposition becomes a bottleneck only for
//! graphs with n > 500.
//!
//! ## Glossary
//!
//! - **Adjacency matrix** — Matrix A where A[i][j] = weight of edge (i,j)
//! - **Algebraic connectivity** — Second-smallest eigenvalue of L (Fiedler value)
//! - **Characteristic scale** — Scale at which a signal exhibits maximum structure
//! - **Combinatorial Laplacian** — L = D - A, the basic graph Laplacian
//! - **Degree matrix** — Diagonal matrix D where D[i][i] = sum of row i of A
//! - **Deflation** — Removing a known eigenvalue's contribution to find the next
//! - **Eigendecomposition** — Factoring a matrix into eigenvalues and eigenvectors
//! - **Fiedler vector** — Eigenvector corresponding to the Fiedler value
//! - **Heat kernel** — K_t = exp(-tL), describes diffusion at time t
//! - **HKS** — Heat Kernel Signature: multi-scale local shape descriptor
//! - **Laplacian eigenmaps** — Dimensionality reduction using Laplacian eigenvectors
//! - **Normalized Laplacian** — L_norm = I - D^{-1/2}AD^{-1/2}, eigenvalues in [0,2]
//! - **Power method** — Iterative algorithm for finding the dominant eigenvalue
//! - **QR iteration** — Iterative QR decomposition for full eigendecomposition
//! - **Spectral embedding** — Mapping vertices to ℝᵏ using Laplacian eigenvectors
//! - **Spectral gap** — Difference between smallest non-zero eigenvalues
//! - **WKS** — Wave Kernel Signature: frequency-domain shape descriptor
//!
//! ## References
//!
//! 1. **Belkin, M. & Niyogi, P.** (2003). *Laplacian Eigenmaps for Dimensionality
//!    Reduction and Data Representation.* Neural Computation, 15(6), 1373-1396.
//! 2. **Sun, J., Ovsjanikov, M., & Guibas, L.** (2009). *A Concise and Provably
//!    Informative Multi-Scale Signature Based on Heat Diffusion.* Computer Graphics
//!    Forum, 28(5), 1383-1392.
//! 3. **Aubry, M., Schlickewei, U., & Cremers, D.** (2011). *The Wave Kernel
//!    Signature: A Quantum Mechanical Approach to Shape Analysis.* IEEE
//!    International Conference on Computer Vision (ICCV) Workshops, 1626-1633.
//! 4. **Coifman, R.R. & Lafon, S.** (2006). *Diffusion Maps.* Applied and
//!    Computational Harmonic Analysis, 21(1), 5-30.
//! 5. **Chung, F.R.K.** (1997). *Spectral Graph Theory.* CBMS Regional Conference
//!    Series in Mathematics, No. 92. American Mathematical Society.
//! 6. **Lowe, D.G.** (2004). *Distinctive Image Features from Scale-Invariant
//!    Keypoints.* International Journal of Computer Vision, 60(2), 91-110.
//! 7. **Lindeberg, T.** (1994). *Scale-Space Theory: A Basic Tool for Analyzing
//!    Structures at Multiple Scales.* Journal of Applied Statistics, 21(2), 225-270.
//! 8. **Fiedler, M.** (1973). *Algebraic Connectivity of Graphs.* Czechoslovak
//!    Mathematical Journal, 23(2), 298-305.
//! 9. **Mohar, B.** (1991). *The Laplacian Spectrum of Graphs.* In Y. Alavi et al.
//!    (Eds.), Graph Theory, Combinatorics, and Applications (pp. 871-898).
//!
//! ## Examples
//!
//! See the `examples/` directory for complete runnable examples:
//!
//! - [`spectral_embedding`] — Embed an agent communication graph in 2D
//! - [`heat_kernel_signatures`] — Compute HKS for behavioral analysis
//! - [`shape_distance`] — Compare two agent distributions
//! - [`scale_space_analysis`] — Multi-scale analysis of time-series data

// Matrix operations naturally use index-based loops
#![allow(clippy::needless_range_loop)]

pub mod descriptor;
pub mod embedding;
pub mod heat;
pub mod laplacian;
pub mod scale;
pub mod wave;

pub use descriptor::{
    BehaviorPattern, ShapeClassification, ShapeClustering, ShapeDescriptor, ShapeDistance,
};
pub use embedding::{EmbeddingDistance, KNearestNeighbors, MapGraphPoints, SpectralEmbedding};
pub use heat::{CompareHeatTraces, HeatKernel, HeatKernelSignature, HeatTrace};
pub use laplacian::{
    AlgebraicConnectivity, LaplacianMatrix, NormalizedLaplacian, SpectralGap, power_method,
    qr_eigen, qr_eigenvalues, top_k_eigen,
};
pub use scale::{
    CharacteristicScale, GaussianScaleSpace, ScaleInvariance, ScaleSpace, ScaleSpaceExtrema,
};
pub use wave::{WaveAnalysis, WaveKernel, WaveKernelSignature, WaveTrace};
