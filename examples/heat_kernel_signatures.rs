//! Heat kernel signatures for behavioral analysis.
//!
//! Shows how HKS captures local shape at multiple scales and can
//! identify structurally similar vs different agents.

use spectral_shape::{
    CompareHeatTraces, HeatKernel, HeatKernelSignature, HeatTrace, LaplacianMatrix,
};

fn main() {
    println!("=== Heat Kernel Signatures for Behavioral Analysis ===\n");

    // Agent interaction graph: a ring of 4 agents
    let adjacency = vec![
        vec![0.0, 1.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0, 0.0],
    ];

    let lap = LaplacianMatrix::from_adjacency(&adjacency);

    // Build heat kernel
    let hk = HeatKernel::from_laplacian(&lap);

    // Multi-scale time parameters
    let time_scales = vec![0.1, 0.5, 1.0, 2.0, 5.0];

    println!("Time scales: {:?}", time_scales);
    println!();

    // Compute HKS
    let hks = HeatKernelSignature::compute(&hk, &time_scales);

    println!("Heat Kernel Signatures:");
    for v in 0..4 {
        let sig = hks.signature(v);
        let formatted: Vec<String> = sig.iter().map(|v| format!("{:.6}", v)).collect();
        println!("  Agent {v}: [{}]", formatted.join(", "));
    }
    println!();

    // In a ring graph, all vertices are equivalent → HKS should be identical
    println!("Ring graph → all agents have identical HKS (by symmetry):");
    for v in 1..4 {
        let dist = hks.distance(0, v);
        println!("  HKS distance Agent 0 ↔ Agent {v}: {:.10}", dist);
    }
    println!();

    // Compare with a different graph topology
    let adj_path = vec![
        vec![0.0, 1.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    let lap_path = LaplacianMatrix::from_adjacency(&adj_path);

    // Heat trace comparison
    let ht_ring = HeatTrace::from_laplacian(&lap, &time_scales);
    let ht_path = HeatTrace::from_laplacian(&lap_path, &time_scales);

    println!("Heat Traces:");
    println!(
        "  Ring graph: {:?}",
        ht_ring
            .trace
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );
    println!(
        "  Path graph: {:?}",
        ht_path
            .trace
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
    );
    println!();

    let l2_dist = CompareHeatTraces::l2_distance(&ht_ring, &ht_path);
    let norm_dist = CompareHeatTraces::normalized_l2_distance(&ht_ring, &ht_path);
    println!("Heat trace L2 distance: {:.6}", l2_dist);
    println!("Heat trace normalized L2 distance: {:.6}", norm_dist);

    println!("\nThe ring and path graphs have different heat traces,");
    println!("revealing their structural differences through diffusion dynamics.");
}
