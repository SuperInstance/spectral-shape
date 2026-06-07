//! Shape distance between two agent distributions.
//!
//! Demonstrates building shape descriptors for two agent groups
//! and computing their spectral distance for behavioral comparison.

use spectral_shape::{BehaviorPattern, ShapeClassification, ShapeDescriptor, ShapeDistance};

fn main() {
    println!("=== Shape Distance Between Agent Distributions ===\n");

    // Group A: tightly connected cluster (stable team)
    let group_a = vec![
        vec![0.0, 0.9, 0.8, 0.7],
        vec![0.9, 0.0, 0.8, 0.6],
        vec![0.8, 0.8, 0.0, 0.9],
        vec![0.7, 0.6, 0.9, 0.0],
    ];

    // Group B: loosely connected chain (transitional team)
    let group_b = vec![
        vec![0.0, 0.3, 0.0, 0.0],
        vec![0.3, 0.0, 0.2, 0.0],
        vec![0.0, 0.2, 0.0, 0.1],
        vec![0.0, 0.0, 0.1, 0.0],
    ];

    // Group C: similar to group A (another stable team)
    let group_c = vec![
        vec![0.0, 0.8, 0.9, 0.7],
        vec![0.8, 0.0, 0.7, 0.8],
        vec![0.9, 0.7, 0.0, 0.9],
        vec![0.7, 0.8, 0.9, 0.0],
    ];

    // Build shape descriptors
    let desc_a = ShapeDescriptor::from_adjacency(&group_a);
    let desc_b = ShapeDescriptor::from_adjacency(&group_b);
    let desc_c = ShapeDescriptor::from_adjacency(&group_c);

    println!("Shape descriptor dimensions: {}", desc_a.feature_dim());
    println!("  (5 HKS + 5 WKS + 3 spectral embedding = 13 features per vertex)\n");

    // Classify each group
    let cls_a = ShapeClassification::classify(&desc_a);
    let cls_b = ShapeClassification::classify(&desc_b);
    let cls_c = ShapeClassification::classify(&desc_c);

    println!("Behavior Classification:");
    print_classification("Group A (tight cluster)", &cls_a);
    print_classification("Group B (loose chain)", &cls_b);
    print_classification("Group C (tight cluster)", &cls_c);

    // Compute pairwise distances
    let dist_ab = ShapeDistance::global_distance(&desc_a, &desc_b);
    let dist_ac = ShapeDistance::global_distance(&desc_a, &desc_c);
    let dist_bc = ShapeDistance::global_distance(&desc_b, &desc_c);

    println!("Global Shape Distances:");
    println!("  Group A ↔ Group B: {:.6}", dist_ab);
    println!("  Group A ↔ Group C: {:.6}", dist_ac);
    println!("  Group B ↔ Group C: {:.6}", dist_bc);
    println!();

    // EMD approximation
    let emd_ab = ShapeDistance::emd_approx(&desc_a, &desc_b);
    let emd_ac = ShapeDistance::emd_approx(&desc_a, &desc_c);
    println!("EMD Approximation:");
    println!("  Group A ↔ Group B: {:.6}", emd_ab);
    println!("  Group A ↔ Group C: {:.6}", emd_ac);
    println!();

    println!("Conclusion:");
    println!("  A and C (both stable clusters) are closer than A and B.");
    println!("  dist(A,C) = {:.4} < dist(A,B) = {:.4}", dist_ac, dist_ab);
}

fn print_classification(name: &str, cls: &spectral_shape::ShapeClassification) {
    let pattern_str = match cls.pattern {
        BehaviorPattern::Stable => "Stable",
        BehaviorPattern::Oscillating => "Oscillating",
        BehaviorPattern::Chaotic => "Chaotic",
        BehaviorPattern::Transitioning => "Transitioning",
    };
    println!(
        "  {}: {} (confidence: {:.2}, spectral_gap: {:.4}, connectivity: {:.4})",
        name, pattern_str, cls.confidence, cls.spectral_gap, cls.algebraic_connectivity
    );
}
