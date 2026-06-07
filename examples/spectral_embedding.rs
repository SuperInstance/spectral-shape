//! Spectral embedding of an agent communication graph.
//!
//! Demonstrates how to construct a graph from agent communication patterns
//! and embed it into 2D using spectral methods (Laplacian eigenmaps).

use spectral_shape::{
    EmbeddingDistance, KNearestNeighbors, LaplacianMatrix, MapGraphPoints, SpectralEmbedding,
};

fn main() {
    // Agent communication graph: 5 agents
    // Agent 0 communicates heavily with 1 and 2 (team A)
    // Agent 3 communicates heavily with 4 (team B)
    // Agent 2 bridges teams A and B with light communication to 3
    let adjacency = vec![
        vec![0.0, 0.9, 0.8, 0.0, 0.0],
        vec![0.9, 0.0, 0.7, 0.0, 0.0],
        vec![0.8, 0.7, 0.0, 0.3, 0.0],
        vec![0.0, 0.0, 0.3, 0.0, 0.9],
        vec![0.0, 0.0, 0.0, 0.9, 0.0],
    ];

    println!("=== Spectral Embedding of Agent Communication Graph ===\n");

    // Build Laplacian
    let lap = LaplacianMatrix::from_adjacency(&adjacency);
    println!("Laplacian matrix:");
    for row in &lap.matrix {
        println!("  {:?}", row);
    }
    println!();

    // Compute spectral embedding in 2D
    let embedding = SpectralEmbedding::from_adjacency(&adjacency, 2);
    println!("Spectral embedding (2D):");
    for (i, point) in embedding.points.iter().enumerate() {
        println!("  Agent {i}: ({:.4}, {:.4})", point[0], point[1]);
    }
    println!();

    // Map graph points for easy access
    let mapper = MapGraphPoints::new(&adjacency, 2);
    println!("Agent 0 embedding: {:?}", mapper.point(0));
    println!("Agent 3 embedding: {:?}", mapper.point(3));
    println!();

    // Distances in spectral space
    let d_01 = EmbeddingDistance::euclidean(&embedding.points[0], &embedding.points[1]);
    let d_04 = EmbeddingDistance::euclidean(&embedding.points[0], &embedding.points[4]);
    println!(
        "Spectral distance Agent 0 ↔ Agent 1: {:.4} (same team)",
        d_01
    );
    println!(
        "Spectral distance Agent 0 ↔ Agent 4: {:.4} (different teams)",
        d_04
    );
    println!();

    // k-nearest neighbors
    let knn = KNearestNeighbors::from_embedding(&embedding, 2);
    println!("2-Nearest Neighbors in spectral space:");
    for i in 0..5 {
        let neighbors: Vec<String> = knn
            .neighbors_of(i)
            .iter()
            .map(|(j, d)| format!("Agent {j} (d={d:.4})"))
            .collect();
        println!("  Agent {i}: {}", neighbors.join(", "));
    }

    println!("\nNotice: Team A agents (0,1,2) cluster together");
    println!("and Team B agents (3,4) cluster together in spectral space.");
}
