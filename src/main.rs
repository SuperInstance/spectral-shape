use spectral_shape::laplacian::LaplacianMatrix;

fn main() {
    let adj = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
    let lap = LaplacianMatrix::from_adjacency(&adj);
    println!("Laplacian of K2: {:?}", lap.matrix);
}
