//! Scale-space analysis of time-series behavior data.
//!
//! Demonstrates multi-scale analysis of agent activity levels over time,
//! detecting stable features and finding the characteristic scale.

use spectral_shape::{
    CharacteristicScale, GaussianScaleSpace, ScaleInvariance, ScaleSpace, ScaleSpaceExtrema,
};

fn main() {
    println!("=== Scale-Space Analysis of Agent Behavior Time Series ===\n");

    // Simulated agent activity levels over 30 time steps
    // Pattern: periodic bursts with noise
    let signal: Vec<f64> = (0..30)
        .map(|i| {
            let base = (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin();
            let noise = if i % 7 == 0 { 0.5 } else { 0.0 };
            base + noise
        })
        .collect();

    println!("Original signal (30 time steps):");
    print_signal(&signal);
    println!();

    // Build Gaussian scale-space with 8 scales
    let sigmas = GaussianScaleSpace::log_spaced_scales(8, 0.3, 5.0);
    println!(
        "Scale parameters (σ): {:?}",
        sigmas
            .iter()
            .map(|s| format!("{:.2}", s))
            .collect::<Vec<_>>()
    );
    println!();

    let ss = GaussianScaleSpace::new(&signal, &sigmas);

    // Show smoothed signals at selected scales
    println!("Smoothed signals at different scales:");
    for (idx, sigma) in [0, 2, 5, 7].iter().map(|&i| (i, ss.sigmas[i])) {
        let smoothed = ss.at_scale(idx);
        println!("  σ = {:.2}:", sigma);
        print_signal_smoothed(smoothed);
    }
    println!();

    // Detect scale-space extrema
    let extrema = ScaleSpaceExtrema::detect(&ss);
    println!(
        "Scale-space extrema detected: {} features",
        extrema.extrema.len()
    );
    for (pos, scale_idx, val) in &extrema.extrema {
        println!(
            "  Position {}, Scale σ={:.2}, Value {:.4}",
            pos, ss.sigmas[*scale_idx], val
        );
    }
    println!();

    // Find characteristic scale
    let char_scale = CharacteristicScale::find(&ss);
    println!("Characteristic scale:");
    println!("  σ = {:.4}", char_scale.sigma);
    println!("  Scale index = {}", char_scale.scale_index);
    println!("  Structure measure = {:.4}", char_scale.structure_measure);
    println!();

    // Scale-normalized derivatives
    let derivs = ScaleInvariance::normalized_derivatives(&ss);
    println!("Scale-normalized derivatives at finest scale (first 10):");
    for (i, &d) in derivs[0].iter().take(10).enumerate() {
        println!("  t={}: σ·dL/dx = {:.4}", i, d);
    }
    println!();

    // Full pipeline analysis
    let analysis = ScaleSpace::analyze(&signal, 0.3, 5.0);
    println!("Full ScaleSpace analysis:");
    println!("  Number of scales: {}", analysis.gauss.sigmas.len());
    println!("  Extrema count: {}", analysis.extrema.extrema.len());
    println!("  Characteristic σ: {:.4}", analysis.characteristic.sigma);

    println!("\nInterpretation:");
    println!("  The periodic burst pattern (period ≈ 10) is detected at");
    println!("  appropriate scales. Fine-scale noise is suppressed at larger σ.");
    println!("  The characteristic scale reveals the dominant periodicity.");
}

fn print_signal(signal: &[f64]) {
    let display: Vec<String> = signal.iter().map(|v| format!("{:6.2}", v)).collect();
    println!("  [{}]", display.join(", "));
}

fn print_signal_smoothed(signal: &[f64]) {
    let display: Vec<String> = signal.iter().map(|v| format!("{:6.2}", v)).collect();
    println!("    [{}]", display.join(", "));
}
