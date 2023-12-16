use anyhow::anyhow;
use anyhow::Result;
use std::vec;

use ndarray::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let now = tokio::time::Instant::now();

    let inputs = array![1.0, 2.0, 3.0, 2.5];
    let weights = array![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ];

    let biases = array![2.0, 3.0, 0.5];

    let outputs = weights.dot(&inputs.t()) + &biases;

    println!("outputs: {:?}", outputs.to_vec());

    println!("elapsed: {:?}", now.elapsed());

    Ok(())
}
