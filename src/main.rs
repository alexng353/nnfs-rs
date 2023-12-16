use anyhow::anyhow;
use anyhow::Result;
use std::vec;

#[tokio::main]
async fn main() -> Result<()> {
    let now = tokio::time::Instant::now();

    let inputs = vec![1.0, 2.0, 3.0, 2.5];

    let weights = vec![
        vec![0.2, 0.8, -0.5, 1.0],
        vec![0.5, -0.91, 0.26, -0.5],
        vec![-0.26, -0.27, 0.17, 0.87],
    ];
    let biases = vec![2.0, 3.0, 0.5];

    let layer = Layer::new(&weights, &biases);

    let outputs = layer.feedforward(&inputs)?;

    println!("outputs: {:?}", outputs);
    println!("elapsed: {:?}", now.elapsed());

    Ok(())
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new<W: AsRef<Vec<Vec<f64>>>, B: AsRef<Vec<f64>>>(weights: W, biases: B) -> Self {
        let weights = weights.as_ref();
        let biases = biases.as_ref();

        let neurons = weights
            .iter()
            .zip(biases.iter())
            .map(|(w, b)| Neuron::new(w, *b))
            .collect::<Vec<Neuron>>();

        println!("num of neurons: {}", neurons.len());

        Self { neurons }
    }

    fn feedforward<T: AsRef<Vec<f64>>>(&self, inputs: T) -> Result<Vec<f64>> {
        let inputs = inputs.as_ref();

        let outputs = self
            .neurons
            .iter()
            .map(|n| n.feedforward(inputs))
            .collect::<Result<Vec<f64>>>()?;

        Ok(outputs)
    }
}

struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    fn new<T: AsRef<Vec<f64>>, U: Into<f64>>(weights: T, bias: U) -> Self {
        Self {
            weights: weights.as_ref().to_vec(),
            bias: bias.into(),
        }
    }

    fn feedforward<T: AsRef<Vec<f64>>>(&self, inputs_: T) -> Result<f64> {
        let inputs = inputs_.as_ref();

        if self.weights.len() != inputs.len() {
            return Err(anyhow!("The length of weights and inputs must be same."));
        }

        let outputs = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()
            + self.bias;

        Ok(outputs)
    }
}
