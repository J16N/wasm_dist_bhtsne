mod hyperparameters;
mod tsne;
mod utils;

use crate::hyperparameters::Hyperparameters;
use crate::utils::set_panic_hook;
use num_traits::{AsPrimitive, Float};
use rayon::prelude::*;
use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;

mod algorithm;
#[cfg(test)]
mod test;
use crate::algorithm::tsne_encoder;

#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// Computes the pairwise affinity matrix for the input data set.
fn calculate_pairwise_affinity<T>(
    distances: &[tsne::Aligned<T>],
    perplexity: &T,
    n_neighbors: &usize,
    n_samples: usize,
) -> Vec<tsne::Aligned<T>>
where
    T: Send
        + Sync
        + Float
        + Sum
        + DivAssign
        + MulAssign
        + AddAssign
        + SubAssign
        + AsPrimitive<usize>,
{
    // Number of entries in pairwise measures matrices.
    let pairwise_entries = n_samples * n_neighbors;
    // The P distribution values are restricted to a subset of size n_neighbors for each input
    // sample.
    let mut p_values: Vec<tsne::Aligned<T>> = vec![T::zero().into(); pairwise_entries];
    // Compute the conditional distribution P using the perplexities in parallel.
    p_values
        .par_chunks_mut(*n_neighbors)
        .zip(distances.par_chunks(*n_neighbors))
        .for_each(|(p_values_row, distances_row)| {
            tsne::search_beta(p_values_row, distances_row, perplexity);
        });
    p_values
}

/// t-distributed stochastic neighbor embedding. Provides a parallel implementation of both the
/// exact version of the algorithm and the tree accelerated one leveraging space partitioning trees.
#[wasm_bindgen]
#[allow(non_camel_case_types)]
pub struct DistbhtSNEf32 {
    tsne_encoder: tsne_encoder<f32>,
}

#[wasm_bindgen]
impl DistbhtSNEf32 {
    /// Creates a new instance of the t-SNE algorithm that works with distance matrix.
    ///
    /// # Arguments
    ///
    /// `distances` - The distance matrix.
    /// `indices` - The indices of the nearest neighbors.
    /// `n_samples` - The number of vectors.
    /// `n_neighbors` - The number of nearest neighbors.
    /// `opt` - The hyperparameters.
    #[wasm_bindgen(constructor)]
    pub fn new(
        distances: &[f32],
        indices: &[usize],
        n_samples: usize,
        n_neighbors: usize,
        opt: JsValue,
    ) -> Result<DistbhtSNEf32, JsValue> {
        set_panic_hook();
        let hyperparameters: Hyperparameters<f32> = serde_wasm_bindgen::from_value(opt).unwrap();
        let distances: Vec<tsne::Aligned<f32>> =
            distances.iter().map(|el| tsne::Aligned(*el)).collect();
        let p_columns: Vec<tsne::Aligned<usize>> =
            indices.iter().map(|el| tsne::Aligned(*el)).collect();

        let p_values = calculate_pairwise_affinity(
            &distances,
            &hyperparameters.perplexity,
            &n_neighbors,
            n_samples,
        );

        let mut tsne = tsne_encoder::new(p_values, n_samples, hyperparameters);

        tsne.barnes_hut_data(p_columns)?;

        Ok(Self { tsne_encoder: tsne })
    }

    /// Performs a parallel Barnes-Hut approximation of the t-SNE algorithm.
    ///
    /// # Arguments
    ///
    /// `epochs` - the maximum number of fitting iterations. Must be positive
    pub fn step(&mut self, epochs: usize) {
        self.tsne_encoder.run(epochs);
    }

    /// Returns the embedding.
    pub fn get_embedding(&mut self) -> *const f32 {
        self.tsne_encoder.embeddings()
    }
}

#[wasm_bindgen]
#[allow(non_camel_case_types)]
pub struct DistbhtSNEf64 {
    tsne_encoder: tsne_encoder<f64>,
}

#[wasm_bindgen]
impl DistbhtSNEf64 {
    /// Creates a new instance of the t-SNE algorithm that works with distance matrix.
    ///
    /// # Arguments
    ///
    /// `distances` - The distance matrix.
    /// `indices` - The indices of the nearest neighbors.
    /// `n_samples` - The number of vectors.
    /// `n_neighbors` - The number of nearest neighbors.
    /// `opt` - The hyperparameters.
    #[wasm_bindgen(constructor)]
    pub fn new(
        distances: &[f64],
        indices: &[usize],
        n_samples: usize,
        n_neighbors: usize,
        opt: JsValue,
    ) -> Result<DistbhtSNEf64, JsValue> {
        set_panic_hook();
        let hyperparameters: Hyperparameters<f64> = serde_wasm_bindgen::from_value(opt).unwrap();
        let distances: Vec<tsne::Aligned<f64>> =
            distances.iter().map(|el| tsne::Aligned(*el)).collect();
        let p_columns: Vec<tsne::Aligned<usize>> =
            indices.iter().map(|el| tsne::Aligned(*el)).collect();

        let p_values = calculate_pairwise_affinity(
            &distances,
            &hyperparameters.perplexity,
            &n_neighbors,
            n_samples,
        );

        let mut tsne = tsne_encoder::new(p_values, n_samples, hyperparameters);

        tsne.barnes_hut_data(p_columns)?;

        Ok(Self { tsne_encoder: tsne })
    }

    /// Performs a parallel Barnes-Hut approximation of the t-SNE algorithm.
    ///
    /// # Arguments
    ///
    /// `epochs` - Sets epochs, the maximum number of fitting iterations.
    pub fn step(&mut self, epochs: usize) {
        self.tsne_encoder.run(epochs);
    }

    /// Returns the embedding.
    pub fn get_embedding(&mut self) -> *const f64 {
        self.tsne_encoder.embeddings()
    }
}
