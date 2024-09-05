use crate::tsne;

use crate::hyperparameters::Hyperparameters;
pub(crate) use num_traits::{cast::AsPrimitive, Float};
use rayon::prelude::*;
use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};
use wasm_bindgen::JsValue;

#[allow(non_camel_case_types)]
pub struct tsne_encoder<T>
where
    T: Send + Sync + Float + Sum + DivAssign + MulAssign + AddAssign + SubAssign,
{
    pub(crate) theta: T,
    learning_rate: T,
    epochs: usize,
    momentum: T,
    final_momentum: T,
    momentum_switch_epoch: usize,
    stop_lying_epoch: usize,
    pub(crate) embedding_dim: usize,
    perplexity: T,
    pub(crate) p_values: Vec<tsne::Aligned<T>>,
    pub(crate) p_rows: Vec<usize>,
    pub(crate) p_columns: Vec<usize>,
    pub(crate) y: Vec<tsne::Aligned<T>>,
    dy: Vec<tsne::Aligned<T>>,
    uy: Vec<tsne::Aligned<T>>,
    gains: Vec<tsne::Aligned<T>>,
    positive_forces: Vec<tsne::Aligned<T>>,
    negative_forces: Vec<tsne::Aligned<T>>,
    forces_buffer: Vec<tsne::Aligned<T>>,
    q_sums: Vec<tsne::Aligned<T>>,
    means: Vec<T>,
    n_samples: usize,
    embeddings: Vec<T>,
}

impl<T> tsne_encoder<T>
where
    T: Float
        + Send
        + Sync
        + AsPrimitive<usize>
        + Sum
        + DivAssign
        + AddAssign
        + MulAssign
        + SubAssign,
{
    pub fn new(
        p_values: Vec<tsne::Aligned<T>>,
        n_samples: usize,
        hyperparameters: Hyperparameters<T>,
    ) -> Self {
        assert!(
            hyperparameters.theta > T::from(0.0).unwrap(),
            "error: theta value must be greater than 0.0.
            A value of 0.0 corresponds to using the exact version of the algorithm."
        );

        Self {
            theta: hyperparameters.theta,
            learning_rate: hyperparameters.learning_rate,
            epochs: 0,
            momentum: hyperparameters.momentum,
            final_momentum: hyperparameters.final_momentum,
            momentum_switch_epoch: hyperparameters.momentum_switch_epoch,
            stop_lying_epoch: hyperparameters.stop_lying_epoch,
            embedding_dim: hyperparameters.embedding_dim,
            perplexity: hyperparameters.perplexity,
            p_values,
            p_rows: Vec::new(),
            p_columns: Vec::new(),
            y: Vec::new(),
            dy: Vec::new(),
            uy: Vec::new(),
            gains: Vec::new(),
            positive_forces: Vec::new(),
            negative_forces: Vec::new(),
            forces_buffer: Vec::new(),
            q_sums: Vec::new(),
            means: Vec::new(),
            n_samples,
            embeddings: Vec::new(),
        }
    }

    /// Performs a parallel Barnes-Hut approximation of the t-SNE algorithm.
    ///
    /// # Arguments
    ///
    /// * `theta` - determines the accuracy of the approximation. Must be **strictly greater than
    /// 0.0**. Large values for θ increase the speed of the algorithm but decrease its accuracy.
    /// For small values of θ it is less probable that a cell in the space partitioning tree will
    /// be treated as a single point. For θ equal to 0.0 the method degenerates in the exact
    /// version.
    ///
    /// * `metric_f` - metric function.
    ///
    ///
    /// **Do note that** `metric_f` **must be a metric distance**, i.e. it must
    /// satisfy the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality).
    pub fn barnes_hut_data(
        &mut self,
        p_columns: Vec<tsne::Aligned<usize>>,
    ) -> std::result::Result<(), JsValue> {
        // Checks that the supplied perplexity is suitable for the number of samples at hand.
        tsne::check_perplexity(&self.perplexity, &self.n_samples)?;

        let embedding_dim = self.embedding_dim;
        // Number of points to consider when approximating the conditional distribution P.
        let n_neighbors: usize = (T::from(3.0).unwrap() * self.perplexity).as_();
        // NUmber of entries in gradient and gains matrices.
        let grad_entries = self.n_samples * embedding_dim;

        // resize embeddings
        self.embeddings.resize(grad_entries, T::zero());

        // Prepare buffers
        tsne::prepare_buffers(
            &mut self.y,
            &mut self.dy,
            &mut self.uy,
            &mut self.gains,
            &grad_entries,
        );

        // Symmetrize sparse P matrix.
        tsne::symmetrize_sparse_matrix(
            &mut self.p_rows,
            &mut self.p_columns,
            p_columns,
            &mut self.p_values,
            self.n_samples,
            &n_neighbors,
        );

        // Normalize P values.
        tsne::normalize_p_values(&mut self.p_values);

        // Initialize solution randomly.
        tsne::random_init(&mut self.y);

        // Prepares buffers for Barnes-Hut algorithm.
        self.positive_forces = vec![T::zero().into(); grad_entries];
        self.negative_forces = vec![T::zero().into(); grad_entries];
        self.forces_buffer = vec![T::zero().into(); grad_entries];
        self.q_sums = vec![T::zero().into(); self.n_samples];

        // Vector used to store the mean values for each embedding dimension. It's used
        // to make the solution zero mean.
        self.means = vec![T::zero(); embedding_dim];
        self.epochs = 0;

        Ok(())
    }

    // Main Training loop.
    pub fn run(&mut self, epochs: usize) {
        let embedding_dim = self.embedding_dim;

        for _epoch in 0..epochs {
            self.epochs += 1;
            {
                // Construct space partitioning tree on current embedding.
                let tree = tsne::SPTree::new(&embedding_dim, &self.y, &self.n_samples);
                // Check if the SPTree is correct.
                debug_assert!(tree.is_correct(), "error: SPTree is not correct.");

                // Computes forces using the Barnes-Hut algorithm in parallel.
                // Each chunk of positive_forces and negative_forces is associated to a distinct
                // embedded sample in y. As a consequence of this the computation can be done in
                // parallel.
                self.positive_forces
                    .par_chunks_mut(embedding_dim)
                    .zip(self.negative_forces.par_chunks_mut(embedding_dim))
                    .zip(self.forces_buffer.par_chunks_mut(embedding_dim))
                    .zip(self.q_sums.par_iter_mut())
                    .zip(self.y.par_chunks(embedding_dim))
                    .enumerate()
                    .for_each(
                        |(
                            index,
                            (
                                (
                                    ((positive_forces_row, negative_forces_row), forces_buffer_row),
                                    q_sum,
                                ),
                                sample,
                            ),
                        )| {
                            tree.compute_edge_forces(
                                index,
                                sample,
                                &self.p_rows,
                                &self.p_columns,
                                &self.p_values,
                                forces_buffer_row,
                                positive_forces_row,
                            );
                            tree.compute_non_edge_forces(
                                index,
                                &self.theta,
                                negative_forces_row,
                                forces_buffer_row,
                                q_sum,
                            );
                        },
                    );
            }

            // Compute final Barnes-Hut t-SNE gradient approximation.
            // Reduces partial sums of Q distribution.
            let q_sum: T = self.q_sums.par_iter_mut().map(|sum| sum.0).sum();
            self.dy
                .par_iter_mut()
                .zip(self.positive_forces.par_iter_mut())
                .zip(self.negative_forces.par_iter_mut())
                .for_each(|((grad, pf), nf)| {
                    grad.0 = pf.0 - (nf.0 / q_sum);
                    pf.0 = T::zero();
                    nf.0 = T::zero();
                });
            // Zeroes Q-sums.
            self.q_sums.par_iter_mut().for_each(|sum| sum.0 = T::zero());

            // Updates the embedding in parallel with gradient descent.
            tsne::update_solution(
                &mut self.y,
                &self.dy,
                &mut self.uy,
                &mut self.gains,
                &self.learning_rate,
                &self.momentum,
            );

            // Make solution zero-mean.
            tsne::zero_mean(
                &mut self.means,
                &mut self.y,
                &self.n_samples,
                &self.embedding_dim,
            );

            // Stop lying about the P-values if the time is right.
            if self.epochs == self.stop_lying_epoch {
                tsne::stop_lying(&mut self.p_values);
            }

            // Switches momentum if the time is right.
            if self.epochs == self.momentum_switch_epoch {
                self.momentum = self.final_momentum;
            }
        }
    }

    pub fn embeddings(&mut self) -> *const T {
        self.embeddings
            .par_iter_mut()
            .zip(self.y.par_iter())
            .for_each(|(emb, y)| *emb = y.0);
        self.embeddings.as_ptr()
    }
}
