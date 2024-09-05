use super::{tsne, DistbhtSNEf32};
use wasm_bindgen::JsValue;

extern crate wasm_bindgen_test;
use crate::hyperparameters::Hyperparameters;
use hora::core::ann_index::ANNIndex;
use num_traits::{AsPrimitive, Float};
use rayon::prelude::*;
use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);
const D: usize = 4;

const NO_DIMS: u8 = 2;

/// Computes the pairwise distances between the samples in the input data set
/// taking only the close `k` neighbors.
fn calculate_distances<T>(
    data: &[T],
    cols: &usize,
    n_neighbors: &usize,
    n_samples: usize,
) -> (Vec<tsne::Aligned<T>>, Vec<tsne::Aligned<usize>>)
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
    let data: Vec<&[T]> = data.chunks_exact(*cols).collect();
    let mut distances: Vec<tsne::Aligned<T>> = vec![T::zero().into(); pairwise_entries];
    // This vector is used to keep track of the indexes for each nearest neighbors of each
    // sample. There's a one to one correspondence between the elements of p_columns
    // an the elements of p_values: for each row i of length n_neighbors of such matrices it
    // holds that p_columns[i][j] corresponds to the index sample which contributes
    // to p_values[i][j]. This vector is freed inside `symmetrize_sparse_matrix`.
    let mut indices: Vec<tsne::Aligned<usize>> = vec![0.into(); pairwise_entries];
    // Computes sparse input similarities using a vantage point tree.
    {
        let metric_f = |a: &&[T], b: &&[T]| {
            a.iter()
                .zip(b.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<T>()
                .sqrt()
        };
        // Build ball tree on data set. The tree is freed at the end of the scope.
        let tree = tsne::VPTree::new(&data, &metric_f);
        distances
            .par_chunks_mut(*n_neighbors)
            .zip(indices.par_chunks_mut(*n_neighbors))
            .zip(data.par_iter())
            .enumerate()
            .for_each(|(index, ((distances_row, indices_row), sample))| {
                // Writes the indices and the distances of the nearest neighbors of sample.
                tree.search(
                    sample,
                    index,
                    n_neighbors + 1, // The first NN is sample itself.
                    indices_row,
                    distances_row,
                    &metric_f,
                );
                debug_assert!(!indices_row.iter().any(|i| i.0 == index));
            });
    }
    (distances, indices)
}

/// Using approximate nearest neighbors to calculate the distances between the samples.
fn calculate_distances_approx<T>(
    data: &[T],
    cols: &usize,
    n_neighbors: &usize,
    n_samples: usize,
) -> (Vec<T>, Vec<usize>)
where
    T: Send
        + Sync
        + Float
        + Sum
        + DivAssign
        + MulAssign
        + AddAssign
        + SubAssign
        + AsPrimitive<usize>
        + hora::core::node::FloatElement,
{
    // Number of entries in pairwise measures matrices.
    let pairwise_entries = n_samples * n_neighbors;
    let data: Vec<&[T]> = data.chunks_exact(*cols).collect();
    let mut distances: Vec<T> = vec![T::zero(); pairwise_entries];
    // This vector is used to keep track of the indexes for each nearest neighbors of each
    // sample. There's a one to one correspondence between the elements of p_columns
    // an the elements of p_values: for each row i of length n_neighbors of such matrices it
    // holds that p_columns[i][j] corresponds to the index sample which contributes
    // to p_values[i][j]. This vector is freed inside `symmetrize_sparse_matrix`.
    let mut indices: Vec<usize> = vec![0; pairwise_entries];
    // init index
    let mut index = hora::index::hnsw_idx::HNSWIndex::<T, usize>::new(
        *cols,
        &hora::index::hnsw_params::HNSWParams::<T>::default(),
    );
    for (i, sample) in data.iter().enumerate() {
        // add sample to index
        index.add(sample, i).unwrap();
    }
    index.build(hora::core::metrics::Metric::Euclidean).unwrap();

    distances
        .par_chunks_mut(*n_neighbors)
        .zip(indices.par_chunks_mut(*n_neighbors))
        .zip(data.par_iter())
        .enumerate()
        .for_each(|(i, ((distances_row, indices_row), sample))| {
            // Writes the indices and the distances of the nearest neighbors of sample.
            let neighbors = index
                .search_nodes(sample, n_neighbors + 1)
                .into_iter()
                .filter_map(|(node, dist)| {
                    let idx = node.idx().unwrap();
                    if idx != i {
                        Some((idx, dist))
                    } else {
                        None
                    }
                });

            neighbors
                .into_iter()
                .zip(distances_row.iter_mut())
                .zip(indices_row.iter_mut())
                .for_each(|((neighbor, distance), index)| {
                    *distance = neighbor.1;
                    *index = neighbor.0;
                });
        });
    (distances, indices)
}

#[wasm_bindgen_test]
#[cfg(not(tarpaulin_include))]
fn barnes_hut_tsne() {
    // TODO: implementing I/O for testing with iris.csv and maybe a pkg feature

    // for now this don't work, wasm doesn't support I/O out of the box
    //   let data: Vec<f32> =
    //      crate::load_csv("iris.csv", true, Some(&[4]), |float| float.parse().unwrap()).unwrap();

    // this is ugly but i kinda like it
    let data: Vec<f32> = vec![
        5.1, 3.5, 1.4, 0.2, 4.9, 3., 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5., 3.6,
        1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4, 0.3, 5., 3.4, 1.5, 0.2, 4.4, 2.9, 1.4, 0.2,
        4.9, 3.1, 1.5, 0.1, 5.4, 3.7, 1.5, 0.2, 4.8, 3.4, 1.6, 0.2, 4.8, 3., 1.4, 0.1, 4.3, 3.,
        1.1, 0.1, 5.8, 4., 1.2, 0.2, 5.7, 4.4, 1.5, 0.4, 5.4, 3.9, 1.3, 0.4, 5.1, 3.5, 1.4, 0.3,
        5.7, 3.8, 1.7, 0.3, 5.1, 3.8, 1.5, 0.3, 5.4, 3.4, 1.7, 0.2, 5.1, 3.7, 1.5, 0.4, 4.6, 3.6,
        1., 0.2, 5.1, 3.3, 1.7, 0.5, 4.8, 3.4, 1.9, 0.2, 5., 3., 1.6, 0.2, 5., 3.4, 1.6, 0.4, 5.2,
        3.5, 1.5, 0.2, 5.2, 3.4, 1.4, 0.2, 4.7, 3.2, 1.6, 0.2, 4.8, 3.1, 1.6, 0.2, 5.4, 3.4, 1.5,
        0.4, 5.2, 4.1, 1.5, 0.1, 5.5, 4.2, 1.4, 0.2, 4.9, 3.1, 1.5, 0.2, 5., 3.2, 1.2, 0.2, 5.5,
        3.5, 1.3, 0.2, 4.9, 3.6, 1.4, 0.1, 4.4, 3., 1.3, 0.2, 5.1, 3.4, 1.5, 0.2, 5., 3.5, 1.3,
        0.3, 4.5, 2.3, 1.3, 0.3, 4.4, 3.2, 1.3, 0.2, 5., 3.5, 1.6, 0.6, 5.1, 3.8, 1.9, 0.4, 4.8,
        3., 1.4, 0.3, 5.1, 3.8, 1.6, 0.2, 4.6, 3.2, 1.4, 0.2, 5.3, 3.7, 1.5, 0.2, 5., 3.3, 1.4,
        0.2, 7., 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4., 1.3, 6.5,
        2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1., 6.6, 2.9, 4.6,
        1.3, 5.2, 2.7, 3.9, 1.4, 5., 2., 3.5, 1., 5.9, 3., 4.2, 1.5, 6., 2.2, 4., 1., 6.1, 2.9,
        4.7, 1.4, 5.6, 2.9, 3.6, 1.3, 6.7, 3.1, 4.4, 1.4, 5.6, 3., 4.5, 1.5, 5.8, 2.7, 4.1, 1.,
        6.2, 2.2, 4.5, 1.5, 5.6, 2.5, 3.9, 1.1, 5.9, 3.2, 4.8, 1.8, 6.1, 2.8, 4., 1.3, 6.3, 2.5,
        4.9, 1.5, 6.1, 2.8, 4.7, 1.2, 6.4, 2.9, 4.3, 1.3, 6.6, 3., 4.4, 1.4, 6.8, 2.8, 4.8, 1.4,
        6.7, 3., 5., 1.7, 6., 2.9, 4.5, 1.5, 5.7, 2.6, 3.5, 1., 5.5, 2.4, 3.8, 1.1, 5.5, 2.4, 3.7,
        1., 5.8, 2.7, 3.9, 1.2, 6., 2.7, 5.1, 1.6, 5.4, 3., 4.5, 1.5, 6., 3.4, 4.5, 1.6, 6.7, 3.1,
        4.7, 1.5, 6.3, 2.3, 4.4, 1.3, 5.6, 3., 4.1, 1.3, 5.5, 2.5, 4., 1.3, 5.5, 2.6, 4.4, 1.2,
        6.1, 3., 4.6, 1.4, 5.8, 2.6, 4., 1.2, 5., 2.3, 3.3, 1., 5.6, 2.7, 4.2, 1.3, 5.7, 3., 4.2,
        1.2, 5.7, 2.9, 4.2, 1.3, 6.2, 2.9, 4.3, 1.3, 5.1, 2.5, 3., 1.1, 5.7, 2.8, 4.1, 1.3, 6.3,
        3.3, 6., 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3., 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5, 3., 5.8, 2.2,
        7.6, 3., 6.6, 2.1, 4.9, 2.5, 4.5, 1.7, 7.3, 2.9, 6.3, 1.8, 6.7, 2.5, 5.8, 1.8, 7.2, 3.6,
        6.1, 2.5, 6.5, 3.2, 5.1, 2., 6.4, 2.7, 5.3, 1.9, 6.8, 3., 5.5, 2.1, 5.7, 2.5, 5., 2., 5.8,
        2.8, 5.1, 2.4, 6.4, 3.2, 5.3, 2.3, 6.5, 3., 5.5, 1.8, 7.7, 3.8, 6.7, 2.2, 7.7, 2.6, 6.9,
        2.3, 6., 2.2, 5., 1.5, 6.9, 3.2, 5.7, 2.3, 5.6, 2.8, 4.9, 2., 7.7, 2.8, 6.7, 2., 6.3, 2.7,
        4.9, 1.8, 6.7, 3.3, 5.7, 2.1, 7.2, 3.2, 6., 1.8, 6.2, 2.8, 4.8, 1.8, 6.1, 3., 4.9, 1.8,
        6.4, 2.8, 5.6, 2.1, 7.2, 3., 5.8, 1.6, 7.4, 2.8, 6.1, 1.9, 7.9, 3.8, 6.4, 2., 6.4, 2.8,
        5.6, 2.2, 6.3, 2.8, 5.1, 1.5, 6.1, 2.6, 5.6, 1.4, 7.7, 3., 6.1, 2.3, 6.3, 3.4, 5.6, 2.4,
        6.4, 3.1, 5.5, 1.8, 6., 3., 4.8, 1.8, 6.9, 3.1, 5.4, 2.1, 6.7, 3.1, 5.6, 2.4, 6.9, 3.1,
        5.1, 2.3, 5.8, 2.7, 5.1, 1.9, 6.8, 3.2, 5.9, 2.3, 6.7, 3.3, 5.7, 2.5, 6.7, 3., 5.2, 2.3,
        6.3, 2.5, 5., 1.9, 6.5, 3., 5.2, 2., 6.2, 3.4, 5.4, 2.3, 5.9, 3., 5.1, 1.8,
    ];

    let opt: Hyperparameters<f32> = Hyperparameters {
        learning_rate: 200.0,
        momentum: 0.5,
        final_momentum: 0.8,
        momentum_switch_epoch: 250,
        stop_lying_epoch: 250,
        theta: 0.5,
        embedding_dim: 2,
        perplexity: 20.0,
    };

    let n_samples = data.len() / D;
    let n_neighbors = (3.0 * opt.perplexity) as usize;
    let (distances, indices) = calculate_distances_approx(&data, &D, &n_neighbors, n_samples);

    let opt_js: JsValue = serde_wasm_bindgen::to_value(&opt).unwrap();
    let mut tsne: DistbhtSNEf32;

    match DistbhtSNEf32::new(&distances, &indices, n_samples, n_neighbors, opt_js) {
        Ok(t) => tsne = t,
        Err(e) => panic!("Error: {:?}", e),
    }

    for _x in 0..1000 {
        tsne.step(1);
    }
    let embedding_js = tsne.get_embedding();
    let length = n_samples * NO_DIMS as usize;
    let flattened_array: Vec<f32> =
        unsafe { Vec::from_raw_parts(embedding_js as *mut f32, length, length) };
    let points: Vec<_> = flattened_array.chunks(NO_DIMS as usize).collect();

    assert_eq!(points.len(), n_samples);

    assert!(
        tsne::evaluate_error_approximately(
            &tsne.tsne_encoder.p_rows,
            &tsne.tsne_encoder.p_columns,
            &tsne.tsne_encoder.p_values,
            &tsne.tsne_encoder.y,
            &n_samples,
            &(tsne.tsne_encoder.embedding_dim),
            &tsne.tsne_encoder.theta,
        ) < 5.0
    );
}
