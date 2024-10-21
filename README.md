# wasm-dist-bhtsne

This is a WebAssembly (WASM) port of the modified Barnes-Hut t-SNE algorithm that works with pre-computed distance vector. The original Rust implementation can be found [here](https://github.com/frjnn/bhtsne). Make sure `wasm-pack` is installed on your system. If not, you can install it by running:

```bash
cargo install wasm-pack
```

To build the WASM module, run:

```bash
wasm-pack build --target web --features parallel
```

This will generate a `pkg` directory containing the WASM module and a JavaScript wrapper.

## Installation
Install the [wasm-dist-bhtsne](https://www.npmjs.com/package/wasm-dist-bhtsne) package from npm:

```bash
npm i wasm-dist-bhtsne
```

## Example
```javascript
import { threads } from 'wasm-feature-detect';
import init, { initThreadPool, DistbhtSNEf64 } from 'wasm-dist-bhtsne';

// Final embedding dimension
const outputDim = 2;
// example data format
const exampleData = {
    distances: [1.0, 2.0, 3.0, 4.0, ..., 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    indices: [0, 11, 23, 43, ..., 4, 5, 64, 7, 8, 9],
    n_samples: 1000
};

(async {
    const { memory } = await init();
    if (await threads()) {
        console.log("Browser supports threads");
        initThreadPool(navigator.hardwareConcurrency);
    }
    else {
        console.log("Browser does not support threads");
        return;
    }

    // set hyperparameters
    const opt = {
        learning_rate: 150.0,
        perplexity: 30.0,
        theta: 0.5,
    };

    try {
        const tsneEncoder = new DistbhtSNEf64(
            exampleData.distances, // precomputed distance matrix
            exampleData.indices, // precomputed indices matrix
            exampleData.n_samples, // number of vectors
            opt
        );

        // run the algorithm for 1000 iterations
        for (let i = 0; i < 1000; i++) {
            tsneEncoder.step(1);
        }

        // get the embedding
        const embedding = tsneEncoder.get_embedding();
        const result = new Float64Array(
            memory.buffer, 
            embedding, 
            exampleData.n_samples * outputDim
        );
    }
    catch (e) {
        console.error(e);
    }
})();
```

## Features
- Harnesses multi-threading capabilities through [wasm-bindgen-rayon](https://github.com/RReverser/wasm-bindgen-rayon).
- Allows passing t-SNE hyperparameters through a JavaScript object, where you only need to include the parameters you want to change from the defaults. If you don't specify any, default values are used.
- Supports running the algorithm in iterations, enabling progressive refinement of the embedding.
- Supports both Float32Array and Float64Array for data input.

## Requirements
To use the multithreading feature, you need to enable `SharedArrayBuffer` on the Web. As stated in the [wasm-bindgen-rayon readme](https://github.com/RReverser/wasm-bindgen-rayon/blob/main/README.md):

In order to use `SharedArrayBuffer` on the Web, you need to enable [cross-origin isolation policies](https://web.dev/articles/coop-coep). Check out the linked article for details.

## Hyperparameters
Here is a list of hyperparameters that can be set in the JavaScript object, along with their default values and descriptions:

- `learning_rate` (default: `200.0`): controls the step size during the optimization.
- `momentum` (default: `0.5`): helps accelerate gradients vectors in the right directions, thus leading to faster converging.
- `final_momentum` (default: `0.8`): momentum value used after a certain number of iterations.
- `momentum_switch_epoch` (default: `250`): the epoch after which the algorithm switches to final_momentum for the map update.
- `stop_lying_epoch` (default: `250`): the epoch after which the P distribution values become true. For epochs < stop_lying_epoch, the values of the P distribution are multiplied by a factor equal to `12.0`.
- `theta` (default: `0.5`): Determines the accuracy of the approximation. Larger values increase the speed but decrease accuracy. Must be strictly greater than `0.0`.
- `embedding_dim` (default: `2`): the dimensionality of the embedding space.
- `perplexity` (default: `20.0`): the perplexity value. It determines the balance between local and global aspects of the data. A good value lies between `5.0` and `50.0`.

## Acknowledgements
Here are the list of people whom I would like to express my gratitude for helping me with this project:
- [Andrey Vasnetsov (@generall)](https://github.com/generall)
- [Cristian Baiunco (@Lv-291)](https://github.com/Lv-291)
- [Jan (@frjnn)](https://github.com/frjnn)
