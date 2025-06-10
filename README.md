# Hadamard Product On The GPU

### Description
- This simple library exposes a gpu kernel to perform hadamard products (primarily for CUDA, though WGPU could be used as well)

### How to use

Clone the repo and add it to your Cargo.toml:
```toml
hadamard_product_gpu = { path = "../path/to/cloned/hadamard_product_gpu"}
```

```rust
use hadamard_product_gpu::launch_hp;

//example

let a: &Vec<Vec<i64>> = vec![vec![1, 2], vec![3, 4]];
let b: &Vec<Vec<i64>> = vec![vec![5, 6], vec![7, 8]];

#[cfg(feature = "cuda")]
let c = hadamard_product_gpu::launch_hp::<cubecl::cuda::CudaRuntime>(
    &Default::default(),
    &a,
    &b
);
// c should be [[6, 8], [10, 12]]
```
- see the simple example in the example folder for more in depth example
