//! run this with: cargo run --example simple_example  -
use rand::distributions::{Distribution, Uniform};
use std::cmp::max;
use std::time::Instant;

fn vec_factory(m: u32, n: u32, min: i64, max: i64) -> Vec<Vec<i64>>{
    let mut rng = rand::thread_rng();
    let distr = Uniform::from(min..=max);

    let random_vec: Vec<Vec<i64>> = (0..m).map(
            |_| { 
                (0..n).map(
                    |_| distr.sample(&mut rng)).collect()
            }
        ).collect();

    random_vec
}

fn detect_dim(a: &Vec<Vec<i64>>) -> (u32, u32) {
    let m: u32 = a.len() as u32;
    let n: u32 = a[0].len() as u32;
    (m,n)
}

fn hp_on_cpu(a: &Vec<Vec<i64>>, b: &Vec<Vec<i64>>) -> Vec<Vec<i64>>{

    //get all dimensions
    let (a_m, a_n) = detect_dim(a);
    let (b_m, b_n) = detect_dim(b);
    let max_m: usize = max(a_m as usize, b_m as usize);
    let max_n: usize = max(a_n as usize, b_n as usize);

    //define c, expand to max size
    let mut c: Vec<Vec<i64>> = vec![];
    //resize n dimension
    for m in &mut c {
        m.resize(max_n, 0i64);
    }
    //resize m dimension
    c.resize(max_m, vec![0i64; max_n]);

    //add the two matrices together and save in c
    for i in 0..a.len() {
       for j in 0..a[0].len() {
            let zero_vec = vec![0i64; max_n];
            let zero = 0;
            let a_i = a.get(i).unwrap_or(&zero_vec);
            let a_ij = a_i.get(j).unwrap_or(&zero);
            let b_i = b.get(i).unwrap_or(&zero_vec);
            let b_ij = b_i.get(j).unwrap_or(&zero);
            c[i][j] = a_ij + b_ij;
        }
    }

    c // return sum
}

fn main() {
    println!("Hello, world!");
    //define our vectors
    let a = vec_factory(2, 2, 0, 10);
    let b = vec_factory(2, 2, 0, 10);
    println!("a = {:?}", a);
    println!("b = {:?}", b);

    // run hadamard product on the cpu and time it
    let start_time_cpu = Instant::now();
    let c  = hp_on_cpu(&a, &b);
    let duration_cpu = start_time_cpu.elapsed();
    println!("c = {:?}", c);
    println!("cpu time: {:?}", duration_cpu);

    // run hadamard product on the gpu and time it
    let start_time_gpu = Instant::now();
    #[cfg(feature = "cuda")]
    hadamard_product_gpu::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
    // #[cfg(feature = "wgpu")]
    // hadamard_product_gpu::launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
    let duration_gpu = start_time_gpu.elapsed();
}
