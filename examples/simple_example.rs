//! run this with: cargo run --example simple_example --features="cuda"
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

    //define c
    let mut c: Vec<Vec<i64>> = vec![vec![0i64; max_n]; max_m];

    //add the two matrices together and save in c
    for i in 0..max_m {
       for j in 0..max_n {
            let a_ij = match a.get(i){
                Some(a_i) => {
                    match a_i.get(j) {
                        Some(a_ij) => *a_ij,
                        None => 0
                    }
                },
                None => 0
            };
            let b_ij = match b.get(i){
                Some(b_i) => {
                    match b_i.get(j) {
                        Some(b_ij) => *b_ij,
                        None => 0
                    }
                },
                None => 0
            };
            c[i][j] = a_ij + b_ij;
        }
    }

    c // return sum
}

fn main() {
    //define our vectors
    let a = vec_factory(34, 32, 0, 10);
    let b = vec_factory(54, 32, 0, 10);

    println!("random matrices: ");
    println!("a = {:?}", a);
    println!("b = {:?}", b);

    // run hadamard product on the cpu and time it
    println!("");
    println!("running on the cpu...");
    let start_time_cpu = Instant::now();
    let c_cpu  = hp_on_cpu(&a, &b);
    let duration_cpu = start_time_cpu.elapsed();
    println!("c = {:?}", c_cpu);
    println!("cpu time: {:?}", duration_cpu);

    // run hadamard product on the gpu and time it
    println!("");
    println!("running on the gpu...");
    let start_time_gpu = Instant::now();
    #[cfg(feature = "cuda")]
    let c_gpu = hadamard_product_gpu::launch_hp::<cubecl::cuda::CudaRuntime>(
        &Default::default(),
        &a,
        &b
    );
    let duration_gpu = start_time_gpu.elapsed();
    println!("c = {:?}", c_gpu);
    println!("gpu time: {:?}", duration_gpu);

    assert!(c_cpu == c_gpu, "the two matrices are not the same!!")
}
