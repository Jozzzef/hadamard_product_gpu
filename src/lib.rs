use cubecl::prelude::*;
use std::cmp::max;

fn matrix_dimensions(matrix: &Vec<Vec<i64>>) -> (u32, u32){
    (matrix.len() as u32, matrix[0].len() as u32)
}

fn vec_to_flat_u8_vec(matrix: &Vec<Vec<i64>>) -> Vec<u8> {
    matrix.iter()
        .flatten()                    
        .flat_map(|&x| x.to_ne_bytes())  
        .collect()
}

#[cube(launch)]
fn hadamard_prod(
        a: &Array<Line<i64>>, 
        b: &Array<Line<i64>>, 
        c: &mut Array<Line<i64>>
) {
    let a_val = if ABSOLUTE_POS < a.len() { a[ABSOLUTE_POS] } else { Line::new(0) };
    let b_val = if ABSOLUTE_POS < b.len() { b[ABSOLUTE_POS] } else { Line::new(0) };
    c[ABSOLUTE_POS] = a_val + b_val; 
    //c[ABSOLUTE_POS] = Line::new(1i64) + 1i64;
}

pub fn launch_hp<R: Runtime>(device: &R::Device, a: &Vec<Vec<i64>>, b: &Vec<Vec<i64>>) {
    println!("entered launch function");
    let client = R::client(device);

    //reshape vecs and send them to gpu memory and intialize output memory
    let a_dims = matrix_dimensions(a);
    let b_dims = matrix_dimensions(b);
    let a_line_vec: Vec<u8> = vec_to_flat_u8_vec(a);
    let b_line_vec: Vec<u8> = vec_to_flat_u8_vec(b);
    let max_len = max(a_line_vec.len(), b_line_vec.len());
    let c_handle = client.empty(max_len); //* core::mem::size_of::<i64>());
    let a_handle = client.create(&a_line_vec);
    let b_handle = client.create(&b_line_vec);

    //vectorization = 
    let vectorization: u32 = 1;

    unsafe {
        hadamard_prod::launch::<R>(
            // pass in ComputeClient with generic R for the runtime
            &client,
            // define a single workgroup
            CubeCount::Static(1, 1, 1), 
            // if vec = 1, then define the same # of threads as the length of the flattened matrix 
            CubeDim::new(max_len as u32 / vectorization, 1, 1),
            // the three handles for the input and output memory in the gpu; our kernel params
            ArrayArg::from_raw_parts::<i64>(&a_handle, a_line_vec.len(), vectorization as u8),
            ArrayArg::from_raw_parts::<i64>(&b_handle, b_line_vec.len(), vectorization as u8),
            ArrayArg::from_raw_parts::<i64>(&c_handle, max_len.clone(), vectorization as u8),
        );
    }

    let c_bytes = client.read_one(c_handle.binding());
    let c_output = i64::from_bytes(&c_bytes);

    let b_bytes = client.read_one(b_handle.binding());
    let b_output = i64::from_bytes(&b_bytes);

    let a_bytes = client.read_one(a_handle.binding());
    let a_output = i64::from_bytes(&a_bytes);

    println!("a = {:?}", a_output);
    println!("b = {:?}", b_output);
    println!("c = {:?}", c_output);

}

