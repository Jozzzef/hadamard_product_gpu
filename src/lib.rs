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

#[cube]
fn hadamard_prod(
        a: &Array<Line<i64>>, 
        b: &Array<Line<i64>>, 
        c: &mut Array<Line<i64>>
) {
    todo!();
}

pub fn launch<R: Runtime>(device: &R::Device, a: &Vec<Vec<i64>>, b: &Vec<Vec<i64>>) {
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
    let vectorization = 4;

    //unsafe {
    //    hadamard_prod(a, b, c);
    //}

    let c_bytes = client.read_one(c_handle.binding());
    let c_output = i64::from_bytes(&c_bytes);

    let b_bytes = client.read_one(b_handle.binding());
    let b_output = i64::from_bytes(&b_bytes);

    println!("c = {:?}", c_output);
    println!("b = {:?}", b_output);

}

