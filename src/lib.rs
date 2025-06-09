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
        a: &Tensor<Line<i64>>, 
        b: &Tensor<Line<i64>>, 
        c: &mut Tensor<Line<i64>>,
) {
    let a_rows: u32 = a.shape(0);
    let a_cols: u32 = a.shape(1);
    let b_rows: u32 = b.shape(0);
    let b_cols: u32 = b.shape(1);
    let c_cols: u32 = c.shape(1);
    let mut a_val = Line::new(0i64);
    let mut b_val = Line::new(0i64);
    if ABSOLUTE_POS_X < a_rows && ABSOLUTE_POS_Y < a_cols {
        a_val = a[ABSOLUTE_POS_X * a_cols + ABSOLUTE_POS_Y]
    }
    if ABSOLUTE_POS_X < b_rows && ABSOLUTE_POS_Y < b_cols {
        b_val = b[ABSOLUTE_POS_X * b_cols + ABSOLUTE_POS_Y]
    }
        
    c[ABSOLUTE_POS_X * c_cols + ABSOLUTE_POS_Y] = a_val + b_val;
}

pub fn launch_hp<R: Runtime>(device: &R::Device, a: &Vec<Vec<i64>>, b: &Vec<Vec<i64>>) {
    println!("entered launch function");
    let client = R::client(device);

    //reshape vecs and send them to gpu memory and intialize output memory
    let (a_rows, a_cols) = matrix_dimensions(a);
    let (b_rows, b_cols) = matrix_dimensions(b);
    let a_line_vec: Vec<u8> = vec_to_flat_u8_vec(a);
    let b_line_vec: Vec<u8> = vec_to_flat_u8_vec(b);
    let max_len = max(a_line_vec.len(), b_line_vec.len());
    let max_rows = max(a_rows, b_rows);
    let max_cols = max(a_cols, b_cols);
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
            CubeDim::new(
                max_rows as u32 / vectorization, 
                max_cols as u32 / vectorization, 
                1),
            // the three handles for the input and output memory in the gpu; our kernel params
            TensorArg::from_raw_parts::<i64>(
                &a_handle, //gpu memory location pointer
                &[a_cols as usize, 1], // strides for row major matrix format
                &[a_rows as usize, a_cols as usize], // shape of the original matrix
                vectorization as u8), // number of elements to go through each thread
            TensorArg::from_raw_parts::<i64>(
                &b_handle, //gpu memory location pointer
                &[b_cols as usize, 1], // strides for row major matrix format
                &[b_rows as usize, b_cols as usize], // shape of the original matrix
                vectorization as u8), // number of elements to go through each thread
            TensorArg::from_raw_parts::<i64>(
                &c_handle, //gpu memory location pointer
                &[max_cols as usize, 1], // strides for row major matrix format
                &[max_rows as usize, max_cols as usize], // shape of the original matrix
                vectorization as u8), // number of elements to go through each thread
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

