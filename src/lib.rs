use cubecl::prelude::*;
use std::cmp::max;

fn matrix_dimensions(matrix: &[Vec<i64>]) -> (u32, u32){
    (matrix.len() as u32, matrix[0].len() as u32)
}

fn vec_to_flat_u8_vec(matrix: &[Vec<i64>]) -> Vec<u8> {
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
    let c_rows: u32 = c.shape(0);
    let c_cols: u32 = c.shape(1);
    let mut a_val = Line::new(0i64);
    let mut b_val = Line::new(0i64);
    if ABSOLUTE_POS_Y < a_rows && ABSOLUTE_POS_X < a_cols {
        a_val = a[ABSOLUTE_POS_Y * a_cols + ABSOLUTE_POS_X]
    }
    if ABSOLUTE_POS_Y < b_rows && ABSOLUTE_POS_X < b_cols {
        b_val = b[ABSOLUTE_POS_Y * b_cols + ABSOLUTE_POS_X]
    }
    if ABSOLUTE_POS_Y < c_rows && ABSOLUTE_POS_X < c_cols {
        c[ABSOLUTE_POS_Y * c_cols + ABSOLUTE_POS_X] = a_val + b_val;
    }
}

pub fn launch_hp<R: Runtime>(
        device: &R::Device, 
        a: &[Vec<i64>], 
        b: &[Vec<i64>]
) -> Vec<Vec<i64>> {
    let client = R::client(device);

    //reshape vecs and prepare to be sent to gpu memory 
    let (a_rows, a_cols) = matrix_dimensions(a);
    let (b_rows, b_cols) = matrix_dimensions(b);
    let a_line_vec: Vec<u8> = vec_to_flat_u8_vec(a);
    let b_line_vec: Vec<u8> = vec_to_flat_u8_vec(b);
    let max_len = max(a_line_vec.len(), b_line_vec.len());
    let max_rows = max(a_rows, b_rows);
    let max_cols = max(a_cols, b_cols);

    let vectorization: u32 = 1; // >1 causing bug for Tensor type
    // split up the data in workgroups of 256 threads per group
    let workgroup_size = 256;
    let square_workgroup_dims = (workgroup_size as f64).sqrt() as u32;
    let (x_work, y_work, z_work): (u32, u32, u32) = (
        // div_ceil() rounds up to the next integer, this will give us enough workgroups to cover
        // all elements needed, leaving some idle typically
        max_cols.div_ceil(square_workgroup_dims), 
        max_rows.div_ceil(square_workgroup_dims),
        1
    );
    let max_cube_count = client.properties().hardware_properties().max_cube_count;
    let max_2d_threads = client.properties().hardware_properties().max_units_per_cube;
    println!("max threads in a workgroup = {}", max_2d_threads);
    println!("max workgroups = {:?}", max_cube_count);
    println!("workgroups in usage = {}", x_work);

    //allocate memory
    let c_handle = client.empty(max_len); //* core::mem::size_of::<i64>());
    let a_handle = client.create(&a_line_vec);
    let b_handle = client.create(&b_line_vec);

    unsafe {
        hadamard_prod::launch::<R>(
            // pass in ComputeClient with generic R for the runtime
            &client,
            // define a single workgroup
            CubeCount::Static(
                x_work, 
                y_work, 
                z_work
            ), 
            // if vec = 1, then define the same # of threads as the length of the flattened matrix 
            CubeDim::new(
                square_workgroup_dims, 
                square_workgroup_dims, 
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

    //convert back to Vec<Vec<i64>>
    let c_vec: Vec<Vec<i64>> = c_output.chunks(max_cols as usize)
        .map(|chunk| chunk.to_vec())
        .collect();

    // let b_bytes = client.read_one(b_handle.binding());
    // let b_output = i64::from_bytes(&b_bytes);

    // let a_bytes = client.read_one(a_handle.binding());
    // let a_output = i64::from_bytes(&a_bytes);

    // println!("a = {:?}", a_output);
    // println!("b = {:?}", b_output);
    // println!("c = {:?}", c_output);

    c_vec
}

