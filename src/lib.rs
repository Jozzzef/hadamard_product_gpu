use cubecl::prelude::*;

#[cube]
fn hadamard_prod() -> i32 {
    i32::new(comptime!(1+1))
}

pub fn launch<R: Runtime>(device: &R::Device) {
    println!("entered launch function");
    let client = R::client(device);
    println!("{:?}", client);
}

