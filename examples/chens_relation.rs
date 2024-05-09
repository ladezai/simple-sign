use ndarray::prelude::*;
use simple_sign::{TruncatedSignature, TruncatedSignatureParams, TruncatedSignatureError};

pub fn impure_1() -> Result<TruncatedSignature<f32>, TruncatedSignatureError> {
    let data1 = array![2., 1.];
    let data2 = array![4., 2.];
    let sig1 : TruncatedSignature<f32> = TruncatedSignatureParams::new(2_usize, 3_usize)
                            .check()?
                            .fit(&data1)?;
    let sig2 : TruncatedSignature<f32> = TruncatedSignatureParams::new(2_usize, 3_usize)
                            .check()?
                            .fit(&data2)?;
    let sig1 = (sig1).chens_addition(&sig2)?;
    Ok(sig1)
}

pub fn impure_2() -> Result<TruncatedSignature<f32>, TruncatedSignatureError> {
    let data = array![6., 3.];
    let sig1 : TruncatedSignature<f32> = TruncatedSignatureParams::new(2_usize, 3_usize)
                            .check()?
                            .fit(&data)?;
    Ok(sig1)
}

pub fn check_distance_l1(sig1 : TruncatedSignature<f32>, sig2 : TruncatedSignature<f32>) -> f32 {
    (sig1 - sig2).unwrap().norm_l1()
}


pub fn main() -> () {
    println!("In this example we consider the simple path W_s = (2s, s).");
    println!("We consider the points W_0 = (0,0), W_1 = (2,1), W_3=(6,3).");
    println!("By Chen's relation S(W_[0,1]) \\otimes S(W_[1,3]) = S(W_[0,3])");
    println!("Therefore, we compute the signature using the LHS in the first step...");

    let sig1 = impure_1().unwrap(); 
    println!("{:?}", &sig1);

    println!("And using the RHS, we get the same result: ");
    let sig2 = impure_2().unwrap(); 
    println!("{:?}", &sig2);
    println!("Chen's relation seem to hold...");

    let l1_dst = check_distance_l1(sig1, sig2);
    println!("The difference in l1-norm is: {:?}", l1_dst);
}

