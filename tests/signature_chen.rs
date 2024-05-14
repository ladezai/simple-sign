use simple_sign::*;
use ndarray::prelude::*;
use std::panic::catch_unwind;

#[test]
fn check_position_order() {
    // checks that the position of the 
    let t_order : usize = 2;
    let dim : usize = 3;
    let position_order : usize = TruncatedSignatureValidParams::position_order(&dim, &t_order);
    let position_order_old : usize = (1.. (t_order as u32)).map(|v| (&dim).pow(v as u32)).sum();

    assert_eq!(position_order, position_order_old);
    
    let t_order : usize = 0;
    let dim : usize = 3;
    let position_order = catch_unwind(|| { 
        TruncatedSignatureValidParams::position_order(&dim, &t_order)
        });
    
    assert!(position_order.is_err());
    
    let t_order : usize = 12;
    let dim : usize = 3;
    let position_order : usize = TruncatedSignatureValidParams::position_order(&dim, &t_order);
    let position_order_old : usize = (1.. (t_order as u32)).map(|v| (&dim).pow(v as u32)).sum();
    
    assert_eq!(position_order, position_order_old);

    let t_order : usize = 1;
    let dim : usize = 1;
    let position_order : usize = TruncatedSignatureValidParams::position_order(&dim, &t_order);
    let position_order_old : usize = (1.. (t_order as u32)).map(|v| (&dim).pow(v as u32)).sum();

    assert_eq!(position_order, position_order_old);

    let t_order : usize = 2;
    let dim : usize = 1;
    let position_order : usize = TruncatedSignatureValidParams::position_order(&dim, &t_order);
    let position_order_old : usize = (1.. (t_order as u32)).map(|v| (&dim).pow(v as u32)).sum();

    assert_eq!(position_order, position_order_old);
}

#[test]
fn check_truncated_signature_params() {
    let err = TruncatedSignatureParams::new(0_usize, 2_usize).check(); 
    assert!(err.is_err());
    let err = TruncatedSignatureParams::new(2_usize, 0_usize).check(); 
    assert!(err.is_err());
}

#[test]
fn check_truncated_signature_fit() {
    let data : Array<f32, Ix1> = array![0., 0.];
    let example : TruncatedSignature<f32> = TruncatedSignatureParams::new(2_usize, 3_usize)
                .check()
                .unwrap()
                .fit(&data)
                .unwrap();
    let verified_sig : TruncatedSignature<f32> = TruncatedSignature::new(2_usize, 
                                                3_usize,
                                                array![0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]);
    println!("{:}", example.signature());
    assert!((example-verified_sig).unwrap().norm_max() < 1e-6);


    // verify correctness for very simple data
    let data : Array<f32, Ix1> = array![1., 1.];
    let example : TruncatedSignature<f32> = TruncatedSignatureParams::new(2_usize, 2_usize)
                .check()
                .unwrap()
                .fit(&data)
                .unwrap();
    let verified_sig : TruncatedSignature<f32> = TruncatedSignature::new(2_usize, 
                                                2_usize,
                                                array![1., 1., 0.5, 0.5, 0.5, 0.5]);
    assert!((example-verified_sig).unwrap().norm_max() < 1e-6);

    // Check if the fit function checks the correct input dimension
    let data : Array<f32, Ix1> = array![1., 2., 3.];
    let err  = TruncatedSignatureParams::new(2_usize, 2_usize)
                .check()
                .unwrap()
                .fit(&data);
    
    assert!(err.is_err());

    // A simple check that the everything works fine in case of 
    // a higher order truncation with suitable data points.
    let data : Array<f32, Ix1> = array![-1., -2., 5.];
    let nerr = TruncatedSignatureParams::new(3_usize, 10_usize)
                    .check()
                    .unwrap()
                    .fit(&data);
    assert!(nerr.is_ok()); 
}

#[test]
fn check_fit_from_points() {
    // Here we check that fitting points by points via the fit_from_points
    // and fitting wrt the distance between start-end point is the same in case
    // of a 2D straight line path.
    let p1 : Array<f32, Ix1> = array![0., 0.];
    let p2 : Array<f32, Ix1> = array![1., 3.];
    let p3 : Array<f32, Ix1> = array![2., 6.];
    let sig1 = TruncatedSignatureParams::new(2_usize, 2_usize)
                .check()
                .unwrap()
                .fit_from_points(&vec![p1, p2, p3])
                .unwrap();
    
    let dp1p3 : Array<f32, Ix1> = array![2., 6.];
    let sig2 =  TruncatedSignatureParams::new(2_usize, 2_usize)
                .check()
                .unwrap()
                .fit(&dp1p3)
                .unwrap();
    
    assert!((sig1-sig2).unwrap().norm_max() < 1e-4, "The distance in max norm is quite large!");

    let points : Vec<Array<f32, Ix1>> = vec![array![0.,0.], array![1., 1.], array![2.,2.]];
    let example : TruncatedSignature<f32> = TruncatedSignatureParams::new(2_usize, 2_usize)
                    .check()
                    .unwrap()
                    .fit_from_points(&points)
                    .unwrap();
    let result : TruncatedSignature<f32> = TruncatedSignature::new(2_usize, 
                                                                2_usize,
                                                                array![2., 2., 2., 2., 2., 2.]);
    println!("{:}", example.signature());
    println!("{:}", result.signature());

    assert!((example-result).unwrap().norm_max() < 1e-6);
}

#[test]
fn check_truncated_signature_chens_addition() {
    // verify correctness for very simple data
    let data1 : Array<f32, Ix1> = array![1., 1.];
    let data2 : Array<f32, Ix1> = array![1., 1.];
    let example1 : TruncatedSignature<f32> = TruncatedSignatureParams::new(2_usize, 2_usize)
                .check()
                .unwrap()
                .fit(&data1)
                .unwrap();
    let example2 : TruncatedSignature<f32> = TruncatedSignatureParams::new(2_usize, 2_usize)
                .check()
                .unwrap()
                .fit(&data2)
                .unwrap();
    let example_res : TruncatedSignature<f32> = example1.chens_addition(&example2).unwrap();

    let verified_sig : TruncatedSignature<f32> = TruncatedSignature::new(2_usize, 
                                                2_usize,
                                                array![2., 2., 2., 2., 2., 2.]);
    assert!((example_res-verified_sig).unwrap().norm_max() < 1e-6);

}
