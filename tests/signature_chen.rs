use simple_sign::*;
use ndarray::prelude::*;

#[test]
fn check_truncated_signature_params() {
    let err = TruncatedSignatureParams::new(0_usize, 2_usize).check(); 
    assert!(err.is_err());
    let err = TruncatedSignatureParams::new(2_usize, 0_usize).check(); 
    assert!(err.is_err());
    
}

#[test]
fn check_truncated_signature_fit() {
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
                .fit_from_points(vec![p1, p2, p3])
                .unwrap();
    
    let dp1p3 : Array<f32, Ix1> = array![2., 6.];
    let sig2 =  TruncatedSignatureParams::new(2_usize, 2_usize)
                .check()
                .unwrap()
                .fit(&dp1p3)
                .unwrap();
    
    assert!((sig1-sig2).unwrap().norm_max() < 1e-4, "The distance in max norm is very high!");
}

#[test]
fn check_truncated_signature_chens_addition() {
    // TODO
}
