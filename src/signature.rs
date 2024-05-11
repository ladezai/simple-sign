use ndarray::prelude::*;
use ndarray::linalg::{kron};
use num_traits::{Float, FromPrimitive};
use crate::error::{TruncatedSignatureParamsError, TruncatedSignatureError}; 

use core::fmt::{Debug};
use std::ops::{Add, Mul, Sub, Div};

#[derive(Clone, Copy, Debug)]
///
/// Algorithm used to compute the Signature of a path W with respect to
/// multiple points. 
/// Available algorithms:
/// * ChenRelation: Iteratively fits at each W_{t_i}  - W_{t_{i+1}} and then takes the product of the
/// signatures S(W_{[t_i, t_{i+1}]}) \otimes S(W_{[t_{i+1}, t_{i+2}]}).
///
pub enum Algorithm {
    ChenAddition
}

#[derive(Clone, Debug)]
///
/// The data for the fitting procedure. 
/// dimension : usze = data dimension. 
/// order : usize = signature truncation order. 
/// method : Algorithm = algorithm to use for the fitting procedure.
/// n_thread : usize = number of threads to spawn during the fitting procedure.
///
pub struct TruncatedSignatureValidParams {
    // Dimension of the space the path belongs to.
    dimension : usize, 
    // Truncation order.
    order : usize,
    // algorithm choice...
    method : Algorithm,
    // n_thread..
    n_thread : usize
}

impl TruncatedSignatureValidParams {
    // Dimension of the space the path belongs to.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    // Truncation order.
    pub fn order(&self) -> usize {
        self.order
    }

    // algorithm choice...
    pub fn method(&self) -> Algorithm {
        self.method
    }

    // n_thread..
    pub fn n_thread(&self) -> usize {
        self.n_thread
    }

    /// Computes 
    ///  $dimension * (dimension^order - 1)  / (dimension - 1)$
    /// which corresponds to the index of the view of the 
    /// component of the tensor dimension^order.
    /// 
    pub fn position_order(dimension : &usize, order : &usize) -> usize {
        //(1.. (*order as u32)).map(|v| dimension.pow(v as u32)).sum()
        if *order == 0_usize || *dimension == 0_usize {
            panic!("Either order ({}) or the dimension ({}) is non-positive", order, dimension);
        }

        if *order == 1_usize {
            return 0_usize;
        }

        if *dimension == 1_usize {
            return *order - 1;
        }

        (dimension * (dimension.pow((order-1) as u32) - 1)).div(dimension-1)
    }

    pub fn fit<F : Float + FromPrimitive + 'static>(&self, data : &Array1<F>) -> Result<TruncatedSignature<F>, TruncatedSignatureError> {
        //println!("{:?}", data);
        let max_idx : usize = TruncatedSignatureValidParams::position_order(&self.dimension, &(self.order + 1));
        let mut signature : Array1<F> = Array1::zeros(max_idx);
        let m : usize = data.shape()[0];
        if m != self.dimension() {
            return Err(TruncatedSignatureError::DataIncompatibleDimension {
                data_dim : m,
                signature_dim : self.dimension()
            });
        }
        let matrix_data  : Array2<F>  = data.to_owned().into_shape((m,1)).unwrap();
        //println!("{:?}", matrix_data);
        let mut kron_mul : Array2<F> = matrix_data.to_owned(); 
        // normalization factor
        let mut fact : f64 = 1.0;
        for i in 1.. (self.order+1) {
            let size         : usize  = (&self).dimension.pow(i as u32);
            let initial_idx  : usize  = TruncatedSignatureValidParams::position_order(&(self.dimension), &i);
            //println!("order: {}, size: {}, initial_idx: {}", &i, &size, &initial_idx);
            //
            fact = fact / (i as f64);
            let flattened_kron = Array::from_iter(kron_mul.mapv(|v| v * F::from_f64(fact).unwrap())
                                                  .iter()
                                                  .cloned());
            //println!("flattened_kron shape: {:?}", flattened_kron.shape());
            // assigns the value of the i-th order signature.
            signature
                .slice_mut(s![initial_idx.. (initial_idx+size)])
                .assign(&flattened_kron);

            //println!("signature linear array shape: {:?}", self.signature.shape());
            // do not compute the last step as it is useless, we truncate
            // before that to be stored.
            if i < self.order {
                kron_mul = kron(&kron_mul, &matrix_data);
            }
        }

        Ok(TruncatedSignature::<F> {
            order     : self.order, 
            dimension : self.dimension, 
            signature : signature
        }) 
    }
    
    /// 
    /// Generates a TruncatedSignature given points on a path.
    ///
    pub fn fit_from_points<F : Float + FromPrimitive + 'static>(&self, data : Vec<Array1<F>>) -> Result<TruncatedSignature<F>, TruncatedSignatureError> {
        // switch between different methods to compute the signature
        match self.method() {
            Algorithm::ChenAddition => {
                // computes the increment vectors between the given data points
                let delta_data : Vec<Array1<F>> = data.iter()
                                                .zip(data.iter().skip(1))
                                                .map(|(v1, v2)| v2-v1)
                                                .collect();
                // computes the 0 element for the given signature 
                let zero_elem : TruncatedSignature<F> = self.fit(&Array1::zeros(self.dimension()))?;
                // Iteratively fit on the next increment vector, then add it via 
                // chens relation.
                delta_data.iter().fold(Ok(zero_elem), |acc, v| {
                        let fit = self.fit(v)?;
                        acc?.chens_addition(&fit)
                })
           },
            _ => Err(TruncatedSignatureError::Params(TruncatedSignatureParamsError::AlgorithmNotImplemented))
        }
    }
}

#[derive(Clone, Debug)]
pub struct TruncatedSignatureParams(TruncatedSignatureValidParams);

impl TruncatedSignatureParams {
    pub fn new(dimension : usize, order : usize) -> Self {
        Self(TruncatedSignatureValidParams { 
            dimension : dimension,
            order : order,
            method : Algorithm::ChenAddition,
            n_thread : 1
        })
    }
    
    pub fn n_thread(mut self, n_thread : usize) -> Self {
        self.0.n_thread = n_thread;
        self
    }

    pub fn algorithm(mut self, method : Algorithm) -> Self {
        self.0.method = method;
        self
    }

    pub fn dimension(mut self, dimension : usize) -> Self {
        self.0.dimension = dimension;
        self
    }

    pub fn order(mut self, order : usize) -> Self {
        self.0.order = order;
        self
    }

    pub fn check_ref(&self) -> Result<&TruncatedSignatureValidParams, TruncatedSignatureParamsError> {
        if self.0.n_thread < 1 {
            return Err(TruncatedSignatureParamsError::NThreadNonPositive(self.0.n_thread()))
        } else if self.0.order < 1 {
            return Err(TruncatedSignatureParamsError::OrderNonPositive(self.0.order()))
        } else if self.0.dimension < 1 {
            return Err(TruncatedSignatureParamsError::PathDimensionNonPositive(self.0.dimension()))
        }

        Ok(&self.0)
    }

    pub fn check(self) -> Result<TruncatedSignatureValidParams, TruncatedSignatureParamsError> {
        self.check_ref()?;
        Ok(self.0)
    }

}


#[derive(Debug)]
pub struct TruncatedSignature<F : Float> {
    order     : usize,
    dimension : usize,
    signature : Array<F, Ix1>
} 

impl<F : Float> TruncatedSignature<F> {
    pub fn new(order : usize, dimension : usize, signature : Array<F, Ix1>) -> TruncatedSignature<F> {
        Self {
            order : order,
            dimension : dimension,
            signature : signature
        }
    }
    pub fn params(order : usize, dimension : usize) -> TruncatedSignatureParams {
        TruncatedSignatureParams::new(order, dimension)
    }

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    ///
    /// Returns the flatten view of the signature for each component.
    ///
    pub fn signature(&self) -> ArrayView<F, Ix1> {
        self.signature.view()
    }

    ///
    /// Returns a flatten view of the signature of the specified order.
    ///
    pub fn signature_order_flatten(&self, order : usize) -> ArrayView<F, Ix1> {
        let initial_idx : usize = TruncatedSignatureValidParams::position_order(&self.dimension, &order);
        let end_idx     : usize = initial_idx + self.dimension.pow(order as u32);

        self.signature.slice(s![initial_idx.. end_idx])
    }

    
}
impl<F : Float + FromPrimitive + 'static> TruncatedSignature<F> {

    pub fn chens_addition(mut self, other : &TruncatedSignature<F>) -> Result<TruncatedSignature<F>, TruncatedSignatureError> {
        //
        // TODO: check that order are the same, otherwise use the minimum between the two.
        // check that the dimensions are the same, otherwise return an error!
        //
        if self.order != other.order {
            return Err(TruncatedSignatureError::IncompatibleOrders { 
                order1 : self.order(), 
                order2 : other.order()
            });
        } 
        if self.dimension != other.dimension {
            return Err(TruncatedSignatureError::IncompatibleDimensions {
                d_path1 : self.dimension(), 
                d_path2 : other.dimension()
            });
        }

        let max_order : usize = self.order;
        let dim       : usize = self.dimension;
        let mut result_signature = self.signature.to_owned();
        
        for i in 1.. (max_order+1) {
            let size         : usize  = self.dimension.pow(i as u32);
            let initial_idx  : usize  = TruncatedSignatureValidParams::position_order(&self.dimension, &i);
            //println!("order: {}, size: {}, initial_idx: {}", &i, &size, &initial_idx);
            let mut orderi_tensor = result_signature
                                    .slice_mut(s![initial_idx.. (initial_idx+size)]);
            let mut orderi_tensor_owned = orderi_tensor.to_owned();
            orderi_tensor_owned = orderi_tensor_owned
                            + &other.signature_order_flatten(i);
            //println!("The signature of order {}, adds: \n{}\n and... \n{}.\n", i, &other.signature_order_flatten(i), &self.signature_order_flatten(i));
            for j in 1.. i {
                let imj : usize = i-j;
                let a = self.signature_order_flatten(j)
                            .into_shape((dim.pow(j as u32), 1))
                            .unwrap();
                let b = other.signature_order_flatten(imj)
                             .into_shape((dim.pow(imj as u32), 1))
                             .unwrap();
                // add the tensor product represented as a kronecker delta 
                // between the two vectors.
                orderi_tensor_owned = orderi_tensor_owned 
                            + &Array::from_iter(kron(&a, &b).iter().cloned());
            }
            // finally assing to our result vector.
            orderi_tensor.assign(&orderi_tensor_owned);

            //println!("The signature is now: \n{:?}\n", result_signature);
        } 
        self.signature = result_signature;
        Ok(self)
    }
}

// ########################################
// BASIC OPERATION IMPLEMENTATION
// #######################################
impl<F : Float> Add for TruncatedSignature<F> {
    type Output = Result<Self, TruncatedSignatureError>;
    fn add(self, other : Self) -> Self::Output {

        if self.order() != other.order() {
            return Err(TruncatedSignatureError::IncompatibleOrders {
                order1 : self.order(), 
                order2 : other.order()
            });
        }
        if self.dimension() != other.dimension() {
            return Err(TruncatedSignatureError::IncompatibleDimensions {
                d_path1 : self.dimension(),
                d_path2 : other.dimension()
            });
        }
        
        Ok(Self {
            order : self.order(),
            dimension : self.dimension(),
            signature : self.signature + other.signature
        })
    }
}

impl<F : Float> Sub for TruncatedSignature<F> {
    type Output = Result<Self, TruncatedSignatureError>;
    fn sub(self, other : Self) -> Self::Output {

        if self.order() != other.order() {
            return Err(TruncatedSignatureError::IncompatibleOrders {
                order1 : self.order(), 
                order2 : other.order()
            });
        }
        if self.dimension() != other.dimension() {
            return Err(TruncatedSignatureError::IncompatibleDimensions {
                d_path1 : self.dimension(),
                d_path2 : other.dimension()
            });
        }
        
        Ok(Self {
            order : self.order(),
            dimension : self.dimension(),
            signature : self.signature - other.signature
        })
    }
}

impl<F : Float> Mul for TruncatedSignature<F> {
    type Output = Result<Self, TruncatedSignatureError>;
    fn mul(self, other : Self) -> Self::Output {
        if self.order() != other.order() {
            return Err(TruncatedSignatureError::IncompatibleOrders {
                order1 : self.order(), 
                order2 : other.order()
            });
        }
        if self.dimension() != other.dimension() {
            return Err(TruncatedSignatureError::IncompatibleDimensions {
                d_path1 : self.dimension(),
                d_path2 : other.dimension()
            });
        }
        
        Ok(Self {
            order : self.order(),
            dimension : self.dimension(),
            signature : self.signature * other.signature
        })
    }
}

// NORMS
impl<F : Float> TruncatedSignature<F> {
    pub fn norm_l1(&self) -> F {
        self.signature.mapv(|v| v.abs()).sum()
    }

    pub fn norm_max(&self) -> F {
        self.signature.fold(F::neg_infinity(), |acc, v| 
            if v.abs() > acc { 
                v.abs() 
            } else { 
                acc 
            )
        )
    }

}

