use ndarray::prelude::*;
use ndarray::ScalarOperand;
use ndarray::linalg::kron;
use num_traits::{Float, FromPrimitive};
use crate::error::{TruncatedSignatureParamsError, TruncatedSignatureError}; 

use core::fmt::{Display, Debug};

#[derive(Clone, Copy, Debug)]
pub enum Algorithm {
    ChenAddition
}

#[derive(Clone, Debug)]
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

    pub fn position_order(dimension : &usize, order : &usize) -> usize {
       (1.. (*order as u32)).map(|v| dimension.pow(v as u32)).sum()
    }

    pub fn fit<F : Float + FromPrimitive + 'static>(&self, data : Array1<F>) -> Result<TruncatedSignature<F>, TruncatedSignatureError> {
        //println!("{:?}", data);
        // NOTE: we can do something better here by using the good old trick of 
        // sum_k 1/d^k =  (d^{M+1} - 1)/ (d-1)
        let max_idx : usize = TruncatedSignatureValidParams::position_order(&self.dimension, &(self.order + 1));
        let mut signature : Array1<F> = Array1::zeros(max_idx);


        let m = data.shape()[0];
        let matrix_data : Array2<F>  = data.into_shape((m,1)).unwrap();
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
            //if checks {
                // TODO: add additional checks to shapes
                // if a check is not working, 
                // return a TruncatedSignatureError...
            //}

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
            order : self.order, 
            dimension : self.dimension, 
            signature : signature} ) 

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
            return Err(TruncatedSignatureParamsError::NThreadNonPositive)
        } else if self.0.order < 1 {
            return Err(TruncatedSignatureParamsError::OrderNonPositive)
        } else if self.0.dimension < 1 {
            return Err(TruncatedSignatureParamsError::PathDimensionNonPositive)
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
    order : usize,
    dimension : usize,
    signature : Array<F, Ix1>
} 

impl<F : Float + Debug + Display + ScalarOperand + FromPrimitive + 'static> TruncatedSignature<F> {
    pub fn params(order : usize, dimension : usize) -> TruncatedSignatureParams {
        TruncatedSignatureParams::new(order, dimension)
    }

    
    ///
    /// Returns a flatten view of the signature of the specified order.
    ///
    pub fn signature_order_flatten(&self, order : usize) -> ArrayView<F, Ix1> {
        let initial_idx : usize = TruncatedSignatureValidParams::position_order(&self.dimension, &order);
        let end_idx     : usize = initial_idx + self.dimension.pow(order as u32);

        self.signature.slice(s![initial_idx.. end_idx])
    }

    pub fn chens_addition(mut self, other : &TruncatedSignature<F>) -> Result<TruncatedSignature<F>, TruncatedSignatureError> {
        //pub fn chens_addition(&self, other : &TruncatedSignature<F>) -> TruncatedSignature<F> {
        //
        // TODO: check that order are the same, otherwise use the minimum between the two.
        // check that the dimensions are the same, otherwise return an error!
        //
        if self.order != other.order {
            return Err(TruncatedSignatureError::IncompatibleOrders)
        } 
        if self.dimension != other.dimension {
            return Err(TruncatedSignatureError::IncompatibleDimensions)
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
                            + &other.signature_order_flatten(i) 
                            + &self.signature_order_flatten(i);
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