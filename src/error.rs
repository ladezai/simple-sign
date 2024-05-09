use thiserror::Error;

#[derive(Error, Debug)]
pub enum TruncatedSignatureParamsError {
    #[error("Truncation order ({0}) must be strictly positive.")]
    OrderNonPositive(usize),
    #[error("Dimensions ({0}) must be stricly positive")]
    PathDimensionNonPositive(usize),
    #[error("Number of threads ({0}) must be strictly positive")]
    NThreadNonPositive(usize),
    #[error("The algorithm is not implemented")]
    AlgorithmNotImplemented,    
}

#[derive(Error, Debug)]
pub enum TruncatedSignatureError {
    #[error("The data has dimension has incompatible dimension with the signature. The expected dimension of the data is {signature_dim:?}, but {data_dim:?} was found.")]
    DataIncompatibleDimension {
        data_dim : usize, 
        signature_dim : usize,
    },
    #[error("Incompatible dimension of the two signatures. The first has dimension {d_path1:?}, the second {d_path2:?}")]
    IncompatibleDimensions { 
        d_path1 : usize, 
        d_path2 : usize 
    },
    #[error("Incompatible order of the two signatures. The first has order {order1:?}, the second {order2:?}")]
    IncompatibleOrders {
        order1 : usize,
        order2 : usize
    },
    #[error(transparent)]
    Params(#[from] TruncatedSignatureParamsError)
}
