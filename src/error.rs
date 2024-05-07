use thiserror::Error;

#[derive(Error, Debug)]
pub enum TruncatedSignatureParamsError {
    #[error("The order must be strictly positive >= 1.")]
    OrderNonPositive,
    #[error("The number of dimension must be stricly positive")]
    PathDimensionNonPositive,
    #[error("Number of threads must be strictly positive")]
    NThreadNonPositive,
}

#[derive(Error, Debug)]
pub enum TruncatedSignatureError {
    #[error("Incompatible dimension of the two signature's path")]
    IncompatibleDimensions,
    #[error("Incompatible order of the two signatures")]
    IncompatibleOrders,
    #[error(transparent)]
    BaseCrate(#[from] TruncatedSignatureParamsError)
}
