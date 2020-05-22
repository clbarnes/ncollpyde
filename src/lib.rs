mod utils;

#[cfg(not(test))]
mod interface;

#[cfg(not(test))]
pub use crate::interface::ncollpyde;
