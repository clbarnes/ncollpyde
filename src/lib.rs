use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn get_42() -> usize {
    42
}

#[pymodule]
fn ncollpyde(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(get_42))?;

    Ok(())
}
