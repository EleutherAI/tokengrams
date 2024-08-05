use pyo3::prelude::*;

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum TokenType {
    U16,
    U32,
}